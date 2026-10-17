# 场景得分阻塞诊断（2026-10 评测复盘）

`4/5/6` 得满分、`1/2/3/7/8` 为 0 分，两组原因不同。结论由代码路径静态核对得出，
现场仍需用 `scripts/diagnose_scene.py` 对真实快照复核。

## 一句话结论

两组 0 分都不是崩溃，而是代码在"输入不足"时按设计**主动停车或不执行任务**，
因此关键任务未完成，按评分规则计 0 分。修复在**平台配置**和**行为模型交付**，
不在算法。

## 4/5/6 为什么能得分

04 直道偏离抑制、05 弯道偏离抑制、06 车道居中只需要 GPS + 地图车道，
不需要目标观测，也不需要行为模型。它们通过全部闸门：

```
ego ✓  lane ✓  targets(非必需) ✓  signal(非必需) ✓  behaviour_model(非必需) ✓
```

这同时证明底座是健康的：感知、决策、规划、控制、发送、评价记录全部可用。

## 1/2/3 为什么 0 分：目标数据缺失，且真值兜底被明确拒绝

调用链（逐层核对）：

1. `core/scene_requirements.py` 把 1/2/3 标记为 `TARGET_SCENES` 且属于 `AEB_SCENES`。
2. `simone_platform/simone_adapter.py:296-305`：Sensor API 取不到目标时回退
   `SoGetGroundTruth`，`target_source` 变成 `ground_truth` 或 `none`。
3. `members/decision/protocol.py:45-55`：`targets_usable()` 要求
   `target_source.startswith("sensor:")`，**Ground Truth 被明确排除**。
4. `members/decision/engine.py:409-421`：`requires_targets` 为真而 `targets_fresh` 为假时
   进入保护，连续两帧后 `STOP_LATCHED`。
5. `members/decision/engine.py:170-179` `_protect()`：运动中输出 `EMERGENCY_BRAKE`，
   静止时输出 `STOP`，目标速度均为 0。

**结果**：车不动或起步即刹停，AEB 任务不执行 → 关键任务未完成 → 0 分。

补充证据：`perception/SCENE_DATA_ACCEPTANCE.md:7` 已记录
"01 的两次输入检查未取得正式 Sensor API 目标包，**平台下发的主车传感器配置及节点为空**"。

### "配置为空"有四种含义，原先被合并成一种

`source_status.targets.sensor_presence` 现在区分（`sensor-discovery-v3`）：

| 取值 | 含义 | 修复方向 |
| --- | --- | --- |
| `configured` | 有可识别的检测源 | 正常，继续查读取 |
| `unclassified_types` | **有传感器，但类型名未被识别**为 camera/lidar/perfect/fusion | 对 `sensor_catalog.sensors[].raw_type` 补类型名映射 |
| `no_detection_sensor` | 只有雷达/GNSS/超声波等非检测源 | 平台需增加检测传感器 |
| `not_configured` | 确实没有任何传感器 | 平台需配置传感器 |

这个区分很重要：`unclassified_types` 的修复成本远低于补装传感器，而原先它会被
误报成"平台没配传感器"，把排查引向错误方向。`sensor_catalog` 字段同时给出
每个传感器的 `id`、`kind` 和 `raw_type`。

**目标读取规则不变**：未识别的类型仍然不会被当作目标源，安全原则没有放松。

### 三条修法

| 方案 | 做法 | 前置条件 |
| --- | --- | --- |
| A 补平台传感器（推荐） | 在 NEVC 平台给主车配置一个 camera/lidar/perfect/fusion 目标传感器 | 平台操作权限 |
| B 允许真值兜底 | 让 `targets_usable()` 接受 `ground_truth` | **队长与赛方确认合规**；属决策组安全原则，需授权 |
| C 对齐传感器 ID 或类型名 | 平台已配但 ID/类型名不匹配时改配置或补映射 | 先看 `sensor_catalog` 的 `raw_type` |

方案 B 不可由单人决定：用真值等于完美感知，是否合规需赛方口径。而且它不是两行改动：
`startswith("sensor:")` 这道门槛在决策、规划、感知可见性、运行监护等 **10 处以上**
同时存在，放开真值等于改全线语义，需整体评审。

## 7/8 为什么 0 分：泊车能力根本没启用

调用链：

1. `config/default.ini:12` `behavior_model_file =` **为空**；仓库内只有
   `config/behavior_model.example.json`（`verified: false`、`scope` 全空）。
2. `behavior_inputs.py:414-418`：路径为空即返回默认 `ConfiguredBehaviorInputs`。
3. `behavior_inputs.py:62`：默认 `parking_provider = None`、`parking_actions = ()`。
4. `members/decision/behaviors/parking_runtime.py:70-72`：
   `if self.provider is None and not self.pending(): return False`。
5. `members/decision/engine.py:105`：泊车分支不成立，走 `_arbitrate()` 普通分支。

**结果**：车按车道保持行驶，从不尝试泊车 → 0 分。

### 波及范围比 7/8 更大

同一闸门还关闭了：

| 能力 | 场景 |
| --- | --- |
| 换道 | 16、17、26、27、28、29 |
| 通行/路口 | 18、30、31、32、33–40 |
| 任务路线 | 41 |

这些场景若也是 0 分，原因完全相同。

### 一个必须知道的运营约束

行为模型的 `scope` 硬绑定单个任务（`behavior_inputs.py:288-289`
`BEHAVIOR_MODEL_TASK_SCOPE_MISMATCH`）：

```python
require((p.case_id, p.task_id) == (scope['case_id'], scope['task_id']), ...)
```

**一份模型文件只覆盖一个 case/task。** 跑场景 7 和场景 8 需要两份不同文件，
每换场景都要重新生成并核对。

## 使用方法

```powershell
# 单份快照
python scripts\diagnose_scene.py --snapshot runtime_data\latest_perception.json

# 一批历史快照（每个场景一份）
python scripts\diagnose_scene.py --directory runtime_data\scene_acceptance --quiet

# 强制按某场景号评估（快照场景号不可信时）
python scripts\diagnose_scene.py --snapshot x.json --scene 7

# 落盘报告
python scripts\diagnose_scene.py --directory runtime_data\scene_acceptance --json-out diagnosis.json
```

输出每场景六个闸门（ego / lane / sensor_configuration / targets / signal /
behaviour_model），并汇总阻塞类型，直接给出修复优先级。

**快照必然过期**：`valid_until` 是进程内单调时钟，保存后一定过期。工具把 ego 闸门
标为 `UNKNOWN` 而不是 `PASS`，不伪造新鲜度。实车新鲜度只能看最新 `captain.log`。

## 工具与生产代码的一致性

`tests/test_diagnose_scene.py` 用**生产谓词**交叉校验诊断结论：

```python
live = protocol.targets_usable(perception) or not requires_targets(perception.scene_id)
self.assertEqual(expected, gate["status"] == PASS)
```

即诊断工具的目标闸门与决策组真实判定绑定，避免报告与实现漂移。行为变更导致两者
不一致时该测试会失败。

## 未验证项

- 本轮只在 Python 3.13 上跑离线测试，**未在 SimOne 的 Python 3.6 上执行**。
- 没有真实快照，工具用构造快照验证各分支；实车结论待现场数据。
- 传感器类型数字编码（`1` camera … `9` fusion）取自仓库自身实现与测试
  （`simone_platform/sensor_catalog.py`、`tests/test_target_ingestion.py`），
  未与官方 C 头文件逐一核对。
- 各场景扣分明细仍需对照 `比赛文件/02.评分与规则02.pdf` 逐条核对车辆动作，
  结构性无阻塞不等于能拿满分。
