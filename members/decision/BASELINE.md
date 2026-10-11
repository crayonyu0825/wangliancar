> 2026-10-12：联合分支的正式运行合同、最小换道接线与当前证据见 [联合实现](JOINT_IMPLEMENTATION.md)。下文是上一轮独立策略/提案的历史记录，不能代替当前接线状态。

> 2026-10-08 基于 `main 568e9c9` 增加 M0/M1 行为基础和有界目标交接；D03～D09 独立策略有代码与合成反馈测试，尚无公共行为/反馈通道。当前交付等级见 [行为实现与能力矩阵](BEHAVIOR_IMPLEMENTATION.md)，合同字段见 [R05/R07 提案](BEHAVIOR_HANDOFF.md)。`decision_behavior_info()` 仅返回未派发提案诊断。

# 通用决策框架（当前实现）

2026-10-04 更新至 `decision-constraints-v7`：共享有界路线事实和运动范围，邻道冲突使用定向包络，危险解除按目标 ID 确认，严格关联地图灯具走交通语义。所有适用信号按独立路线边界与前车共同约束，最近绿灯不能掩盖后续红灯；未来未知范围允许边界前接近，当前无边界仍保护。目标运动支持与规划采用同一走廊漂移判断，已知信号边界限制局部运动范围，共享容差/减速度/投影配置。检查范围见 [对接审查](../../DECISION_PLANNING_AUDIT.md)，信号字段见 [信号约束](../../perception/SIGNAL_CONSTRAINTS.md)，完整结果见 [规划链路修复](../../PLANNING_CHAIN_FIXES.md)。

2026-10-02 P0 集成：统一车头净距、跟车停走语义、精确停车意图和必需信号质量门控，完整合同见 [链路修复](../../P0_HANDOFF_FIXES.md)。

2026-10-03 PR #13 整合：保留已实测的弯道运动、路线归属与恢复修复，并消费 P0 业务净距；已知可信停止线的灯色失效允许有界接近停车，缺线仍保护。完整条件与最新验证见 [通用运动修正](../../perception/GENERAL_MOTION_FIXES.md)。

2026-10-01 从 V1 行为树改为**事实与来源 → 路线关系 → 约束集合 → 行为状态 → `DecisionTarget`**。仍使用 Python 3.6、SimOne 和固定 `decide(perception)` 入口，不新增公共字段或控制指令。V1 的交付历史见 [本地接入记录](INTEGRATION.md)；本文件描述当前算法。

## 输入事实与路线关系

- 自车必须有有效且未过期的 GPS 来源、可用车道与可投影的当前中心线；原始 `ego.gear` 只要求整数，反向运动由速度向路线切向投影判断。
- 场景是否必需目标流仍由 `core.scene_requirements.requires_targets()` 决定。只接受新鲜、可用的 Sensor API 目标；Ground Truth 仅作诊断，可选传感器缺失不阻塞车道场景。`sensor_read_ok=False`、过期、无效记录和未配置有不同的稳定原因前缀。
- `candidates.py` 将目标区分为当前车道、已核实的连续后继路线、已核实其他车道与未知关系。GPS 先锚定当前车道；连续路线须含不变的当前中心线前缀，并用公开的 `forward_lane_spans` 验证各段归属。目标投影拒绝端点外推和回环歧义，未验证的段不作为已知跟车路线。
- 有限目标朝向采用矩形沿路线的半长求后缘；缺失朝向保留外接圆。只有配置经核实的 `front_offset_m`（后轴中心到车头）才把路线弧长变成车头净距与已知障碍停车边界。缺车身/目标尺寸或可靠投影时不以零代替。朝向缺失时为 `None`，不能用默认零朝向缩小包络。

## 约束仲裁与输出

内部 `ConstraintSet` 保留每条适用约束的来源、标识、速度需求、停车边界、有效期和原因。所有目标及交通要求收集完成后按下列顺序仲裁：

1. 任一近距或有符号接近 TTC 紧急风险 → `EMERGENCY_BRAKE`。
2. 必需事实缺失、未知但可能进入路线的目标、横穿/迎面等当前无法确定停车点的冲突 → 在动保护制动，静止保持；不编造远端停车位置。
3. 多个已知停点取最近者；可信限速、所有移动前车速度需求取共同上界。已知停点尚远时输出 `KEEP_LANE` 或 `FOLLOW` 的**正速度需求和非负 `stop_distance`**；规划仍决定逐点加减速。
4. 到停车边界且真正静止时输出 `STOP/0/0`。HOLD 以独立的进入距离与释放余量防抖，连续两个新的清空观测后释放。正常状态为 `CRUISE`、`FOLLOW`、`APPROACH_STOP`、`HOLD` 或 `YIELD`；它们只映射到既有四个公共模式，不假装新增变道或让行协议。

静止目标提供停车边界；同向移动前车提供跟车速度。跟车任务及已跟随目标停稳后保留跟车净距，AEB 静态停车保持独立业务语义。默认静态余量 0.5 m、交通停止线余量 0.3 m；近风险保护可能要求更早停车。路线前车的公共 `target_lane_id` 始终是**当前自车车道**，不把后继车道误报为变道目标。交通停止线距离按同一后轴参考扣除车头偏移与停车余量。理由文本使用 `STOP_TARGET`、`STOP_TRAFFIC`、`FOLLOW_TARGET`、`TARGET_UNKNOWN`、`TARGET_SOURCE` 等稳定前缀，仍不是供下游解析的协议。

移动前车的试验速度需求为：

```text
g_des = min_gap + time_headway × v_ego
v_raw = max(0, v_lead) + gap_gain × (已知车头间隙 - g_des)
target_speed = clamp(v_raw, 0, 可信巡航/道路限速)
```

已知停车边界另施加试验性制动距离速度上限。普通接近使用 `follow_deceleration × approach_deceleration_ratio`，默认比例 0.5，即 1 m/s² 的速度包络；距离不足的硬判断仍使用完整 `follow_deceleration=2 m/s²` 与反应时间。这样在连续重规划的首点等于实测速度时，普通减速需求会先于“已经不能正常刹停”的保护条件出现；不放宽停车边界或急停门槛。`NEVC_DECISION_APPROACH_DECELERATION_RATIO` 可显式覆盖，取值须在 `(0, 1]`，1 对应旧包络。该比例同时适用于已知红灯停止线、静态目标与其他已知停车边界，不按场景编号切换。急停判定使用目标在路线局部切向上的**有符号**速度，自车静止时迎面目标仍可触发 TTC。公式及阈值尚未按 SimOne-Car 标定，不能作为实际刹停能力证明。

## 来源故障与恢复

必需目标源首帧失效即输出零速度保护，按**不同 GPS 帧号**累计故障；连续两帧失效锁存。锁存后即使目标恢复，也须车速降至恢复阈值以下，并得到连续三个有效新 GPS 帧，普通 FOLLOW 不得越过恢复门控。紧急目标风险不受恢复等待延迟。案例、任务、场景变化或帧号回退，以及 `reset_decision()`，均清空旧状态。非必需目标源缺失不触发该锁存。

## 参数与现场边界

默认巡航保持 **30 km/h（30/3.6 m/s）**。所有本模块调参仍用 `NEVC_DECISION_` 环境变量，显式代码参数优先；`NEVC_VEHICLE_FRONT_OFFSET_M` 是决策与规划共用的车头偏移输入。重要实验初值：`min_gap=10.5 m`、`time_headway=1.2 s`、`gap_gain=0.5 1/s`、`follow_deceleration=2 m/s²`、`reaction_time=0.3 s`、`standstill_speed=0.1 m/s`、`hold_distance=0.05 m`、`resume_margin=2 m`、`recovery_frames=3`。正值、有限值及整数帧数会校验。旧 `NEVC_DECISION_LAUNCH_TTC_CAP` 和 `NEVC_DECISION_STOP_MARGIN` 明确报废弃错误，不会静默假装生效。

2026-10-02 集成更新：用户提供的 SimOne-Car 主车预设为轴距 2.9187 m、前悬 1 m、车宽 1.8 m。平台启动脚本 `scripts/StartCaptain.bat` 默认向决策和规划共同传入后轴到车头距离 `NEVC_VEHICLE_FRONT_OFFSET_M=3.9187` 和车身半宽 `NEVC_VEHICLE_HALF_WIDTH_M=0.9`，保留调用环境已有覆盖；实际值记录在启动日志中。更换车型须重新核对。直接运行 `main.py` 不经过该脚本，须自行提供这两个环境变量；未配置时仍保留尺寸未知保护，见 [规划的尺寸门槛](../planning/OBSTACLE_GUARD.md)。静态目标判断、横穿冲突窗、制动减速度和控制响应仍需现场校核；尺寸已提供不表示制动能力已标定。项目仍使用 Python 3.6 和 SimOne。

## 验证范围与下游交接

`tests/test_decision_framework.py` 覆盖多目标顺序、红灯与前车合并、静态接近、后继路线、回头弯、横向目标、来源分类、故障恢复及 HOLD 释放。原决策/兼容/回放测试保留相应风险并更新旧语义；可使用：

```text
python -B -m unittest tests.test_decision tests.test_decision_compat tests.test_decision_framework tests.test_replay_decision -v
python -B -m unittest discover -s tests -v
```

`DecisionTarget` 只是行为要求。规划仍按其自身几何与目标包络检查停车可行性，控制发送仍受队长配置与监护限制。合并后的主分支已在 SimOne Python 3.6 下通过 325 项离线测试，包含真实 `decide → plan → compute_control` 入口的静态停车、跟车启停和红绿灯启停模型测试。平台已有车道闭环发送及车辆运动记录，正式 Sensor API 目标流和上述目标/信号行为仍需逐场景实跑验收；离线通过不代表赛题通过。`Trajectory` 已可表达倒车、精确停车、最短停留和灯光意图，见 [控制交接](../control/MANEUVERS.md)；生产决策与规划仍未生成换道、绕障、倒车和泊车行为路径，也未提供完整预测与规划执行反馈。
