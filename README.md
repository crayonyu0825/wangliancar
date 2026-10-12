# NEVC 队长部分（API、感知与集成）

这是四人方案中的“队长项目”。它不是整车算法，而是全队共同运行的底座：只在这里连接官方 SimOne API，把原始数据整理成统一接口，然后依次调用决策、规划和控制成员的模块。

组员开始开发前，请先阅读 [四人协作开发说明](TEAM_COLLABORATION.md)。

2026-10-04 审查修复：统一路线投影、目标包络、独立监护和危险解除；补充有车身/转向/走廊依据的回正与稀疏弯道处理，并修复提前减速到控制执行的衔接。当前版本与验证范围见 [规划链路修复](PLANNING_CHAIN_FIXES.md)。

2026-10-03 修复分支补充弯道目标运动校验、接头归属恢复、任务点分叉选择及信号质量/停车约束，路线版本为 `successor-continuation-v3`。真实刹车帧的证据、公共字段及平台待验证范围见 [通用运动修正](perception/GENERAL_MOTION_FIXES.md)。

2026-10-12 联合行为接入：固定入口现已消费慢车超越/返回、绕障/汇入、生产泊车搜索/驻留/驶出、路口窗口/出口准入、应急候选和正式任务终点。软件、模型与平台状态见 [M1～M5 联合交付](members/decision/JOINT_IMPLEMENTATION.md)，真实数据生产和缺值保护见 [联合观察接口](perception/JOINT_BEHAVIOR_INPUTS.md)。本轮共享发送开关保持原值，SimOne/评分仍待实例验收。

## 当前已经做到

- 使用官方 `SoInitSimOneAPI` 连接 SimOne，读取 GPS，并按场景传感器配置读取目标；目标传感器不可用时 Ground Truth 仅作诊断回退。
- 目标接入兼容数字传感器类型，支持官方回调发现实际 ID、已知源间读取切换及失败重试；日志区分缺少配置、读取失败和有效空目标帧，见 [目标接入说明](perception/TARGET_DATA_INGESTION.md)。
- 加载 HD Map，输出当前车道号、中心线、左右邻车道、车道宽度和车辆相对车道误差；另输出连接位置和方向已核对的唯一后继车道参考线，供规划跨车道段连续前进。分叉路线仍需上游明确选择，详见 [路线接续说明](perception/ROUTE_CONTINUATION.md)。
- 自动从案例名称识别场景编号，例如 `06.车道居中控制-测试` 识别为场景 6。
- 每一帧生成统一的 `Perception`，写入 `runtime_data/latest_perception.json`，方便全队直接查看。
- 同步写入 `runtime_data/latest_pipeline.json`，记录决策、轨迹、控制和平台发送回执；重复 GPS 帧仍更新快照与各传感器状态，但不会重复发送控制。
- 已接入 SDK 3.0.0001 的本地评价初始化与保存，默认记录官方 Judge/GPS 数据，定期保存并在 SDK 终止前保存；结束脚本先请求正常退出。配置、日志核对和真实评分验收见 [评价记录说明](EVALUATION.md)。
- 目标源读取从实际配置或官方回调发现 ID，处理失败切换、重试和过期；`target_input` 分别显示有效有目标、有效空帧和正式源不可用。只读任务配置核对工具另提供主车/脚本/传感器清单，详见 [目标接入说明](perception/TARGET_DATA_INGESTION.md)。
- 额外读取任务路径点、传感器配置、环境，以及 SDK 可用时的 IMU、毫米波雷达、超声波与摄像头车道观测；这些数据在 `Perception` 中各自标明来源/有效性，不与目标真值混用。
- 适配层对已核对的 SDK `3.0.0001` 旧 Python 绑定补齐路径点、目标检测与 Ground Truth 的布局，并修正交通灯字段偏移；兼容类型仅用于轮询，不修改 SDK 安装文件。未知版本的旧布局会在调用原生读接口之前报错。
- 地图车道输出左右边界和标线；交通灯查询当前车道及已验证的所选后继，保留完整停止横线并求真实路线交点。关联线未定位保持未知保护；多灯同色且本帧均可读时才发布共同灯色，冲突或读取失败继续停车。
- 通用 HDMap 接入另提供交通标志朝向向量、车道作用范围、停车位边界/入口/标线，以及信号或标志关联的停止线和斑马线。来源状态明确静态地图、API 是否可读及查询覆盖；不依赖单个题号。限速按已核对的 SDK 类型、明确单位、适用车道和朝向解析，前方标志距离另保留给规划。详见 [地图语义交接](perception/MAP_OBSERVATIONS.md)。
- 决策、规划、控制保留固定入口；决策已接入按来源与路线整理事实、合并多目标/交通约束并用状态机输出巡航、跟车、接近停车、保持与紧急请求的 `DecisionEngine`；固定公共模式与数据链兼容保持。算法与 `StartCaptain.bat` 的默认巡航统一为 30 km/h（约 8.333 m/s），控制跟踪速度上限同步为 30 km/h；可信限速、弯道和停车约束可进一步降低速度。规划使用原始地图几何估算弯道限速，避免每帧重采样点距变化造成伪限速突降。控制接入前进挡 Pure Pursuit、PI 速度控制与停车保持；带目标的保守停车轨迹已加入，但必须提供经核实的车身前端距离和半宽；未知停车点可请求保持或制动。动态切入预测、分叉路线选择及真实闭环跟车仍待验收，不能将决策模式已实现视为整车能力已验收。启动方法见 [平台运行说明](scripts/PLATFORM_RUN.md)。
- 默认 `send_control=false`，所以目前只观察、不抢车辆控制权。
- GPS 的 `ego.gear` 是原始变速箱位置，与 `ControlOut.gear` 的 N/D/R/P 控制枚举不同。前进控制明确输出 Drive=1，保护制动使用 Neutral=0；不把反馈值 2 直接解释或发送成倒挡。感知保留原始值，反向运动由速度在车头方向的投影判断。
- 控制现支持显式双向轨迹、停稳换挡、精确停车、最短停留和静止手刹，并透传上游灯光意图；倒车速度上限暂为 1.5 m/s。真实决策/规划/控制入口的前进静态停车、直线/弯道跟车停走和红绿灯启停有闭环模型测试；另覆盖坡度负载、垂直/平行倒入与至少 10 秒保持后驶出。启动脚本按用户提供的 SimOne-Car 尺寸传入前端距离 3.9187 m 和半宽 0.9 m，可被已有环境变量覆盖。新生产生成器现能消费正式任务/环境生成倒车、泊车和换道路径；真实观察/权限和平台验收仍待接入，字段与交接见 [控制动作说明](members/control/MANEUVERS.md)。
- 运行层另有刹车安全监护：当前场景必需的目标流或车道失效、紧急停车、轨迹/GPS 失效或 GPS 帧停滞时生成零油门刹车候选，并在 `latest_pipeline.json` 记录原因。06 等车道场景没有目标传感器时不因此停车；01 等目标场景仍要求有效的 Sensor API 目标源。只有 `send_control` 和 `safety_brake_enabled` 同时显式开启才会尝试发送；两项默认均为 `false`，候选刹车比例尚待场景标定。
- 来源帧采用本机单调时钟判断停滞、回退和跨传感器帧差；超时及模块输出的来源帧、有效期在传输节点验证。传感器没有帧号时标记 `timing_unknown`，不伪称同步。
- 停车/跟车净距由决策统一下发，精确停车意图传至控制；信号区分确认无灯与读取失败，必需信号失效由决策和运行监护共同保护。规划先核对路径冲突，邻道正常对向车不再触发整条轨迹失效。当前字段、参数与验证边界见 [P0 链路修复](P0_HANDOFF_FIXES.md)。

## 四个人怎样接入

数据流只有一条：

```text
SimOne API -> 队长 Perception -> 决策 DecisionTarget
           -> 规划 Trajectory -> 控制 ControlOut -> 队长调用 SoSetDrive
```

三个成员只需要改各自文件中的一个函数：

| 成员 | 文件 | 函数 | 输入 | 输出 |
|---|---|---|---|---|
| 决策 | `members/decision_stub.py` | `decide(perception)` | `Perception` | `DecisionTarget` |
| 规划 | `members/planning_stub.py` | `plan(perception, decision)` | 前两级结果 | `Trajectory` |
| 控制 | `members/control_stub.py` | `compute_control(perception, trajectory)` | 感知和轨迹 | `ControlOut` |

接口字段全部在 `core/interfaces.py`。三个成员不要直接 import SimOne SDK；SDK 字段变化只由队长在 `simone_platform/simone_adapter.py` 中处理。

## 运行方法

按联调指导文件在平台“资源库 → 控制器”的 **启动脚本** 填写本机项目目录下 `scripts\StartCaptain.bat` 的绝对路径，在 **结束脚本** 填写 `scripts\KillCaptain.bat` 的绝对路径。这两个批处理脚本会启动/结束整个 `nevc_auto` 运行链，不需要把 `members/control/controller.py` 填进平台。平台更新控制器后按指导刷新本地端。

启动脚本会在存在 `config/local.ini` 时加载这份本机覆盖配置，否则使用共享默认配置；该文件不提交。观察/低速验收切换只修改本机覆盖，不需要重新填写平台脚本路径。每次启动的 Python 与原生 SDK 输出保存到单独的 `runtime_data/captain-console-*.log`，具体文件名记录在 `launcher.log`；异常退出后可据此检查连接失败和原生错误。JSON 快照仅用于诊断，文件占用或其他写入错误会记录警告、跳过本帧快照并在后续帧恢复，控制和监护继续执行；读取快照时需检查修改时间和来源帧，不能把旧文件当成当前状态。

1. 在 SimOne 中选择并启动案例。
2. 双击 `scripts/StartCaptain.bat`。
3. 日志在 `runtime_data/captain.log`；感知和整条链的最新快照分别在 `runtime_data/latest_perception.json`、`runtime_data/latest_pipeline.json`。
4. 停止时按 `Ctrl+C`，或运行 `scripts/KillCaptain.bat`。

平台调用记录还会写入 `runtime_data/launcher.log`。如果案例结束后没有快照，先看这个文件确认启动脚本是否执行、Python 是否异常退出，再看 `captain.log` 的“等待案例运行”和“首帧检查”日志定位卡在案例状态还是感知读取。同一案例重复调用启动脚本时，第二个队长进程会退出并记下“忽略重复启动”，避免两个进程同时连接同一辆车。Windows PowerShell 可用 `Get-Content .\runtime_data\captain.log -Tail 50 -Wait` 实时看日志。默认配置仍是 `send_control=false`、`safety_brake_enabled=false`；因此即使出现有效控制候选，也不会驱动车辆。

普通链路日志每 100 次循环输出一次，可能漏掉短暂刹车。运行层另即时记录 `braking_event=brake_start/brake_change/brake_release`，稳定制动期间每两秒采样，区分规划失效、安全监护和控制器正常减速。开启 JSON 诊断时，在 `runtime_data/braking_events/` 保存每进程最多 80 个完整事件快照，文件名含 PID、序号和感知帧。快照包含同帧感知、路线、目标原始运动和宽度、决策、轨迹、控制、监护与发送回执，并保留前八帧摘要；因此后来的巡航快照不会覆盖早先的刹车帧。候选或发送成功回执不等于平台已经执行，需与自车速度和 GPS 刹车反馈一起核对。事件记录不改变规划或控制指令，写入失败不终止控制循环。

只验证 SDK 能否加载，不连接案例：

```bat
E:\Sim-One\Tools\python36\python.exe main.py --self-test
```

只读取一帧后退出：

```bat
E:\Sim-One\Tools\python36\python.exe main.py --once
```

## 开启车辆控制前必须完成

控制算法已实现离线首版。默认 `control_calibrated=true` 表示已配置一套 **SimOne-Car 试运行参数**：轴距来自平台预设，其余转向和踏板比例是待验证的工程初值。有效感知与轨迹可产生 `ControlOut.valid=True` 的候选指令；这不等于整车响应已验收。默认 `send_control=false`、`safety_brake_enabled=false`，不会发送。参数含义和低速现场验收步骤见 [控制实现说明](members/control/IMPLEMENTATION.md)。全队确认后才由队长开启控制发送；正常发送也要求安全制动出口以 `safety_brake_enabled=true` 布防。

## 感知数据的含义

- `Perception.valid` 表示 GPS 自车基础状态可用；目标、车道、交通、路线及辅助传感器要分别查看各自的 `valid/status`。成功读取但没有目标与接口失败不同。
- `source_status` 记录每个来源的 `read_ok`、帧号、本机重复帧持续时间、质量与 `usable`。`targets_valid=False` 表示目标读取失败、损坏、过期或不同步；空 `targets` 不能单独解释为“道路无车”。
- `source_status.targets.sensor_presence` 区分 `configured`、`not_configured`、`unknown`；`sensor_read_ok` 记录目标 Sensor API 是否读取成功。`not_configured` 在不要求目标流的场景仅作状态记录，未知场景保守处理。
- 本机 IMU 独立结构没有帧号；读数正常时 `imu_valid=True`，同时 `source_status["imu"].quality="timing_unknown"`、`usable=False`，需要同步状态的算法不能据此假定它与 GPS 同帧。
- `Target.same_lane` 只在地图车道归属已核实时有效；`same_lane_valid=False` 时可参考 `lateral_band_match`，但它只是横向距离筛选。车道限速缺少可信来源时保持 `-1`，交通灯方向不明确时保留候选列表并标记 `ambiguous`。
- `ego.acceleration` 是沿当前车头方向的有符号加速度（m/s²）；`ego.speed` 是水平速度大小（m/s），不表示倒车方向。`ego.age_ms` 和 `targets_age_ms` 是本进程从首次看到该帧起计算的时间，`-1` 表示未知。
- `route_points` 是二维案例任务路径提示，`route_waypoints` 另保留原始点序号和朝向四元数；两者都不是规划成员输出的 `Trajectory`。`traffic.stop_line_distance` 是沿当前车道中心线到真实停止线的距离，参考点为 GPS 位置；没有可靠停止线时交通字段保持无效。
- 本机 Python 地图模块没有直接车道限速接口；现已解析 SDK 明确的限速标志，在已核对且已通过的作用范围内更新 `lane/traffic.speed_limit`，未知情况仍为 `-1`。前方限速观测只提供事实，提前减速需规划消费其距离；泊车位也不表示已生成泊车轨迹或验证占用。图像、点云、V2X 原始流仍需独立处理链。
- 41 场景的数据需求、必需 API 与现场验收步骤见 [场景数据验收清单](perception/SCENE_DATA_ACCEPTANCE.md)；只读采样命令为 `python scripts/accept_scene.py --scene 6 --frames 20 --timeout-sec 30`。报告的 `STRUCTURAL_PASS` 仅表示帧结构通过，事件仍需现场核对。
- `DecisionTarget`、`Trajectory`、`ControlOut` 通过 `frame_id`、`timestamp`、`valid_until` 关联同一帧；`valid_until` 是本机单调时钟截止时间，不能跨进程持久化复用。当前规划输出前向参考轨迹及受加减速、弯道、地图末端和目标包络约束的点速度与时间，见 [规划基线说明](members/planning/BASELINE.md) 与 [带目标轨迹约束](members/planning/OBSTACLE_GUARD.md)。已有真实前进车道闭环及离线模型验证，制动响应、正式目标场景、倒车与泊车仍待现场验收。
