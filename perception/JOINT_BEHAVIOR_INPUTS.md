# 联合决策/规划的正式观察输入

2026-10-12。使用本轮已授权的公共协议；运行版本 `joint-behavior-runtime-v1`。算法、实际入口、模型验证和未验收项见 [联合交付](../members/decision/JOINT_IMPLEMENTATION.md) 与 [联合规划](../members/planning/JOINT_IMPLEMENTATION.md)。

## 实际接线与当前数据状态

`SimOneAdapter.read_raw_snapshot()` 接受可选 `behavior_observation_provider` 回调，其 detached 字典进入 `raw.behavior_observations`。`perception/behavior_builder.py` 解析并做来源、帧、时钟、原始有效期和质量门控，发布 `Perception.behavior_environment` 的公共类型。固定成员入口只消费这些公共字段，不直接调用 SDK，也不从诊断 JSON 回读。

已有 HDMap 邻道采样可以提供真实车道样本；原有地图车位/停止线/横道观测、任务路径点和 Sensor API 对象流保留。它们各自的可读/有效状态不自动证明整段跨线权限、动态自由空间或完整覆盖。正式平台扩展的默认值仍未知，回调未安装就不会生成新动作所需的事实。模型测试有显式 synthetic 生产者并经过该解析器；该生产者只用于模型，不能投入生产。

现场仍需队长接入可核验的融合/任务生产者，发布下表的实际实例事实。本文记录可接线的字段及拒绝条件，不代表本机已经拿到这些真实数据。公用 SDK/运行/感知的最小改动已由用户明确授权。

## 外层头部和共同质量

| 字段 | 含义/要求 |
| --- | --- |
| `clock_id` | 必须为同一进程的 `process_monotonic`；SDK 的毫秒时间戳需由队长明确转换，不能直接冒用 |
| `frame_id/observed_at_s/valid_until_s` | 保留原始观察帧、观察时刻和截止；本帧不得早于来源，缓存不得延长有效期 |
| `source/source_kind` | 生产动态来源为实际 `sensor:<id>` 或明确 `verified_fusion`；融合还需 `input_sources` 关联当前正式目标流 |
| `dynamic_coverage` | `coverage_complete`、世界 XY 覆盖多边形 `regions`、`scope`（front/side/rear）、`proof_id/proof_kind` 及位置/速度不确定性和预测期限 |
| `execution_timing` | 可核验早到/晚到上界 `max_early_s/max_late_s`、`bounds_verified`、来源/种类；生产只接受 planning/verified_fusion 的实际证明 |
| `road_regions` | 已核实可行驶的简单轮廓，支持非凸；不能用凸包填平禁行凹陷或将不完整采样当作全覆盖 |
| `task` | 正式任务 `route_id/route_verified/source/source_kind`、返回/要求车道、终点和适用特殊停留规则 |

生产 `proof_kind` 必须是实际 FOV/遮挡等检查的 `sensor_fov_occlusion_validated`，不能由量程、传感器名称或空目标列表补齐。模型另用 `model_complete_world`，仅在 `observation_mode=model` 且 adapter 明确标为模型时可用。synthetic、诊断 Ground Truth、缺来源、过期或未知时间界都不能形成生产授权。

`Evidence` 另带可信 `TaskContext`（case/task/scene/session）。同一 session 标签变化/帧回退不等于新任务；明确真实任务变更才能生成新 session。新增事实序列化保留完整公共字段；JSON 只供诊断。

## 各行为的正式内容

### 车道机会

`lanes` 每项有 `lane_id/center_line/left_boundary/right_boundary`、来源/几何证明、相对 side、同向、标线和有界 `crossing_range_m`。范围采用该有限中心线累计站位；必须覆盖所请求的整段跨线，局部标线不能外推。任务 `return_lane_id/desired_lane_id` 给出规定路线要求；与实际车道关系同时核对。

### `parking_task` → `core.parking_protocol.ParkingTask`

- `id`：稳定停车任务；`bays` 含唯一稳定整数 `id`、车位世界边界、车身航向和 occupancy（empty/self/other/unknown）。
- `free_regions`：已经核实车身可以行驶/调整的自由轮廓，动态目标仍逐个检查。
- `exit_goal`：规定任务的驶出后轴目标（x/y/body_heading_rad），来源为任务路线；不传手写轨迹点。
- `minimum_dwell_s`：至少 10 s；`ego_target_id`：融合/任务核实的主车目标身份。自身占用还需实际车身与车位相交；排除同 ID 对象时核对实际车身中心位置。

决策/规划共同使用车头、车尾、半宽、轴距和最大转角；车位中心不是后轴目标。所选车位、任务义务、驶出目的改变需重新验证，不能将变动包装为同一个旧候选。

### `intersection_task` → `core.intersection_protocol.IntersectionTask`

- `id/route`：稳定路口和明确已选的有限世界 XY 路线。
- `entry_s_m/exit_s_m`：该路线上的入口/出口边界累计站位；与整车投影一致。
- `conflict_regions/drivable_region/exit_region`：真实冲突、可行驶、规定出口轮廓；支持简单非凸形状。
- `permission_windows_s`：同进程单调时钟的绝对 [start,end]；不是一次绿灯锁存值。
- `right_of_way`：EGO_PRIORITY/YIELD/UNKNOWN；`signal_permits`：当前适用信号许可；未知权利不准入。
- `speed_cap_mps/exit_gap_m`：任务通行上限/整车之外的业务间隙。

目标占用来自正式 Sensor API 全目标流及明确的覆盖/不确定性，不由本字段重新捏造“空路口”。规划反馈另含 `occupancy_window_s`，覆盖车头进入至车尾离开，加执行时间上下界；只有该区间整体落在当前可用窗口才准入。

### `emergency_task` → `core.emergency_protocol.EmergencyTask`

- `danger_id/dangerous`：融合层核实的危险身份/状态；显著占用变化修订身份，旧候选不得用于新危险。
- `allowed_regions/lateral_offsets_m`：规则允许且当前已核实的道路区域/有界左右横移幅度；列表含 0 才评估直线制动。
- 生产必须同时给 `allowed_regions_verified=true` 和 `rule_permissions_verified=true`，关联当前正式任务证据；动态危险证据必须为 verified_fusion。
- `return_lane_id`：危险解除后的原任务路线，不以临时邻道替换任务目的。

来源/权限/时间未知时继续已有急刹。当前生成器只给前向制动或转向加制动；反向危险使用原立即制动。风险/规则/净距代理和不可避免风险范围见共同交付页。`emergency_maneuver_authorized` 是生成器/监护核对的执行输出，不是外部绕过急停的开关。

### `task_progress` → `core.task_protocol.TaskRouteFacts`

- `route_id/segments`：规定的有序段序列；每段 `[stable_segment_id,lane_id,start_s,end_s]`，站位属于该已选任务路线，不能每帧换定义。
- `segment_id/progress_s_m/selected_branch_verified`：路线层的已选当前段及可达进度；不由决策另作全局最近点匹配。
- `reference_end_reason`：task_destination/window_end/map_end/branch_ambiguous/map_query_failed；后四类是覆盖/续接边界，不能完成任务。
- `destination_region/destination_goal`：规定终点的整车目标区/后轴姿态；只有最终规定段实际到达并停稳才完成。
- `required_obligations`：必须由实际反馈履行的稳定义务（例如 `signal:42`、`parking:<task_id>`），不能由规划或发送回执满足。

路线层目前已有有限后继/分叉选择和可达窗口；将其正式段序列、累计进度及真实终点语义发布到本字段仍需实例任务生产者。动态动作保留恢复锚点，帧回退保留义务；未完成/未知数据继续保护。

## 队长联调所需事实

| 依赖 | 当前结果 | 需要真实数据/核对 |
| --- | --- | --- |
| 前侧后动态覆盖 | 格式、解析、门控及模型已接通 | FOV、遮挡、作用区域、预测/意图不确定性；有效空帧不是覆盖证明 |
| 车道/跨线/路权 | 原地图数据与新协议兼容 | 整段跨线范围、同向、适用规则和信号未来许可窗口 |
| 停车/路口/应急任务 | 公共事实与实际算法已消费 | 本实例车位/自由区/入口/冲突/出口/允许区和任务目的 |
| 车型/执行时间 | 共同几何一致性和时间包络检查 | 已确认 Python 3.6、主车尺寸、真实转向/制动/挡位及早/晚界 |
| 任务进度/完成 | 实际终点/义务及恢复锚点接通 | 正式段、已选分支、可达进度、终点与局部参考区别 |
| 灯光 | 命令发送/持续时间证据 | 实际灯具生效及评分观察；当前未宣称灯具测量 |

联调仍按原全员确认与队长操作打开本机发送/监护开关；本轮共享默认值未改变。真实场景与官方评分结果须另存案例/实例、参数、版本和实际记录。
