# 行为模型与任务的本机配置

标准入口 `main.py` / `CaptainRuntime` 读取 `behavior_model_file`，安装同一来源的决策、规划输入与独立安全检查。没有文件，或文件的 `verified=false`，不会启用本次新增模型或业务。实现、测试与未完成的现场验收见 [2026-10-12 实现报告](../members/planning/IMPLEMENTATION_20261012.md)。

## 配置方法

1. 将 [未确认模板](behavior_model.example.json) 复制到本机忽略目录，例如 `runtime_data/behavior_model.local.json`。模板的 `null` 表示未知；其中已有搜索参数只是数值尝试范围，不是车辆标定结果。
2. 队长与负责成员核对下表中的来源，补齐模型及对应任务，再明确设置确认字段。`verified=true` 是提供者的确认声明，程序不会因此产生实测证据。旧字段 `control_calibrated=true` 也只表示控制映射已经配置。
3. 在不提交的 `config/local.ini` 的 `[app]` 中添加：

   ```ini
   behavior_model_file = runtime_data/behavior_model.local.json
   send_control = false
   safety_brake_enabled = false
   ```

   文件路径相对仓库根目录解析。可以在本机覆盖文件使用绝对路径，但不把本机安装路径或数据路径写入共享配置。
4. 使用原有启动入口。格式错误、必要确认或数值缺失、控制与模型几何冲突会阻止模型加载，不会自动补参数。更改已加载的文件不会在运行中改变任务；重新加载由队长管理。

提供完整配置后，标准入口会安装对应输入供应器，成员入口仍是 `decide`、`plan`、`compute_control`。配置加载始终不修改两个发送开关；实际控制继续遵守团队联调与队长操作要求。

## 必需来源与负责成员

“误差界”表示允许的最大偏差，例如实际车身与参考轨迹相差多少米；它需要有适用车辆和运行范围的依据。“参考点”表示坐标对应车辆哪个位置，本实现统一使用后轴中心。

| 配置部分 | 需要提供的内容与来源 | 负责确认 |
| --- | --- | --- |
| `scope`、顶层确认记录 | 实际主车 ID、案例/任务 ID、SDK 所选 OpenDRIVE 的 MD5；确认记录路径或编号 | 队长 |
| `model.vehicle`、`limits` | 后轴前后车身距离、半宽、轴距；速度、加减速度、横向加速度、前轮转角/转角速率、数值检查容差；注明车辆资料或实测依据 | 控制成员提供，队长统一确认 |
| `model.gps_steering` | 官方 GPS 原始转向数值到前轮角弧度的比例、正负方向、已核实原始量程及依据；这与控制指令的归一化映射分别确认 | 队长/感知核对 SDK，控制成员核对车辆映射 |
| `execution_envelope`、`speed_error_mps` | 位置、朝向、提前/滞后时间、速度误差界；适用速度、曲率与控制器参数 | 控制成员提供跟踪测量，规划成员检查模型适用范围 |
| `prediction` | 正式 `sensor:<id>`、预测时域、目标位置/速度误差界及确认记录；当前工厂对同一正式 Sensor 的全部对象使用同一显式误差模型 | 队长/感知提供，规划成员消费 |
| Sensor 可见范围 | 另由 `sensor_visibility_file` 提供已核实的传感器覆盖与完整检测声明；模型文件不替代这份配置和当帧观测 | 队长/感知 |
| 搜索、预算与缓存 | 候选长度/形状/速度选项，动态速度搜索与停车格点搜索范围，最多检查次数、时间预算；搜索失败或预算耗尽不证明道路无解 | 规划成员，队长核对运行预算 |
| `overtake` | 明确任务 ID、原路线车道、速度上限、确认记录；动作 `OVERTAKE` / `AVOID` / `MERGE`、指定目标车道和是否返回；是否接受灯光发送回执作为任务允许的准备证据 | 决策成员与队长 |
| `parking` | 指定任务/车位/道路车道/入口边、车头朝向、接近/定位/退出后轴姿态；到点/朝向/停稳容差、误差模型与搜索配置；`task_verified` 及来源 | 决策成员提供任务，队长核对地图与授权，规划/控制核对容差 |
| `driving` | 指定通行区 ID、优先权、允许时间窗、可授权紧急侧向目标及路线；时间窗必须使用当前进程相同的单调时钟 | 决策成员提供业务授权，队长确认来源和时钟 |

`verification_reference` 应指向可核对的资料或测量记录，不能用合成测试的参数冒充现场标定。共享模板保留未知项，现场数值和原始记录留在本机忽略目录。

通行时间窗使用运行进程的单调时钟：这是连续递增的计时秒数，不是日期、SDK 毫秒或 Unix 时间戳。任务来源应在本次运行中提供同一时钟下的窗口；保存的旧窗口不能在下一次运行直接复用。

## 启用范围

- 只确认 `model` 时，安装共同运动模型，普通前进/跟车可以使用动态速度求解。此时不会从案例编号推断超车、停车、通行优先权或紧急侧向目标。
- `overtake.enabled=true` 安装车道转换任务；保留旧配置名以兼容已有文件。省略新增字段时仍为 `maneuver="OVERTAKE"`、`target_lane_id=null`、`return_required=true`，需当帧同车道慢车才能换出、超过并返回。
- `maneuver="AVOID"` 还需原决策的结构化停车/阻路/跟车约束与当帧正式 Sensor 静止对象关联；准备和路径请求阶段保留原停车约束。执行候选重新检查实际障碍物，不能因静止对象速度为零而把绕行授权速度也设为零。`AVOID` 与 `OVERTAKE` 都须在换出后实际超过原对象车身及所需余量，才算通过：`return_required=false` 此时完成；`true` 则继续检查返回机会，再实际返回完成。仅侧向进入邻道不表示已绕过障碍或超越慢车。
- `maneuver="MERGE"` 必须指定 `target_lane_id` 并设置 `return_required=false`；`return_lane_id` 此时仍指定出发路线车道。已确认任务不需要慢车触发，实际整车进入目标车道并稳定后才完成。任务不会自动推断车道末端或目标车道。
- 三种动作都需当前邻道合法穿越、足够的实测地图长度、完整后方/侧方/前方观测和原模型预算。未知后方、指定目标不可用或可观测距离不足均不派发；一次完成的无返回任务不会在同一任务下自动重复。
- `parking.enabled=true` 还需 `parking.verified=true` 和 `mission.task_verified=true`，并填写完整任务与搜索配置。任务指定车位不会使其自动存在、空闲或可进入；实际地图车位边界、完整 Sensor 覆盖和占用仍逐帧检查。倒车与前进段分别授权，实际停稳后才换段，原车位实际保持至少 10 秒后才驶出。
- `driving.enabled=false` 时，共同模型不附带通行或侧向业务授权。启用通行任务需要明确 `passage_id`、优先权与时间窗；启用紧急侧向候选需要明确目标与允许车道。空目标列表不表示获得通行权。
- `allow_command_lights=true` 只允许已授权任务使用连续成功发送灯光指令的时间；反馈仍明确 `lights_confirmed=false`，不宣称实际灯具已经亮起。需要实际灯具确认的现场任务应保持 `false`，并提供独立确认来源。

## 运行时保护与证据边界

输入供应器每帧绑定原始当前 GPS、正式 Sensor 对象、地图身份及原截止时间。原始 GPS 未包含转向字段时，兼容默认值 `0` 不被当作观测；不满足比例/量程也不生成初始曲率。配置的任务/地图/Sensor 范围改变、观测缺失或过期会拒绝输入并保护，不能静默把旧模型用于新任务。

任务配置是静态授权声明；当帧时间和来源绑定仅说明本次消费了这份声明，不会把它变成实时 Sensor 数据。允许时间窗、可见范围和目标误差不会因为刷新帧号而延长。所有候选与独立检查继续受当前来源剩余寿命及有限计算预算限制。

独立安全检查核对实际交付路径，并在每帧检查上次成功发送轨迹的实际位置、朝向与速度偏差；换道后的普通超越阶段也继续检查。只有 `attempted=true` 且 `ok=true` 的发送回执才登记已发送轨迹；安全覆盖发送不能登记原路径。发送成功不等于车辆实际执行，到位/停留/驶出使用实际反馈确认。

离线合成模型验证不等于真实 SimOne 行为验收，也不建立 20 Hz 实时预算或官方评分结论。实际来源、车辆响应、算时和评分仍由对应成员与队长分别验收。
# 决策增量：任务路线、停留规则和泊车选位

在原 `behavior-inputs-v1` 确认模型文件中可选添加 `task_route`。未提供或 `enabled=false` 时不安装任务进度提供者。启用需要 `verified=true`、`verification_reference`、`route_id`、`speed_cap_mps` 和明确的 `reference_end_reason`（`task_destination / window_end / map_end / branch_ambiguous / map_query_failed`）。

`segments` 是规定路段数组，每项包含 `id`、`lane_id`、`start_s_m`、`end_s_m`、`center_line`（世界 XY 米）。标准安装按当前实际 SDK 车道和有限线段投影得到进度；同一车道重复出现在多段时保持未知，需要队长通过 `task_inputs_provider` 提供已核实的当前任务段，不能按最近点猜测折返段。真实目的地另外提供 `destination_goal`（`x/y/body_heading_rad`）、完整 `destination_region` 和可选 `required_obligations`。局部参考末端不会完成任务。

可选 `stop_rule` 为 `id / minimum_duration_s / strict`：ID 必须匹配当前实际信号停线观测键（单灯 `signal:<id>`，分组 `signals:<排序编号>`）。规则明确要求停留时，即使初始就是绿灯也要完成。`strict=true` 请求增加 0.001 秒以表达严格大于；车辆必须真实到位、连续静止。未完成规则不可更换；完成后可以进入新的不同 ID 规则。同一 ID 不因重复帧或暂时失联重启。沿用原信号义务命名规则，例如观测键 `signal:42` 对应控制义务 `signal:signal:42`。

自动泊车可将 `parking.mission` 中的 `space_id/access_edge_index/body_heading_rad/approach_pose/position_pose` 替换为 `candidate_spaces` 数组。每项明确给出 `space_id/access_edge_index/body_heading_rad`；原 `task_verified/source/mission_id/road_lane_ids/exit_goal` 仍必需。选择器只选择当前正式地图/Sensor 证明空闲且容纳整车的候选，冻结后轴目标和实际道路中的接近点，并交原 P04 搜索。未知车位、入口、覆盖、地图或预算不足保持保护；候选配置不是实际空位证据。

完整行为和数据责任见 [决策集成说明](../members/decision/PR23_MAIN_INTEGRATION.md)。这些扩展不改变共享发送开关，也不把模板当作现场确认。
