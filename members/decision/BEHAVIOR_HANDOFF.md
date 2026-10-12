# 行为与反馈交接：M0 可执行草案

更新：基于 main `aa65aeb` 的感知/集成第一阶段已将协议类型提升到公共层，并接入
D01 信号停点与实际停留反馈。当前接口、能力边界与未完成项见
[行为感知与正式交接](../../perception/MANEUVER_ENVIRONMENT.md)。下文保留原 M0 基线记录，
其中“公共接口尚未接入”仅描述原始交付，不代表本分支现状。

当前可选扩展：公共 `BehaviorGoal.source_lane_id` 表示换道原源道，非空时须
有不同的 `target_lane_id`，参与语义修订、阻塞及恢复保持。仅非空写入 v1
平面请求，原前进/信号字典保持原形；旧前进/信号规划消费者明确拒绝非空
源道，不会静默丢弃。固定终点复用既有 `goal_pose`。当前 D03 阶段消费和
剩余生产派发见[换道执行交接](LANE_CHANGE_EXECUTION.md)。

基线 `main 568e9c9`，2026-10-08。`behaviors/contract.py` 与 `behaviors/session.py` 是可单独调用和验证的决策代码，承接规划 R05/R07 草案。**公共接口尚未接入此合同**：`Perception` 没有反馈/能力通道，`DecisionTarget` 没有行为阶段、灯光和停留目标。本模块不加动态属性，不解析 `reason`，也不读取诊断文件构成反馈环。

## 现有字段和拟议字段

现有 `DecisionTarget.mode/target_speed/target_lane_id/stop_distance/obstacle_clearances_m/precision_stop` 继续由真实 `decide()` 生产。现有 `Trajectory.motion_direction/hold_duration_s/parking_brake_at_stop/left_signal/right_signal/hazard_signal` 可以被控制消费，但决策尚无获批准的透传路径。

| 私有草案字段 | 默认/无效语义 | 提供方 → 消费方 | 需要队长落实的通道 |
| --- | --- | --- | --- |
| `contract_version` | 不匹配则不可派发 | 集成 → 全链 | 版本协商与能力事实 |
| `task_context` | case/task/scene/session 任一不符即拒绝 | 运行 → 决策/反馈提供者 | session 明确重置与任务身份 |
| `intent_id/stage/revision` | ID 稳定，阶段或目标语义变化才递增修订；旧修订拒绝 | 决策 → 规划/控制/运行 | R07 行为目标通道 |
| `source_frame_id/issued_at_s` | 创建当前修订的感知帧与本机秒数 | 决策 → 反馈校验 | 与实际输入关联 |
| `produced_frame_id/produced_at_s/valid_until_s/clock_id` | 当前请求继承本帧截止时间；反馈拥有独立可核对年龄，不延长旧输出 | 各生产者 → 消费者 | 固定单进程本机单调时钟；其他时钟须集成转换 |
| `goal_pose/motion_direction/speed_cap_mps/stop_distance_m` | 位姿是后轴位置和车体朝向；速度非负，方向单独 ±1；未定路径停距 `-1` | 决策 → 规划 | 复用 R07 草案及已存在轨迹方向消费者 |
| `minimum_standstill_duration_s/light_intent/stop_obligation_id` | 无义务为 0/全 False；不计控制倒计时 | 决策 → 规划 → 控制 | 复用轨迹的停留和灯光字段，并核对安全覆盖 |
| `status/reason_code/retryable/progress` | 未收到可信反馈不完成阶段；有限超时/重试 | 规划/控制 → 运行 → 决策 | R05 反馈通道 |
| `actual_standstill_confirmed/goal_pose_arrived/standstill_duration_s/hold_completed` | 默认 False/0；发送或规划接受不能代替实际执行 | 控制/运行 → 决策 | 到点、停稳、连续停留的独立证据 |
| `candidate_id/actual_lane_id/lights_duration_s` | 候选标识须对应规划确认；到车道/灯光持续时间由实际反馈提供 | 规划/控制/运行 → 决策 | 候选和执行证据，不以方向标志代替安全 |
| `Capabilities.actions/usable/valid_until_s` | 默认没有下游能力，`dispatch_allowed=False` | 集成/规划 → 决策 | 能力/观察质量门槛；草案不可默认开启 |

决策侧类型明确定义字段、合法值与校验，不意味着公共字段已批准。队长应在核心协议添加相应类型后统一感知、运行、规划、控制、序列化与验证；本模块类型可作为参考，不是另一条隐式生产链。

R06/R08/R09 输入也有独立 `Evidence`：默认 source_kind=unknown、不可用、覆盖未核实；动态占用只接受 Sensor/经核实融合或明确标记的独立合成测试证据。静态地图或 diagnostic_ground_truth 不能代替动态覆盖。后续策略所需车道边界、车位占用/自由区、路口冲突区/出口和段进度见 [分阶段交付](BEHAVIOR_IMPLEMENTATION.md)，对应原草案，未写进公共 `Perception`。

## 生命周期和时间

`BehaviorSession` 维护稳定 ID、阶段修订、接受/执行状态和有界事件记录。反馈必须同任务、同意图/阶段/修订，来源不早于该修订创建帧，生产帧不在未来，并在同一已核实单调时钟的有效年龄内。上一帧反馈不要求等于当前帧；重复与乱序反馈不重复推进。反馈状态 PLANNED 只能表示规划接受，不能完成动作。

缺少能力时只产生不可派发的请求。`dispatch_allowed` 只允许向具有相应能力的规划/运行消费方提交目标，不授权直接驱动车辆；规划可行性、控制联锁与运行监护仍需逐级核对。暂停、失效、长观测间隙及急停暂停当前阶段；恢复重验时递增修订，让旧执行结果失效。路径拒绝只有明确可重试且未耗尽次数才重发；接受超时和无实际进展超时有终止出口。默认实验值为反馈最大年龄 0.5 s、接受 2 s、无进展 5 s、重试间隔 0.5 s、同一意图累计最多 2 次（跨阶段不归零）；尚未以平台响应标定。

样例 [behavior_contract.synthetic.json](examples/behavior_contract.synthetic.json) 明确标记手写/合成，不是正式观测或已批准接口。M0 测试在 `tests/test_decision_behavior_contract.py`，没有模拟新字段已经出现在生产 `Perception` 中。
