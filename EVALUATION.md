# 本地评价记录接入

连接 SimOne 后，适配层通过已安装 SDK 的 `InitEvaluationServiceWithLocalData(vehicle_id, true)` 开启本地评价。SDK 提供 Judge 和主车 GPS 记录，项目不自行生成替代评分数据。默认开启，与车辆控制发送开关独立。

当前原生包装只支持已经核对的 SDK `3.0.0001` 两参数签名。它复用 `SimOneServiceAPI` 已加载的 DLL，不修改安装目录的 Python 文件。在线文档的新版接口多了一个参数，因此未知版本会在调用原生接口前报错。

运行中默认每秒调用 `SaveEvaluationRecord()`，在控制指令发送后执行；暂停时也可保存已经收到的数据。案例停止、Ctrl+C、结束脚本停止、运行异常都会进入清理路径，先保存再调用 `SoTerminateSimOneAPI()`。初始化失败会阻止进入控制循环；定期保存失败会记录错误并重试，最终保存失败使进程返回错误，同时仍清理 SDK、PID 文件和实例锁。

SDK 3.0.0001 的多次保存会追加多个 JSON 文档，并把主车记录单独写入 `<taskID>_mainvehicle.json`。安装平台的打包器只按单份 JSON 读取 `<taskID>.json`，原来可能因此报 `json read fail` 并生成空记录包。现由 `EvaluationRecords` 将两个 SDK 文件中的真实记录按原始时间戳排序、归并为一份任务 JSON，供原平台打包器读取。累计快照中的重复记录去重，同一快照内实际存在的相同事件保留原次数；`endreason` 采用 SDK 最近一次提供的值。程序不生成分数、pass 或替代 Judge 事件。

周期整理、备份和写盘在同一进程的后台线程完成，该线程不调用 SDK。原生保存与整理使用互斥门：后台忙时主循环跳过本次周期保存并稍后重试，不等待写盘；SDK 后续累计快照仍包含期间记录。退出前的最后保存同步确认最终记录已经交付，再终止 SDK。没有可用记录、格式错误或最终交付失败均明确报错，不把原生保存回执当作完整评分成功。

被替换的任务原始文档保存在本机忽略目录 `runtime_data/evaluation_raw/<taskID>/sdk-judge-*.jsonl`，SDK 的主车原始文件保持不变。每个任务分别维护状态，任务 ID 必须符合 UUID 格式；不从其他任务借记录。为限制异常输入，单次新增输入最多 256 MiB、任务归并记录最多 200,000 条，超过限制时交付失败并保留原数据。

`KillCaptain.bat` 现在调用 `stop_captain.ps1`，向本目录的 `runtime_data/captain.stop` 写入当前 PID。运行层只响应属于自身 PID 的请求，旧请求不会停止新进程。结束脚本等待最多 5 秒；SDK 不响应时才核对进程身份并强制结束，日志会提示最后一次保存可能不完整。Windows 强制结束进程不能保证执行 Python 的 finally，因此运行中也保存。

## 本机测试入口

当前 main 工作目录的启动脚本：

```text
F:\wangliancar\nevc_auto_main\scripts\StartCaptain.bat
```

结束脚本：

```text
F:\wangliancar\nevc_auto_main\scripts\KillCaptain.bat
```

必须启动新任务才能加载本次修改。其他工作目录的控制器和历史回放不会使用该目录的新代码。

配置默认值：

```ini
[app]
evaluation_enabled = true
evaluation_flush_interval_sec = 1.0
evaluation_record_dir =
```

可以在未提交的 `config/local.ini` 中覆盖。纯诊断时可显式关闭评价记录；控制仍由原有的 `send_control` 和 `safety_brake_enabled` 开关决定。

`evaluation_record_dir` 留空时按安装目录 `Common/Conf/simone.cfg` 的 `user_data` 和 TaskServer 的评价目录覆盖配置定位；没有用户目录配置时采用安装平台的 `~/.simone/evaluation` 默认值。特殊部署可显式指定实际评分目录，相对路径以项目根目录解析。不要把本机路径写入共享配置。启动过程中取得实际任务 ID 后才会整理该任务的文件。

只检查 SDK 加载与评价导出绑定，不连接案例、不记录或发送控制：

```powershell
E:\Sim-One\Tools\python36\python.exe F:\wangliancar\nevc_auto_main\main.py --self-test
```

## 平台验收

1. 本次 `captain.log` 应出现“评价记录已初始化”，随后出现“评价记录已保存”；正常退出前最后保存日志应为 `final=True`。
2. `latest_pipeline.json.evaluation` 应显示 `initialized=true`、`save_count>0`、`last_save_ok=true`。同时核对 `runtime.project_dir`、本次启动时间和快照修改时间。保存成功是 SDK 回执，仍需检查实际评价文件内容。
   新增 `evaluation.record_delivery` 给出实际任务、目录、原生保存/发布代数、记录类别数量、`packaging_ready` 和交付错误。周期快照可能早于最后一次保存；最终以“评分记录已交付”日志及实际包内容为准。
3. 案例运行期间检查平台用户目录 `evaluation/<taskID>.json` 中是否有真实 Judge / mainvehicle 记录。结束后平台会将其打包成 `<suiteID>.dat` 并删除单任务 JSON；不能仅以事后 JSON 不存在判断记录失败。
4. 以 NEVC 评分端认可的正式赛题验证上传响应和数字评分。本地场景 pass、SDK 保存成功、评分端接受赛题是分别需要验证的条件。

离线测试覆盖 C ABI 参数、保存重试、异常清理和真实结束批处理，以及累计文档、独立主车文件、事件次数、任务隔离、损坏源保护、后台阻塞隔离和终止 SDK 前的最终交付。另用真实历史 SDK 文件和安装平台的打包代码在隔离目录验证了加密包包含真实记录；未修改原平台记录或上传评分。新运行的实际时序、Judge 事件完整性及服务端数字评分仍须现场确认。

接口依据：安装 SDK 的 `include/SimOneEvaluationAPI.h`；[官方评价初始化文档](https://simone-docs.51sim.com/en/api/python/evaluation/init/index.html)及[官方保存文档](https://simone-docs.51sim.com/en/api/python/evaluation/record/index.html)。在线版本差异按上文处理。
