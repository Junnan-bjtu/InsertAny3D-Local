# 命令行参考

## 入口状态

当前统一入口是 `tools/insertany3d.py`。它已实现持久批次状态、本机 Unity/APIYi worker、SSH 远端
worker、人工图片审核、六视角清单发现和 GPTEval 汇总。真实执行不会默认发生，必须显式选择 `--real`
并提供完整配置。

先查看当前版本的真实命令：

```bash
python3 tools/insertany3d.py --help
python3 tools/insertany3d.py batch --help
python3 tools/insertany3d.py evaluate --help
```

## 持久批次控制

`batch` 把批次和每个处理步骤的状态保存到 SQLite 数据库文件。默认数据库是
`.insertany3d/state.sqlite3`，也可在 `batch` 前用 `--db <path>` 指定。

| 子命令 | 当前用途 |
| --- | --- |
| `plan` | 校验显式批次清单并建立任务依赖；正式清单必须是 12 个工程、每个 5 个固定任务 |
| `start` | 把已规划批次推进到可调度状态；本命令本身不承诺真实 GPU 适配器已经部署 |
| `resume` | 检查过期占用记录并恢复可继续调度的状态 |
| `status` | 输出一次状态；`--watch` 持续显示固定表格，管道或 `--json` 使用 JSON Lines |
| `retry` | 按工程、任务或处理步骤选择失败项重试 |
| `cancel` | 取消整个批次或指定工程/任务的未完成步骤；存在活跃远端步骤时拒绝释放，须先走 `recover-remote` |
| `doctor` | 检查状态、占用记录和恢复风险 |
| `gc` | 默认只列出可清理暂存目录；只有显式 `--execute` 才删除 |
| `review list/decide` | 分页查看编辑图，并逐任务接受、拒绝或要求重新生成 |
| `fake-run` | 用本地假执行器推进任务依赖，不启动 TRELLIS、GIM、SAGS 或付费接口 |
| `worker` | 持续租用就绪步骤并写回结果；必须显式选择 `--fake` 或 `--real` |
| `run-all` | 规划/启动并续跑整个批次；显示任务进度，人工审核逐项确认，全部 eval6 后可接 GPTEval |
| `stage-command` | 为一个已就绪的 Unity 或远端步骤生成请求文件、结果位置和可执行命令；默认只生成不执行 |
| `evaluate` | 从数据库中成功提交的 `unity_eval6` 产物发现清单并复用 GPTEval；默认仍需 `--fake-score` 或显式允许付费调用 |

仓库附带的单任务示例只用于结构检查，必须加 `--draft`，不能把它当成正式 60 任务批次：

```bash
python3 tools/insertany3d.py --db /tmp/insertany3d-demo.sqlite3 batch plan \
  examples/batch-one-task.draft.json \
  --root /tmp/insertany3d-demo-runs \
  --draft
```

人工审核默认每页 5 项；每项决定独立保存。所有准确参数以对应的 `--help` 为准。

`batch worker --fake` 是队列执行循环的低成本模式，禁止用于正式实验数据库。省略执行模式会直接拒绝，
避免占位产物污染正式数据库。它会返回
`processedStages`、`succeededStages`、`failedStages` 和 `blockedReason`；手动审核模式正常停在
`waiting_manual_review`。审核一个任务后再次运行同一命令，该任务可以独立继续：

```bash
python3 tools/insertany3d.py --db <test-state.sqlite3> batch worker <batch-id> --fake
python3 tools/insertany3d.py --db <test-state.sqlite3> batch review list <batch-id>
python3 tools/insertany3d.py --db <test-state.sqlite3> batch review decide \
  <batch-id> <project-id> <task-id> <edit-attempt> accepted
python3 tools/insertany3d.py --db <test-state.sqlite3> batch worker <batch-id> --fake
```

 Farm_Test_001 的三个任务可以用一个可续跑入口串起来。wrapper 首次运行会自动生成新的 run_id；恢复旧运行应使用
`resume`，不会因为残留环境变量隐式复用旧目录。真实评价由 Farm wrapper 的 `run-all` 自动执行，底层 CLI
仍需显式 `--allow-paid-api`；离线联调可用 `--fake-score`；
`run-all` 默认就是一条命令到底：worker 执行期间持续显示状态监控，遇到人工审核会自动打开编辑图片，
等待 `Y/R/N` 决定后自动 resume，直到任务终态或整批评价完成。脚本/无终端环境可加
`--non-interactive` 安全返回待审核状态，之后重复同一命令即可继续：

```bash
python3 tools/insertany3d.py --db /var/tmp/insertany3d-runs/farm-test-001.sqlite3 batch run-all \
  farm_test_001_canary \
  --manifest ../MyProjects/Farm/InsertAny3D_Farm_Canary.batch.json \
  --root /var/tmp/insertany3d-runs/farm-test-001 \
  --real --allow-paid-api --expected-tasks 3 --expected-scenes 1 --tasks-per-scene 3
```

监控默认每 2 秒刷新一次；需要调整频率可加 `--monitor-interval 5`，不需要内置监控时才使用
`--no-monitor`。预览图片默认自动打开，使用 `--no-open-review-images` 可关闭。`--json` 只影响最终摘要
输出，不会关闭 stderr 上的状态监控；审核记录由流程统一标记为 `manual`。

真实模式覆盖以下步骤：

- Unity：`unity_anchor`、`unity_apply`、`unity_eval6`；
- 图片编辑：APIYi `generateContent`，按令牌指纹和模型共享并发闸门；
- 传输：输入逐文件哈希后原子上传，远端阶段结果逐阶段下载并复核，最后写下载回执；
- 远端：TRELLIS、对齐渲染、分割、GIM、联合位姿、SAGS ring6 和 debug bundle。

真实 worker 在领取第一项任务前检查所有 Unity Project、场景、公共包安装和 GUI 占用，并通过只读 SSH 按
`tools/remote_runtime.lock.json` 核对服务器脚本的路径、大小和 SHA-256。服务器缺文件或哈希漂移时会在
APIYi 产生费用前停止。API key 不提供命令行参数，推荐一次性保存在当前用户的私有配置文件中：

```bash
mkdir -p ~/.config/insertany3d
chmod 700 ~/.config/insertany3d
${EDITOR:-nano} ~/.config/insertany3d/apiyi_key
chmod 600 ~/.config/insertany3d/apiyi_key
```

文件中只写一行 APIYi key。之后 `batch worker --real` 和 GPTEval 会自动读取，不需要每次 `export`。
临时覆盖时可使用 `APIYI_API_KEY`；也可用 `APIYI_API_KEY_FILE` 指向另一份私有文件。旧的
`GEMINI_API_KEY_FILE`、`GEMINI_API_KEY` 和 `BEE_API_KEY` 仍兼容，但优先级低于默认配置文件。

其余真实 worker 配置仍通过环境变量提供：

```bash
export UNITY_EXECUTABLE='<Unity executable>'
export GEMINI_IMAGE_URL='https://api.apiyi.com/v1beta/models/gemini-3.1-flash-image-preview:generateContent'
export INSERTANY3D_REMOTE_TARGET='<user@host>'
export INSERTANY3D_REMOTE_PORT='<port>'
export INSERTANY3D_REMOTE_PROJECT_ROOT='<server InsertAny3D root>'
export INSERTANY3D_REMOTE_ARTIFACT_ROOT='<server batch artifact root>'

python3 tools/insertany3d.py --db <state.sqlite3> batch worker <batch-id> \
  --real --max-parallel 4
```

Linux/WSL 下 key 文件必须禁止其他用户和同组用户读取；权限不是 `600`（或更严格）时，程序会在领取任务和
调用付费接口前停止，并显示需要修复的文件路径。密钥内容不会进入任务请求、结果、日志或 SSH 远端环境。

默认手动审核时，本命令生成编辑图后正常返回 `waiting_manual_review`。逐任务确认后再次运行同一命令；它会
从已提交步骤继续，不会重做成功步骤。远端出现“请求可能已送达但本机无法确认”的情况时，状态进入
`recovering` 并继续占用对应远端资源，不能自动重跑。

真实 `--real` 当前固定要求 `editPolicy.mode=manual`。自动模式虽有队列状态和离线测试，但经用户决定暂缓的
图片解码、尺寸与内容完整性验收尚未实现，因此不能把“HTTP 成功”直接当作正式自动通过。

```bash
python3 tools/insertany3d.py --db <state.sqlite3> batch status <batch-id> --watch
python3 tools/insertany3d.py --db <state.sqlite3> batch status <batch-id> --watch --json --interval 5
```

断线恢复必须使用原 attempt 和 lease token。先只读探测；`RESULT` 只能走 `--recover-result`，`RUNNING` 不允许
直接释放，只有 `EXITED/MISSING` 才能显式选择 `--retry` 或 `--terminal`。如果确认是原 attempt 且需要主动停止，
可以使用 `--cancel-running retry` 或 `--cancel-running terminal`；只有整个远端进程组清理完成后才释放资源：

```bash
python3 tools/insertany3d.py --db <state.sqlite3> batch recover-remote \
  <batch-id> <project-id> <task-id> <stage> <attempt> \
  --lease-token <token> --probe
```

不要用通用 `batch retry` 绕过 `recovering`，也不要手工删除数据库中的 lease。活跃远端步骤同样不能用
通用 `batch cancel` 强制释放；必须先用 `recover-remote` 证明原 attempt 的状态。若无法证明整组已经清空，
命令会保持 `recovering` 和资源占用，等待后续 probe。

批次已经包含真实 `eval6` 清单后，可从批次 ID 运行不联网的评测闭环：

```bash
python3 tools/insertany3d.py --db <state.sqlite3> batch evaluate <batch-id> \
  --output <evaluation-output> --fake-score 8
```

未生成真实评测清单、数据库登记的文件哈希错误、角度混用或队列还没完成全部 `unity_eval6` 时，该命令会
在调用网络前失败。运行中的批次只读取成功 attempt 已提交到 artifact 表的清单，不扫描失败 attempt 的
`output.staging` 或旁路文件。汇总成功后，它为每个任务提交一份 GPTEval 结果并把批次推进到 `succeeded`；
重复运行只读取缓存并补交尚未完成的结果，不会用假执行器占位文件冒充评测图片。

GPTEval 的 `plan`、`run`、`status` 和 `summarize` 命令见[评测说明](evaluation.md)。其中 `plan`、
`status`、`summarize` 和带 `--fake-score` 的 `run` 不访问付费接口。
评价默认只计算 `visual_quality` 和 `geometric_accuracy`；只有显式传入
`--dimensions visual_quality insertion_rationality geometric_accuracy` 才会额外评价插入合理性。
`run` 和 `summarize` 除 JSON/JSONL/CSV 外还会写出 `gpteval_summary.xlsx`，其中包含场景汇总和任务明细。

## Server 仓库兼容入口

以下命令属于 Server 仓库，不是 Local 安装步骤：

| 命令 | 用途 | 状态 |
| --- | --- | --- |
| `Server/tools/run_insert_pipeline.py` | 顺序处理一个任务 | Server 兼容入口，保留 |
| `Server/tools/run_insert_batch.py` | 读取 jobs JSON 并处理多个任务 | Server 兼容入口，保留 |
| `Server/tools/stage_adapter.py` | 校验并执行一个远端阶段请求 | 由 Local `stage-command` 生成命令；不会自动登录 SSH |
| `Server/tools/bootstrap_third_party.sh` | 初始化固定子模块 | Server 安装入口 |
| `Server/tools/verify_environments.sh` | 检查 Python/CUDA 环境 | Server 诊断入口 |

## 批处理计划检查

`--dry-run` 会解析 jobs 文件、验证任务参数并写计划，但不启动模型：

```bash
python3 tools/run_insert_batch.py \
  --jobs /path/to/insert_jobs.json \
  --dry-run
```

常用参数：

- `--skip-ready`：跳过已有 `ready` manifest 的任务。
- `--fail-fast`：首个失败后不再启动后续任务。
- `--run-root`：覆盖 jobs JSON 中的输出根目录。
- `--cuda-device`：覆盖任务的 `CUDA_VISIBLE_DEVICES`。
- `--require-edit-manifest`：要求图片编辑响应、提示词和哈希契约完整。

旧 batch 是串行兼容入口，不代表 SQLite 队列的并发、恢复和取消能力。

`stage-command` 会从数据库里已完成的上游步骤收集文件和哈希，填写阶段选项，写出 `stage-request v1`，
并打印实际命令。Unity 命令会使用显式的 `-insertAny3DArtifactRoot`，让请求文件、输入文件和
`output.staging/stage_result.json` 采用同一个批次根目录。默认不会启动任何进程；确认项目路径和包安装后，
只有 Unity 阶段可以显式加 `--execute`，远端重模型阶段不允许此选项。使用 Windows Unity 时，
CLI 会把 WSL 下的项目路径转换成 Unity 能读取的 Windows 路径；执行前还会查询是否已有同一项目的 Unity 进程，发现占用就拒绝启动第二个实例。

例如为某个指定任务生成第一步 Unity 命令：

```bash
python3 tools/insertany3d.py --db /path/state.sqlite3 batch stage-command \
  <batch-id> <project-id> Task_001 unity_anchor \
  --unity-executable /path/to/Unity
```

Farm 的私有 canary 清单位于 `MyProjects/Farm/InsertAny3D_Farm_Canary.batch.json`，包含
`Farm_Test_001` 的 `Task_001`、`Task_002` 和 `Task_003`。推荐使用私有工作区的
`Run_InsertAny3D_Farm_Canary.sh run-all`，或直接调用上面的 `batch run-all`；它会续跑三个任务、在人工审核处逐项等待确认，
并在三个任务的 eval6 都完成后按参数自动接 GPTEval。Farm wrapper 不内置状态轮询；需要观察时在另一终端运行
`./Run_InsertAny3D_Farm_Canary.sh status --run-id <run_id>`。
人工审核是刻意的暂停点，因此无终端或 `--non-interactive` 模式会安全返回待审核状态：
单候选时交互提示使用 `Y`（接受）、`R`（重新生成）和 `N`（取消任务）；多候选时额外支持数字选择。
WSL 且系统提供
`explorer.exe`/`wslpath` 时，CLI 会自动打开编辑图预览；确认后只关闭本次启动的预览进程，
无法打开或关闭时会提示原因但不会阻塞队列。终端显示的 `image_edit` 尝试编号只保留前 10 位，
程序内部仍使用完整路径。

```bash
cd <private-farm-project>
./Run_InsertAny3D_Farm_Canary.sh run-all
```

wrapper 会自动生成唯一的路径级 `run_id`，并派生外部运行根目录下的数据库、任务目录和 `evaluation/` 子目录；清单内的
`batch_id` 仍固定为 `farm_test_001_canary`，二者不能混用。需要恢复既有运行时执行
`./Run_InsertAny3D_Farm_Canary.sh resume`，从只读候选列表选择 `run_id` 后进入同一完整流程。
数字选择、Enter 展开下一项、Esc/EOF/非 TTY 退出都不会创建新运行。直接调用通用 CLI 的
`batch resume <batch_id>` 仅恢复过期 lease/未完成提交并刷新状态，不会启动 worker；它不是 Farm
`resume` 的替代入口。

下面的显式数据库命令仅用于底层 CLI/测试，不是 Farm 推荐入口；其中 `farm-test-002` 只是示例 batch id：

```bash
python3 tools/insertany3d.py --db <run-root>/<run_id>.sqlite3 \
  batch run-all farm_test_001_canary \
  --manifest MyProjects/Farm/InsertAny3D_Farm_Canary.batch.json \
  --root <run-root>/<run_id> --real --expected-tasks 3 \
  --expected-scenes 1 --tasks-per-scene 3 --max-parallel 1 --monitor-interval 2
```

这条命令会自动规划（首次运行）、启动 worker、持续 monitor、打开编辑图片、等待 `Y/R/N` 审批，
并在审批后自动 resume。终端中断会关闭本地状态连接，但不会强制释放活动 lease：本地 Unity/API
进程可能继续运行到子进程自己的清理或 lease 到期；等待默认 60 秒后执行同一 wrapper 的
`resume --run-id <run_id>`（或明确带同一 `--run-id` 的 run-all）才会回收过期的本地 lease，不能直接
再次运行不带 `--run-id` 的 run-all，因为它会创建新运行。若中断的是远端步骤，恢复后
状态会是 `recovering/delivery_unknown`，必须先用原 attempt 和 lease token 执行 `recover-remote --probe`，
不能直接重跑或手工删除 lease。线程池会在 worker 退出路径等待正在进行的 API 请求，不会留下跨进程的持久线程。
需要无交互运行时才添加 `--non-interactive`，此时遇到审核会安全返回，而不是自动替用户批准。

运行 `run` 前必须关闭 Farm Unity GUI。真实 worker 会在领取步骤前检查占用，因此发现同一工程已经打开时
不会消耗 Unity attempt。当前服务器运行树尚未部署这份锁定 runtime；在保存服务器脏改动并核对哈希之前，
服务器运行时代码由 Server 仓库维护；Local 的 `tools/` 只是受锁文件保护的发布快照。Farm 脚本仍需通过
单任务真实 canary 才能证明模型链路已跑通，低成本测试本身不等同于模型实跑。

## 单任务入口

参数较多，先用以下命令读取当前版本的真实参数：

```bash
cd /path/to/InsertAny3D-Server
"$INSERTANY3D_PYTHON" "$INSERTANY3D_SERVER_ROOT/tools/run_insert_pipeline.py" --help
```

单任务至少需要编辑图，以及 `--output-dir` 或 `--run-root` + `--task-id`。正式任务还应提供 Unity 三视图、
深度、相机和 manifest。不要从 README 复制一条长命令后跳过输入检查。
