# InsertAny3D-Local

这是 InsertAny3D 的本地控制面：批次命令、SQLite 调度、任务 worker、图片审核、APIYi/GPTEval 调用、
远端传输和跨端文件契约。Unity 包是独立的
[InsertAny3D-Unity](https://github.com/Junnan-bjtu/InsertAny3D-Unity)，服务器模型运行时代码位于
[InsertAny3D-Server](https://github.com/Junnan-bjtu/InsertAny3D-Server)。实验场景、Farm 工程、模型、
运行结果和密钥都不在本仓库。

## 快速开始

```bash
git clone https://github.com/Junnan-bjtu/InsertAny3D-Local.git
cd InsertAny3D-Local
python3 tools/insertany3d.py --help
python3 tools/insertany3d.py batch --help
python3 tools/insertany3d.py evaluate --help
```

安装控制面依赖时可以使用 `uv sync --locked`，也可以在已有 Python 3.10+ 环境中安装 `pyproject.toml`。
离线测试不访问模型、图片或评测服务：

```bash
python3 -m py_compile insertany3d/*.py insertany3d/contracts/*.py insertany3d/evaluation/*.py
python3 -m unittest discover -s tests -v
python3 tools/test_stage_adapter.py
python3 tools/check_public_boundary.py
```

## 运行方式

通常由 Farm wrapper 为每次运行创建独立的数据库和运行目录，再调用：

```bash
python3 tools/insertany3d.py --db /path/to/run.sqlite3 \
  batch run-all <batch-id> --manifest /path/to/batch.json \
  --root /path/to/run --real
```

真实模式需要本机 Unity、APIYi key、SSH 目标和 Server checkout 的私有配置；日常开发使用 `--fake` 或固定假
评测分数验证队列、审核、重试和断点续跑。默认图片审核为手动模式，每页显示 5 个任务，输入 `Y`、`R`、`N`。

Local 只生成远端请求并校验传输结果，真实 TRELLIS/GIM/SAGS 由 Server 仓库在服务器现有环境中执行。
`tools/` 中的阶段脚本是由 Server 发布树生成的、受 `remote_runtime.lock.json` 保护的发布快照；Server 才是
这些运行时代码的来源。Local 保留这份快照是为了让本地预检、离线契约回放和 SSH 传输保持可运行，更新时必须
使用 `tools/sync_remote_runtime.py` 从指定 Server checkout 逐文件同步，不能手工修改。`stage_adapter.py` 同时
支持离线契约校验；远端实际执行仍以 Server 仓库和服务器现有环境为准。

## 配置边界

将 `.env.example` 复制成私有 `.env`。APIYi key 推荐放在
`~/.config/insertany3d/apiyi_key`（权限 `0600`），不要写入环境模板。远端 GPU、模型缓存和凭据由 Server
checkout 的私有 `runtime.env` 管理，Local 不读取或覆盖它。

## 文档

- [命令行参考](docs/cli-reference.md)：批次、审核、恢复和评测命令。
- [工作流程](docs/workflow.md)：Unity、Local、Server 三端的步骤顺序。
- [流水线契约](docs/pipeline-contract.md)：输入、输出和失败重试规则。
- [评测](docs/evaluation.md)：GPTEval 两维默认评分和 XLSX 输出。
- [Unity 集成](docs/unity-integration.md)：UPM 与私有工程边界。
- [代码来源与同步](docs/code-authority-and-sync.md)：逐文件核对和发布顺序。
- [排错](docs/troubleshooting.md)：日志、预检和恢复处理。

## 数据安全

不提交 API key、远端令牌、绝对私有路径、Unity 场景、模型权重、虚拟环境、缓存、任务输入、运行目录或调试包。
运行产物应放在仓库之外，并由 run ID 隔离。
