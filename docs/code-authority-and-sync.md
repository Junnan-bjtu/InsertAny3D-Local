# 三端代码来源与同步规范

## 目的

InsertAny3D 有本地编排区、服务器运行区和 Unity 区。任何一端都可能有未提交更新，因此同步前必须审计，不能按目录整体覆盖。

## 三个代码区

| 代码区 | 位置 | Git 状态 |
| --- | --- | --- |
| 本地编排区 | 当前 Local checkout | CLI、调度器、评测、契约和 Local tools |
| 服务器运行区 | 独立 Server checkout | 模型脚本、GPU 环境、缓存和运行数据 |
| 顶层集成区 | `Junnan-bjtu/InsertAny3D` | 只管理 submodule 指针、共享契约和集成检查 |
| Unity 区 | 私有 Farm 工程 | Unity C#、场景、wrapper 和清单 |

`codex_remote_tools/`、`InsertAny3D/tools/` 和服务器 `tools/` 只有在逐文件 hash 一致时才视为同一版本。

## 当前审计结论（2026-09-02）

- 发布时记录 Local 和 Server 的 Git HEAD/status，运行目录不作为代码来源。
- 本地运行时副本与服务器当前 `tools/run_insert_pipeline.py`、`workspace.py`、`stage_adapter.py`、`remote_runtime.lock.json` hash 一致，说明这些文件曾按锁文件部署。
- 服务器 `.insertany3d-backups/20260902T003022/` 保留了上传前版本，其中旧 `run_insert_pipeline.py` 与当前版本不同，说明服务器曾有独立修改。
- 服务器比本地运行时副本多出 `select_sags_views.py`、`download_models.sh`、`SAGS_ALGORITHM.md`、`verify_environments.sh`、`bootstrap_third_party.sh`、`ipc_server.py`、`trellis.py` 等文件；本地也有服务器没有的测试/辅助文件。
- Unity 没有 Git 历史；`InsertWorkflowService.cs` 在 2026-09-01 修改过，`LegacyInsertAny3DSourceBackup_20260829` 只能作为历史参考。

“最新”按文件判断：已提交版本看 Git commit，未提交版本看 mtime 与 hash，备份只作证据，不自动作为来源。

## 权威选择规则

每次同步记录相对路径、两端 hash、mtime、Git HEAD、是否已提交、选择来源和理由。

| 比较结果 | 动作 |
| --- | --- |
| hash 相同 | 记录已同步，不复制 |
| 只有一端改变 | dry-run 后分阶段发布 |
| 两端都改变且 hash 不同 | 阻止同步，人工合并或选择版本 |
| 备份、staging、缓存、运行产物 | 不进入代码同步 |
| submodule 指针不同 | 单独核对 commit 后再提交 |

禁止使用“本地永远权威”或“服务器能运行所以权威”的假设，也禁止无校验的 `cp -r`、`rsync` 或整体 checkout 覆盖。

## 推荐 Git 布局

1. `InsertAny3D-Local` 使用正式远程 Git 仓库，管理编排器、契约、评测、Local wrapper 和同步脚本。
2. `InsertAny3D-Server` 独立跟踪服务器运行时代码；服务器只保留 `.insertany3d/runtime.env`、缓存和运行数据，不作为长期开发分支。
3. `Junnan-bjtu/InsertAny3D` 顶层集成仓库只固定两个 submodule、共享契约和集成检查；验证前不迁移到 FlagOpen。
4. `MyProjects/Farm` 建立独立 `InsertAny3D-Unity` 仓库，管理 C#、场景、package、wrapper 和清单；`Library/`、日志、运行结果和旧备份不提交。
5. `codex_remote_tools` 视为运行时发布快照，不作为第三个独立开发源；最终应从 Server Git commit 生成 lock 和部署包。
6. 第三方模型使用固定 submodule commit，服务器 detached HEAD 不直接视为主仓库更新。

## 日常同步

本地修改先测试、生成 runtime lock，再与服务器逐文件比较；有服务器独有改动时先保存 patch，不覆盖。服务器紧急修改必须先保存 `git diff`，带回本地分支合并、测试后再提交。Unity 只消费已确认版本的 manifest、workspace 和 artifact 契约，不直接修改服务器 Python。

## 配置加载入口

Local 进程启动时由 `insertany3d.runtime_workers.load_local_environment` 读取仓库根目录的 `.env`；也可以用
`INSERTANY3D_LOCAL_ENV_FILE` 指定一个相对该仓库或绝对路径的文件。读取是增量的，已经由 shell 导出的变量优先；
默认 `.env` 不存在不阻止离线计划和假 worker，但显式指定的文件缺失、不是普通文件或格式错误会在启动时报告清晰错误。
Local `.env.example` 只包含远端连接、本地路径和非敏感默认值，不保存服务器缓存路径或 API key。

Server 只读取项目内 `.insertany3d/runtime.env`。远端 stage 启动和 Server 仓库的 `tools/verify_environments.sh` 使用同一相对位置，
并在执行前检查文件不是符号链接且 `HF_HOME`、`TORCH_HOME`、`MODELSCOPE_CACHE` 是已存在的 POSIX 绝对目录。
模板位于 Server 仓库的 `docs/server.env.example`；实际文件只留在服务器，不上传、不写入 runtime lock，也不由 Local
同步覆盖。Local 发起的 provenance/SSH 子进程会移除 API key 及 key-file 变量，避免敏感配置进入远端环境。

发布顺序固定为：临时目录上传、逐文件 hash 校验、原子替换、保留上传前备份、运行时预检、单 task canary。失败即停止。

常用检查：

```bash
git status --short
git diff --check
python3 tools/sync_remote_runtime.py check --source ../codex_remote_tools
python3 tools/sync_remote_runtime.py verify-lock
ssh <server> 'git -C <server-checkout> status --short && git -C <server-checkout> diff --stat'
```

本地按调度器核心、CLI/审核、评测、workspace/runtime、文档依赖分批提交；服务器只提交已审计的 metrics、部署脚本和 tools 快照；Unity 独立提交。
