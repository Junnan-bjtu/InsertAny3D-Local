# 安装

本文只说明公开仓库的可复现安装方式。模型推理会消耗大量时间、显存和下载流量；完成源码与环境检查，
不代表必须立刻运行模型。

## 1. 准备环境

推荐 Linux x86_64、NVIDIA GPU、Git、Git LFS、uv 和 C/C++ 编译工具。当前安装脚本默认查找
CUDA 12.4 与 CUDA 11.8，可通过 `CUDA_12_HOME` 和 `CUDA_11_HOME` 改成实际路径。

完整模型环境需要较大的磁盘和显存。只审阅代码或测试任务队列时，不需要下载模型权重。

## 2. 克隆 Local 与 Unity 包

```bash
git clone --recurse-submodules https://github.com/Junnan-bjtu/InsertAny3D-Local.git
cd InsertAny3D-Local
```

如果克隆时没有使用 `--recurse-submodules`，只需要执行：

```bash
git submodule update --init --recursive
```

Local 不包含 TRELLIS、GIM、SAGS 的第三方源码和环境；这些内容由
[InsertAny3D-Server](https://github.com/Junnan-bjtu/InsertAny3D-Server) 管理。Unity 包只是 Local 的固定
子模块，不包含 Farm 场景。

## 3. 安装 Local Python 环境

先安装 uv（不要依赖系统 `python3` 或 conda 管理本项目环境）：

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

运行批次计划、审核和评测不需要 CUDA 或模型：

```bash
uv sync --locked
python3 tools/insertany3d.py --help
```

## 4. 配置 APIYi key

图片编辑和 GPTEval 共用同一个 APIYi key。推荐保存为权限 `0600` 的文件：

```bash
mkdir -p ~/.config/insertany3d
chmod 700 ~/.config/insertany3d
chmod 600 ~/.config/insertany3d/apiyi_key
```

文件内容只放 key 本身，不要提交到 Git。

## 5. 验证 Local

低成本检查，不启动模型。控制面环境由 `uv.lock` 复现：

```bash
uv run --locked python tools/insertany3d.py --help
uv run --locked python -m unittest discover -s tests -q
uv run --locked python tools/sync_remote_runtime.py verify-lock
```

这些检查不启动模型，也不访问付费接口。Local 中的 `tools/` 运行时镜像由 Server 发布树生成，修改它之前必须
先按 `docs/code-authority-and-sync.md` 做逐文件同步。

## 6. Server 环境

```bash
cd /path/to/InsertAny3D-Server
bash tools/bootstrap_third_party.sh --check
bash tools/verify_environments.sh
```

Server 的第三方安装、CUDA 检查和模型下载只能在 Server 环境中执行；具体解释器和缓存位置见 Server 仓库 README。

## 安装规则

- 不在 Unity 子模块中执行 `git pull` 或切到任意最新版；Local 的 gitlink 是版本真相。
- Server 的第三方安装不要从 Local 的 `tools/` 镜像执行；先进入 Server 仓库。
- 安装失败时先报告具体依赖和退出码，不通过升级整套依赖来绕过固定版本。
- 不提交 `.venv`、权重、缓存、输入场景或运行结果。
