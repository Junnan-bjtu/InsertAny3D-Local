# 排错

先按以下顺序看证据，不要一开始就重跑昂贵模型：

1. 本机 `workflow.log`。
2. 下载后的 `workflow.remote.log`。
3. 任务 `99_raw_pipeline/logs/batch.log`。
4. 各阶段 manifest、退出码和输入/输出哈希。
5. `03_gim/multiview_summary.png`、`05_pose/pose.json` 与 SAGS diagnostics。

## 常见问题

### Server 子模块未初始化

进入 Server 仓库运行 `bash tools/bootstrap_third_party.sh --check`。看到“未初始化”说明 gitlink 正常但源码尚未
下载；再执行不带参数的 bootstrap。Local 不包含这些第三方子模块，也不要在 Local 目录执行该脚本。

### Server 安装后子模块变脏

Bootstrap 本身不应改第三方源码。先检查是否在子模块内手工安装了 editable 源码、生成了未忽略文件，或
使用了旧 patch/overlay 脚本。不要直接清理，先保存 `git status` 和 diff。

### TRELLIS 没有输出

先检查脚本是否在参数解析阶段退出，再看 provider manifest 和进程退出码。后续 GIM 缺输入通常只是前一步
失败造成的连锁现象。

### `No module named triton`

这通常是 GIM 可选加速不可用，不一定代表匹配失败。以阶段最终状态和输出契约为准。

### SAGS 首次提示缺少 `sam_pt`

首次运行可能先报告缓存不存在，再自动提取。最终是否成功仍看 `inserted_object.ply`、diagnostics 和任务状态。

### 是否需要真实模型复测

优先使用假 worker、合成输入和已有成功调试包。只有这些证据不能覆盖关键接口，而且用户明确批准成本后，
才执行最小真实检查。

### worker 显示 `recovering`

这表示 SSH 请求可能已经送到服务器，但本机无法确认远端进程是运行、结束还是已经产出结果。调度器会继续
占用对应远端 GPU 槽，防止同一任务被重复启动。先执行恢复命令查看远端 PID/result；没有明确状态前不要用
`retry` 强行重跑，也不要手工删除数据库 lease。

### `--real` 在领取任务前退出

这是预检的正常保护。依次检查 Unity 可执行文件、每个 Project/Scene、公共包安装、Farm GUI 是否仍打开、
图片 API 地址、默认的 `~/.config/insertany3d/apiyi_key`（或 `APIYI_API_KEY` /
`APIYI_API_KEY_FILE` 临时覆盖）和三个 `INSERTANY3D_REMOTE_*` 路径。预检失败不会消耗 Unity attempt。
