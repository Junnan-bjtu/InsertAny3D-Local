# 工作流程

InsertAny3D 把 Unity 私有场景的渲染输入交给服务器处理，再把生成物和位姿送回 Unity。Unity 工程、场景、
原始输入与实验结果保持私有；服务器只接收完成当前任务所需的派生文件。

## 正式顺序

1. Unity 为一个任务渲染 `left`、`center`、`right` 三视图、深度和相机参数。
2. 图片编辑阶段生成候选中心图。默认由用户逐任务确认，每页 5 项；也可以显式启用自动确认。
3. 服务器 3D provider 生成包含锚点和新物体的 Gaussian 资产。
4. 服务器渲染对齐三视图，通过 GIM 匹配并联合估计一个相似变换。
5. SAGS 使用六个环拍视角提取新物体，输出 `inserted_object.ply`。
6. Unity 导入 PLY、应用 `pose.json`，再渲染评估图。

步骤 1 和步骤 6 使用同一个全局水平角度配置，默认 24 度；整批任务必须一致。12 度、48 度或其他有效值
通过配置修改，不能为单个任务偷偷改变算法默认值。

## 当前入口

本机批次状态和评测使用已经实现的统一入口：

```bash
python3 tools/insertany3d.py batch --help
python3 tools/insertany3d.py evaluate --help
```

统一 worker 已具备 Unity、APIYi、SSH 远端阶段和逐文件哈希传输的执行适配器；真实调用必须显式选择
`batch worker --real`。默认手动审核会让 worker 在编辑图生成后返回，接受对应任务后重跑相同命令即可继续。
服务器运行时代码由独立的 Server 仓库维护；需要直接调试服务器脚本时，应进入 Server checkout 并使用其
README 中配置好的 Python 环境：

```bash
cd /path/to/InsertAny3D-Server
"$INSERTANY3D_PYTHON" "$INSERTANY3D_SERVER_ROOT/tools/run_insert_pipeline.py" --help
"$INSERTANY3D_PYTHON" "$INSERTANY3D_SERVER_ROOT/tools/run_insert_batch.py" --help
```

`run_insert_pipeline.py` 处理单个任务，`run_insert_batch.py` 读取 jobs JSON 并串行启动多个独立任务。真实
canary 通过前不会删除这两个入口。详细参数见 [CLI 参考](cli-reference.md)，文件契约见
[流水线契约](pipeline-contract.md)。

## 成本受控的验证

日常开发和持续集成不应启动 TRELLIS、GIM、SAGS、分割模型或付费图片接口。任务队列使用假命令验证并发、
心跳、重试、恢复、取消与清理；阶段兼容性回放已有成功产物。只有低成本检查无法证明关键契约且用户明确
同意时，才安排最小真实运行。第一个正式任务同时承担生产 canary（先行观察任务）的作用。
