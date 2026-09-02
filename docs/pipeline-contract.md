# 流水线契约

本文描述 Unity、本机桥接和服务器之间需要长期稳定的文件与状态，不要求调用方了解内部模型实现。

## 输入

每个任务使用稳定且安全的任务 ID，例如 `Task_001`。最小 Unity 输入包含：

```text
<task>/
├── step1/
│   ├── left/image.png    image.raw    image.camera.json
│   ├── center/image.png  image.raw    image.camera.json
│   └── right/image.png   image.raw    image.camera.json
├── edited/center.png
└── task_manifest.json
```

三个视角的 RGB、float32 深度和相机文件必须同序。`edited/center.png` 必须来自已确认的图片编辑结果；自动
确认模式也要记录响应和输入哈希，不能只留下最终图片。

## 服务器输出

```text
<run-root>/<task-id>/
├── 01_segmentation/
├── 02_trellis/
├── 03_rendered_3dgs/
├── 03_sags_views/
├── 04_gim/
├── 05_pose/pose.json
├── 06_sags/inserted_object.ply
├── logs/
└── manifest.json
```

任务只有同时满足以下条件才可视为可导入：

- 顶层任务状态为 `ready`；
- `05_pose/pose.json` 存在且 `status` 为 `ready`；
- `06_sags/inserted_object.ply` 存在且属于本次 task/attempt；
- 完整调试包已经生成，或明确记录为可重试的诊断收集失败。

Unity 应导入 `06_sags/inserted_object.ply`，不能导入仍包含锚点的 `02_trellis/sample.ply`。

## 算法不变量

- Pose 先去掉不能被其他视角支持的点，再用全部有效点拟合一次联合相似变换；不是三个视角各算一个变换
  后互相否决。
- 正式 SAGS ring6 使用六个视角与 3/6 多数票。
- 非源视角标注默认至少覆盖中心 Gaussian 几何先验的 25%；明显遮挡记作 `unknown`，不参与投票。
- 直接调用 `run_sags_text.py` 的兼容默认仍是 2 票。正式流水线必须显式传入其 3/6 配置。
- `No module named triton` 通常表示 GIM 可选加速缺失，阶段是否成功仍以最终状态判断。

## 失败与重试

失败任务保留 manifest、日志和已提交的只读产物。重试写入新的 attempt 暂存目录，只有完整校验通过后才能
原子提交为当前结果；不得让多个任务共享同一个输出目录。
