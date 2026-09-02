# Unity 集成边界

Unity 包位于独立仓库 [InsertAny3D-Unity](https://github.com/Junnan-bjtu/InsertAny3D-Unity)。本仓库只维护
服务器代码和跨端文件契约，不保存实验场景。

## 对新手的安装方式

- 普通使用者：Unity Package Manager（UPM，Unity 的包管理器）从固定 Git tag 安装。
- 开发者：只维护一份本地 package checkout，各私有工程通过 `file:` 引用该目录，便于同步开发版本。

固定 tag 适合可复现使用；本地 `file:` 适合开发。不要在多个 Unity 工程中复制包源码形成不同版本。

## 私有内容

以下内容不上传 GitHub，也不永久保存在服务器公开仓库：

- Unity 工程和 `.unity` 场景；
- 原始实验素材、任务输入和人工确认记录；
- 运行结果与完整调试包；
- API key、远端令牌和本机绝对路径。

远端只接收任务执行所需的三视图、深度、相机参数、提示词和已确认编辑图。输出按 run ID 下载回本地后，
由本地保留策略处理。
