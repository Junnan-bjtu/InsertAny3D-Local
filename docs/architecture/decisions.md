# 架构决策

## 当前有效

- 服务器部署脚本没有永久的本地或服务器权威方；发布前按 `docs/code-authority-and-sync.md` 逐文件审计，
  由审计记录选择来源并用 hash lock 校验。服务器工作树不能被整目录覆盖。
- 第三方源码只有五个项目 fork 的固定 Git 子模块这一条来源；旧 patch/overlay 不再参与 bootstrap。
- 统一 CLI 已负责持久批次状态和 GPTEval；单任务 `run_insert_pipeline.py` 与批量 `run_insert_batch.py` 在真实
  GPU 阶段适配器完成兼容验证前继续保留。
- Pose 使用点级跨视角一致性和一次联合相似变换。
- 正式 SAGS 使用 ring6、3/6 投票、0.25 几何先验覆盖门控和遮挡 `unknown`。
- 日常测试以任务队列、假 worker 与已有产物回放为主，不额外运行昂贵模型或付费 API。
- 当前正式指标只有 GPTEval；GPTEval3D_v2 延期，HPSv2 退出主流程。

## 发布前仍需完成

- 保存服务器脏工作树后部署已锁定的 runtime 快照，核对上传前后哈希并记录公开提交和 defaults 哈希。
- 在干净 clone 中递归初始化五个子模块，证明 bootstrap 重复执行幂等。
- 给每个删除候选补齐 E1-E5 使用证据和可恢复的 pre-cleanup tag；未满足门禁的文件继续保留。
