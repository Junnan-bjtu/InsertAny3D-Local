# 运行代码更新 SOP

本 SOP 适用于本地 InsertAny3D、服务器 checkout 和 Unity Farm 之间的代码变更。三端没有永久主从关系；同步前先审计，再发布。

## 更新步骤

1. 保存三端 Git status（Unity 当前无 Git 时保存目录清单）、Git HEAD、文件 mtime 和 SHA-256。
2. 服务器有未提交改动时先保存 git diff 和备份目录，不允许直接覆盖。
3. 逐文件分类：相同、单边修改、双边冲突、服务器/Unity 专属文件、运行产物。
4. 双边冲突必须人工合并并在本地测试；不能用最近 mtime 自动决定来源。
5. 选定来源生成 runtime lock，上传临时目录并逐文件校验 hash。
6. 原子替换后执行私有环境预检和单 task canary；失败即停止并使用上传前备份回滚。

## 代码边界

- Git 管理：InsertAny3D 编排器、契约、评测和发布工具；服务器专属 metrics；未来独立的 Unity Farm 仓库。
- 不提交：`.insertany3d/runtime.env`、模型缓存、`.insertany3d-backups/`、`.insertany3d-runtime-staging/`、Unity `Library/`、日志、运行结果和验收资料。
- submodule 只提交确认过的子模块 commit，不提交服务器 detached HEAD 的偶然变化。

## 服务器紧急修复

服务器直接修改只允许作为临时修复。完成后必须导出 patch，带回本地分支比较、合并、测试和 commit；在此之前禁止下一次同步覆盖服务器。

## 常用命令

```bash
git status --short
git diff --check
python3 tools/sync_remote_runtime.py check --server-root ../InsertAny3D-Server
python3 tools/sync_remote_runtime.py verify-lock
# 迁移期回退/比较旧镜像时才显式指定：
python3 tools/sync_remote_runtime.py check --source ../codex_remote_tools
ssh <server> 'git -C <server-checkout> status --short && git -C <server-checkout> diff --stat'
```

服务器私有缓存和凭据继续保存在 `.insertany3d/runtime.env`，不由代码同步覆盖。
