# 评测

当前正式评测只运行 GPTEval。GPTEval3D_v2 因没有可用的对照方法而延期；除非后续任务明确启用，否则不安装、
不调用，也不把它的缺失视为失败。

仓库中的 HPSv2 脚本属于历史兼容代码，不再是主流程指标。TRELLIS 朝向搜索已改用独立的
`tools/clip_image_similarity.py`，不再 import HPSv2；历史文件暂时保留供外部消费者迁移，但 README 不再
把它列为正式入口。

开发测试使用固定假响应验证结果解析、汇总和断点恢复，不调用付费评测接口。正式任务的评测请求、响应、
模型名和输入哈希必须写入任务清单文件（manifest），失败不能伪装成空分数或 `ready`（已完整）。

## 评分维度

评价入口默认只启用两项：

- `visual_quality`：视觉质量，例如渲染瑕疵、边缘、材质、光照和跨视角一致性。
- `geometric_accuracy`：几何准确性，例如形状、透视、深度、悬空、穿插和跨视角几何一致性。

`insertion_rationality`（插入合理性）默认关闭。只有确实需要评价物体的语义、尺度、姿态、摆放和接触关系时，
才在命令中显式加入它：

```bash
python3 tools/insertany3d.py evaluate run /path/to/eval6-results \
  --output /path/to/evaluation-output \
  --dimensions visual_quality insertion_rationality geometric_accuracy \
  --fake-score 7
```

`--dimensions` 同样适用于 `evaluate plan/run/status/summarize` 和 `batch evaluate`。它可以接收空格分隔的值，
也可以写成逗号分隔的一项。维度会写入请求缓存键；因此默认二维结果不会复用旧的三维结果。对同一个输出目录
执行 `status` 或 `summarize` 时，必须传入与 `run` 相同的维度配置。

## 先检查输入，不发送请求

`eval6` 是每个任务固定的六个观察角度。每个角度有一张插入前图片、一张插入后图片和一份相机参数，
所以一个任务共有 12 张评测图片。下面的命令检查图片、相机参数、文件哈希和视角配置，并显示预计请求数；
它不会访问网络：

```bash
python3 tools/insertany3d.py evaluate plan /path/to/eval6-results \
  --output /path/to/evaluation-output
```

查看已有响应的完成度同样不会联网：

```bash
python3 tools/insertany3d.py evaluate status /path/to/eval6-results \
  --output /path/to/evaluation-output
```

默认正式分母是 60 个任务、12 个场景、每场景 5 个任务。缺少任何结果时状态为 `partial`（未完成），
不会把已完成子集标成完整结果。

## 本地假响应

开发和流程检查使用固定假分数。它只验证缓存、断点续跑和汇总，不代表真实质量：

```bash
python3 tools/insertany3d.py evaluate run /path/to/eval6-results \
  --output /path/to/evaluation-output \
  --fake-score 7
```

汇总命令只读取本地缓存，不访问网络：

```bash
python3 tools/insertany3d.py evaluate summarize /path/to/eval6-results \
  --output /path/to/evaluation-output
```

`run` 和 `summarize` 都会写出以下四个文件：

- `batch_summary.json`：完整批次汇总。
- `task_scores.jsonl`：每个任务一行的详细分数、理由和 12 张输入图片的相对文件名。
- `scene_scores.csv`：每个场景的简表。
- `gpteval_summary.xlsx`：便于人工查看的 Excel 工作簿。

Excel 的“场景汇总”工作表每个场景一行，按启用维度给出场景均分和场景总分，末尾的“总体”行给出各维度
场景宏平均和总分。“任务明细”工作表每个任务一行，包含提示词、12 张原图/插入图的文件名、每项分数与
理由、任务总分和状态；工作簿不会嵌入图片。这里的“总分”是当前启用维度分数的算术平均值。
如果同一批次包含多个方法，每个方法各有一行“总体”；批次尚未完成时该行会标为 `partial`，此时分数只
汇总已经得到结果的场景，不能当作最终正式分数。

## 正式付费调用

`run` 默认拒绝真实请求。正式调用必须明确加入 `--allow-paid-api`。APIYi key 推荐只配置一次：

```bash
mkdir -p ~/.config/insertany3d
chmod 700 ~/.config/insertany3d
${EDITOR:-nano} ~/.config/insertany3d/apiyi_key
chmod 600 ~/.config/insertany3d/apiyi_key

python3 tools/insertany3d.py evaluate run /path/to/eval6-results \
  --output /path/to/evaluation-output \
  --allow-paid-api
```

文件中只写一行 APIYi key。之后评价和真实图片编辑 worker 都会自动读取，无需每次输入。密钥读取顺序是：
`APIYI_API_KEY` 临时覆盖、`APIYI_API_KEY_FILE` 指定文件、兼容的 `GEMINI_API_KEY_FILE` 指定文件、默认文件
`~/.config/insertany3d/apiyi_key`，最后才是旧的 `GEMINI_API_KEY` 和 `BEE_API_KEY`。因此默认文件中的有效 key
不会再被遗留的无效 `BEE_API_KEY` 抢先使用。

显式指定的文件不存在、不可读或为空时会直接报错；默认文件只有在不存在时才会退回旧环境变量。Linux/WSL
下文件若允许同组或其他用户访问，也会拒绝使用并提示执行 `chmod 600`。密钥不会写入评测结果或日志。

默认会对可重试的网络或服务器错误再试两次。需要严格限制调用次数时加入 `--retries 0`，这样每个尚未缓存的
项目最多发起一次调用。

响应按输入内容生成的唯一键保存在本地。重新运行时只请求尚未成功缓存的项目；两个进程同时处理同一个
请求时也会使用文件锁避免重复付费。

请求失败时，命令行会显示本次使用的密钥来源，例如 `paid_api:~/.config/insertany3d/apiyi_key`，以及每个错误
JSON 的绝对路径；这里只显示环境变量名或 key 文件路径，不显示密钥值。错误 JSON 中保存了服务端状态码和具体消息，可按命令行给出的路径
直接查看。更换或修复密钥后重跑同一条命令，只会补尚未成功的响应。
