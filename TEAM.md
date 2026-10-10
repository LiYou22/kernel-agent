# 分工（截止 2026-10-10 21:00 EDT）

每个 seat 都有自己的芯片和模型服务，所以 benchmark 按 seat 分成 4 份同时跑，代码只由 seat-130 修改。

| seat | 人 | 主要任务 | 这个 pod 跑的 benchmark |
|---|---|---|---|
| **130** | 主力开发 | agent / 检查器；L8–10 检查器；修复 red team 找到的漏洞；合并 | `ours` L1–4 |
| **132** | 实验与分析 | 盯 4 个 pod 的 benchmark，收结果，出表和图，置信度校准 | `naive` L1–4 |
| **133** | red team | 写骗过检查器的 kernel（漏判）和被冤枉的正确 kernel（误判） | `ours` L5–7 |
| **134** | 报告与 demo | 一页复现说明、demo 主线、Stage B（NKI）对照章节、交付清单 | `naive` L5–7 |

`ours` = 我们的 agent（翻译成指令 + 已试清单 + 坏回复拦截）；`naive` = 对照组（检查器报告原样发回）。
两组除了反馈方式，其它完全一样（停止规则、留出卷、打分）。每遍换一种任务说法（见 notes E10）。

## 时间点

| 时间 (EDT) | 检查点 |
|---|---|
| 14:00 | 4 个 pod 都在跑 benchmark |
| 16:00 | L1–7 第一版结果表（132） |
| 16:30 | L8–10 检查器完成（130），各 pod 加跑 L8–10 |
| 18:30 | 全部结果汇总（132） |
| 19:00 | 代码冻结 |
| 20:00 | 报告初稿（134） |
| 20:45 | 定稿提交 |

## 每个人第一步：在自己的 pod 上跑起来（约 10 分钟）

在笔记本上：

```bash
git clone git@github.com:LiYou22/kernel-agent.git && cd kernel-agent
# 粘贴频道里最新的 AWS 凭据（macOS/Linux 那段），然后：
aws eks update-kubeconfig --name hack-hyd --region ap-south-2
export POD=seat-13X                      # 换成你的 seat

# 1. 模型服务在不在？要看到 200
kubectl exec $POD -c app -- bash -c 'curl -s -m 5 localhost:8000/health -o /dev/null -w "%{http_code}\n"'
#    不是 200：kubectl exec -it $POD -- bash，然后 cd /workspace && ./serve.sh，等 READY（约 4 分钟）

# 2. 推代码 + 自测（应当全部 passed）
scripts/sync.sh
kubectl exec $POD -c app -- bash -c 'cd /workspace/stageA && python3 -m pytest -q tests 2>&1 | tail -1'

# 3. 启动你那一份（后台跑，断线不影响）
scripts/launch.sh ours 1 2 3 4      # seat-130
scripts/launch.sh naive 1 2 3 4     # seat-132
scripts/launch.sh ours 5 6 7        # seat-133
scripts/launch.sh naive 5 6 7       # seat-134

# 4. 看进度
kubectl exec $POD -c app -- bash -c 'tail -n 20 /workspace/stageA/runs/*.log'

# 5. 跑完后取回结果并提交（results/ 进 git）
scripts/fetch.sh && git add results/$POD && git commit -m "results: $POD" && git push
```

注意：
- **不要在请求里加 `seed`**：Neuron 版 vLLM 会直接崩溃（notes E5）。
- 同一个 pod 别同时跑两个 benchmark，模型服务最多并发 4 个请求，会互相拖慢。
- 凭据过期（`ExpiredToken`）就重新粘贴；pod 里的进程不受影响。

## 各自的任务细节

### seat-130：主力开发
- L8 layernorm、L9 窗口 attention、L10 卷积：reference、误差上界、刁难测试、手写/故意错误 kernel、翻译器覆盖。
- 处理 133 提交的漏洞（`kernels/redteam/`），每修一个加一条测试。
- 19:00 代码冻结。

### seat-132：实验与分析
- 汇总：`uv run python scripts/analyze.py ours=results/seat-130/<file>.jsonl naive=results/seat-132/<file>.jsonl --tex notes/generated/results-L1234.tex`（L5–7 同理用 133、134 的文件）。
- token 图：`uv run python scripts/plot_tokens.py results/seat-130/<file>.jsonl`。
- 校准：verified 的比例、dev-only 的比例；有没有 verified 但其实错的（用 133 的 red team 测试复查）。
- 交付：结果表、图、失败分类计数（16:00 第一版，18:30 终版），放进 `notes/generated/`。

### seat-133：red team
挑战原文："如果检查器抓不到你自己写的 bug，它也抓不到模型的。"
- **漏判**：写错误但能通过检查的 kernel（只在特定形状错、绕过规则的写法、用掉误差上界的“合法”错误……）。
- **误判**：写正确但被判错/判违规的 kernel（不常见但合法的写法）。
- 每个发现：`kernels/redteam/l<N>_<名字>.py` + 文件开头一行注释说明，push 后告诉 130。
- 本地验证：`uv run python -m kagent.harness <N> kernels/redteam/<file>.py -v --fixes`。

### seat-134：报告与 demo
- 一页复现说明（环境、命令、预期输出）。
- demo 主线：一次失败与恢复。现成例子：L2 残块索引越界后修好；L3 的 E9（翻译器说不出原因 → 加诊断）。
- Stage B 对照：基于 `baselines/upstream-nki/`（上游 NKI agent 5 遍：L2 5/5，L1/L3 恒为 0.30=全部崩溃，L4 恒为 0.62），写“NKI 上失败是 API 误用，不是数值错误”一节。
- 交付清单逐项核对：agent、检查器（含容差理由）、测试集、失败分类、token 统计、复现说明。
- 20:00 初稿，20:45 定稿。

## 参考
- 设计与实验记录：`notes/notes.tex`（英文）；中文通俗版在仓库外，由 130 维护。
- 上游 NKI baseline：`baselines/upstream-nki/`（只读，不要在本机运行其中的脚本）。
