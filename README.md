# safe-motion — 同一道题的三个独立实现 + 跨实现盲测对比

> 面试作业：[VLA 机械臂安全执行器](https://github.com/gaoyuchen820/Interview-assignment/tree/main/Interview%20questions%20on%20embodied%20AI)。
> 本仓库用三个相互独立的 AI 编程助手分别实现同一道题，再用同一套
> 100 组盲测数据横向对比——一次"多实现对拍"的工程实验。

## 三个实现

| 目录 | 实现 | 特点 | 测试 |
|---|---|---|---|
| [`codex/`](codex/) | codex 版 | fail-closed 原型 + 可解释性改进 | `codex/tests/` |
| [`deepseek/`](deepseek/) | deepseek 版 | 速度缩放过滤 + 42 项测试 + 3D 可视化 | `deepseek/tests/`（42 项） |
| [`kimi/`](kimi/) | kimi 版 | 缩放+二分细化、子步检查、三层防线、QP 加分项、`--explain` 干预讲解 | `kimi/tests/`（72 项） |

每个子目录都是**独立可运行的完整项目**（各自的 README、requirements、
scenarios、tests、artifacts），进入对应目录按其 README 操作即可。

## 快速开始（以 kimi 版为例）

```bash
cd kimi
pip install -r requirements.txt
pytest -q
python -m safe_motion.replay scenarios/workspace_boundary.json --explain 28
```

## 跨实现盲测（fuzz）

```bash
python fuzz_compare.py --target kimi --seed <seed>                # scaling
python fuzz_compare.py --target kimi --seed <seed> --method qp    # QP
python fuzz_compare.py --target codex    --seed <seed>
python fuzz_compare.py --target deepseek --seed <seed>
```

- 同一 seed 生成完全相同的 100 组数据（88 随机游走 + 8 非法输入 + 4 关节越界），
  对所有实现逐组判定统一安全不变量（全身 margin、关节限位、速度有限、崩溃捕获）；
- 结果与完整分析见 **[COMPARISON.md](COMPARISON.md)**，原始数据在
  [`fuzz_compare_results/`](fuzz_compare_results/)。

### 头条结果（seed=289838560，100 组）

| 实现 | 安全违规 | 崩溃 | 平均 TCP 偏离 | 平均停滞步数 |
|---|---:|---:|---:|---:|
| codex | 0 | 0 | 7.9 mm | 2.61 |
| deepseek | 0 | **1**（hz=0 除零） | 13.4 mm | 2.83 |
| kimi（缩放） | 0 | 0 | 5.7 mm | 2.68 |
| kimi（QP） | 0 | 0 | **5.0 mm** | **0.0** |

## 目录结构

```
safe-motion/
├── README.md / COMPARISON.md      # 本文件 / 盲测完整报告
├── fuzz_compare.py                # 跨实现盲测 harness
├── fuzz_compare_results/          # 4 组 CSV 明细 + summary JSON
├── codex/                         # codex 实现（完整项目）
├── deepseek/                      # deepseek 实现（完整项目）
└── kimi/                          # kimi 实现（完整项目）
```
