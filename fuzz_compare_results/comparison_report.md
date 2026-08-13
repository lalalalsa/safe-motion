# SafeMotion 三版横向对比报告（codex vs deepseek vs kimi）

> 测试人员视角 · 同一批全新数据（seed=556800785，100 组）· 本地黑盒测试，未上传 GitHub。

## 测试方法

- 100 组：88 随机游走（loguniform 扰动覆盖安全→越界）+ 8 非法输入（NaN/Inf/shape×2/维度/action_hz=0/突跳/joint_state NaN）+ 4 关节越界。
- 三版用**同一 seed**，数据完全一致；seed 由命令行传入，脚本内不写死，防泄露。
- 每组记录偏差（joint_dev/tcp_dev/final_gap）+ 安全不变量（越界/破限位/非有限速度/崩溃）。

## 汇总对比

| 指标 | codex | deepseek | kimi |
|---|---:|---:|---:|
| pass | 88 | 92 | 88 |
| fail | 0 | 0 | 0 |
| rejected | 12 | 7 | 12 |
| **crash** | **0** ✅ | **1** ❌ | **0** ✅ |
| 违规行 | 无 | 无 | 无 |
| mean_modified | 3.86 | 0.34 | 0.26 |
| mean_stopped | 3.55 | 3.90 | 3.72 |
| mean_joint_dev | 0.057 rad | 0.092 rad | 0.046 rad |
| mean_tcp_dev | 0.017 m | 0.022 m | 0.014 m |

## 三个关键发现

### 1. 唯一 crash 来自 deepseek（action_hz=0）

deepseek 版 `replay.py:53` 的 `dt = 1.0 / action_hz` 排在输入检查**之前**，`action_hz=0` 时 `ZeroDivisionError` 崩溃，而不是被拒绝。

codex 与 kimi 都把输入检查（含 `action_hz>0`）放在**所有计算之前**，均正确拒绝：
- codex：`action_hz must be finite and positive`
- kimi：`action_hz 必须为正有限值`

**这违反题目第 3 节「非法输入不得进入 Mock Robot」——deepseek 是唯一一个会让非法输入直接崩程序、而非拒绝的版本。**

### 2. 关节目标越界：codex = kimi = 拒绝，deepseek = 钳位

| 策略 | codex | deepseek | kimi |
|---|---|---|---|
| 关节目标越界 | 拒绝（rejected） | 钳位后继续（pass） | 拒绝（rejected） |

codex 与 kimi 把「目标越界」当作非法输入整体拒绝（题目第 3 节字面含义）；deepseek 选择钳位继续（题目第 7 节「最小修改」）。两种都满足「Mock Robot 不进非法状态」，但语义相反——这是三版最需要你面试时想清楚怎么讲的分歧。

### 3. 最小干预风格差异明显

- **codex**：modified=3.86，最倾向「缩放」而非「停止」。原因是它的 `max_joint_velocity=1.5`（另两版为 3.14），速度基数小，缩放档位更容易命中安全。
- **kimi**：modified=0.26，靠**二分细化**把越界干预精确到最小 scale，`min_margin` 逼近 0（0~1e-5）。
- **deepseek**：modified=0.34，越界时几乎直接 `stop`（固定档位，无二分）。

## 完整度对比（提交可用性）

| 项 | codex | deepseek | kimi |
|---|---|---|---|
| tests/ | ✅ 7 个测试文件 | ✅ 6 个 | ❌ 无 |
| README | ✅（+INTERVIEW_GUIDE） | ✅ | ❌ 无 |
| 5 场景 | ✅ 含 workspace_boundary | ✅ 5 个 | ❌ 仅 3 个 real |
| pyproject | ✅ | ✅ | ❌ 无 |
| 可视化 artifacts | ✅ | ✅ | 部分 |

## 结论

- **安全性**：三版在随机数据上 0 违规、0 fail，安全过滤核心都正确。
- **唯一 bug**：deepseek 的 `action_hz=0` 崩溃（codex/kimi 均正确拒绝）。
- **提交就绪度**：codex 最完整（7 测试 + 5 场景 + README + GUIDE）；kimi 核心算法最精细（二分+子步+三层防线）但缺 tests/README；deepseek 有 bug 且测试缺失。
