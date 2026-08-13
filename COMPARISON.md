# 跨实现盲测报告（fuzz compare）

> 三个独立实现、同一批 100 组数据、同一套安全不变量判定。
> 目的：模拟面试方的"隐藏场景验收"，并检验实现差异。

## 方法

- **数据**：`fuzz_compare.py --seed 289838560` 确定性生成 100 组场景——
  - 88 组随机游走轨迹：起始位形在安全位形附近 rejection sampling，
    步长 sigma 在对数域均匀采样（1.5e-3 ~ 0.10 rad，跨两个数量级）；
  - 8 组非法输入：NaN、Inf、shape 过短/过宽、joint_state 维度错、
    action_hz=0、起始突跳 1 rad、joint_state 含 NaN；
  - 4 组关节越界 ramp：J6→±7 rad、J3→7 rad、J1→−7 rad（超出 ±2π 限位）。
- **判定**（对每组）：全身 min margin ≥ 0、关节限位不越、速度全部有限、
  无未捕获异常。 verdict ∈ {PASS, FAIL, REJECTED, CRASH}。
- **公平性**：同一 seed ⇒ 各实现面对完全相同的数据；harness 只依赖各实现的
  公开接口（`run_scenario` / FK / 全身检查）。

## 结果汇总

| 指标 | codex | deepseek | kimi（缩放） | kimi（QP） |
|---|---:|---:|---:|---:|
| pass | 88 | 92 | 88 | 88 |
| fail（安全违规） | **0** | **0** | **0** | **0** |
| rejected | 12 | 7 | 12 | 12 |
| **crash** | 0 | **1** | 0 | 0 |
| mean_joint_dev (rad) | 0.0285 | 0.0646 | 0.0188 | **0.0103** |
| mean_tcp_dev (m) | 0.0079 | 0.0134 | 0.0057 | **0.0050** |
| mean_modified_steps | 2.98 | 0.51 | 0.36 | 3.05 |
| mean_stopped_steps | 2.61 | 2.83 | 2.68 | **0.0** |

明细：`fuzz_compare_results/fuzz_<target>.csv`；汇总：同目录 `*_summary.json`。

## 发现

### 1. 三个实现的核心安全逻辑都正确

88 组随机游走全部 0 违规——包括步长 0.1 rad/步的激进轨迹。
"越界干预发生的位置"在各实现间完全一致。

### 2. deepseek 版有一个 fail-safe 缺口（crash × 1）

第 93 组 `invalid_hz`（action_hz=0）触发 `ZeroDivisionError` 崩溃。
根因：`deepseek/safe_motion/replay.py` 在**输入检查之前**先计算
`dt = 1.0 / action_hz`——校验顺序颠倒，构造输入即可打崩。
这正是题目验收第 5 条（人为制造失败，系统应受控停止而非崩溃）针对的情形。
修复只需一行：把 `dt` 的计算挪到 `check_input` 之后。
codex / kimi 两版均先验证后使用，干净拒绝（exit 2）。

### 3. 关节越界的两种处理语义（设计分歧，非对错）

12 vs 7 的 rejected 差异全部来自 4 组关节越界：

| | 拒绝（codex / kimi） | 钳位（deepseek） |
|---|---|---|
| 依据 | 题目第 3 节"非法输入不得进入 Mock Robot" | 题目第 7 节"最小程度修改" |
| 行为 | 整体拒绝，机器人不动 | 目标钳到 ±2π 后继续执行 |
| 语义 | fail-fast，问题暴露给上游策略 | 保留可执行性 |

两种都满足 10.4"Mock Robot 不得进入非法关节状态"。

### 4. QP 过滤（kimi 加分项）的任务连续性最好

同一安全保证（0 违规）下：QP 的 100 组**停滞步数为 0**（缩放类实现平均
2.6~2.8 步/组），平均 TCP 偏离最小（5.0 mm）。代价是依赖线性化与求解器，
因此 kimi 版把 QP 放在"提案层"，非线性精确验证 + 缩放回退兜底。

## 结论

- 安全性：kimi / codex 全绿；deepseek 有 1 个可复现崩溃（非越界，是异常泄漏）。
- 最小修改：kimi-QP > kimi-缩放 > codex > deepseek（按 mean_tcp_dev）。
- 可复现：`python fuzz_compare.py --target <t> --seed 289838560`。
