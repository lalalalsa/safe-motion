# SafeMotion：VLA 到 UR5 之间的可解释安全执行层

SafeMotion 接收 VLA 生成的 50 点绝对关节轨迹，并在确定性的 Mock Robot 上以
20 Hz 闭环执行。每个控制周期都重新读取实际关节状态、生成名义速度、预测完整
机械臂是否保持在 keep-in workspace 内，然后原样执行、缩放或停止。

本仓库刻意保持轻量：只使用 NumPy、Matplotlib 和 pytest，不依赖 ROS、MoveIt、
MuJoCo 或优化求解器。目标是让安全因果链容易审查和解释。

## 快速开始

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pytest -q

python -m safe_motion.replay scenarios/free_space.json \
  --output artifacts/free_space_replay.json \
  --plot artifacts/free_space.png

python -m safe_motion.replay scenarios/real_03_tcp_safe_mid_link_unsafe.json \
  --output artifacts/real_03_replay.json \
  --plot artifacts/real_03.png
```

## 一条轨迹是怎样执行的

`action_chunk.shape == (50, 6)`。第 `i` 行是第 `i` 个未来时刻的六轴**绝对关节
目标**，不是增量或速度。对每个目标执行：

1. 从 Mock Robot 读取当前实际状态 `q`；
2. 计算 `q_dot_nom = (q_target - q) / dt`，再做关节速度限幅；
3. 依次测试名义速度的 100%、75%、50%、25% 和 0%；
4. 每个候选命令在一个控制周期内采样 10 个子步；
5. 每个子步检查关节限制和完整机械臂的 workspace 约束；
6. 执行最大的安全比例，并在执行后再次检查安全不变量；
7. 下一目标继续基于 Mock Robot 的实际状态，而非假设上一目标已到达。

因此，安全轨迹基本不被修改；不安全轨迹会减速或停在边界内。当前缩放器只沿
原动作方向寻找安全命令，不负责绕行或全局重规划。

## 正运动学与全身检查

`safe_motion/kinematics.py` 使用题目给出的标准 DH 参数。每个 4×4 齐次变换表示
相邻坐标系之间的旋转和平移，沿运动链依次相乘，得到 base、六个关节和 TCP 的
三维位置。题目没有给额外工具偏置，所以 `tcp` 与最后一个 DH 帧重合。

机械臂被建模为相邻节点之间的零厚度线段。矩形 workspace 是凸集，因此一根线段
的两个端点均在盒内时，整根线段也在盒内；对这个简化模型，检查所有活动节点等价
于解析检查所有连杆，而不会出现采样间距造成的漏检。

固定 base 位于 `z=0`，而题目示例 workspace 从 `z=0.05` 开始。默认配置将它解释
为桌面上方**活动机械臂**的 keep-in 区域，所以固定 pedestal 不参与检查，从
`joint_1` 开始检查。这个约定集中在 `SafetyConfig.first_checked_node`，可配置且有
明确说明，不是隐藏特例。

## 输入契约与 Fail-safe

Mock Robot 构造前一次性拒绝：

- 非 `(6,)` 的 `joint_state`；
- 非 `(50, 6)` 的 `action_chunk`；
- NaN、Inf 或非正 `action_hz`；
- 越过配置关节范围的状态或目标；
- 首目标或相邻目标超过阈值的突跳。

安全配置本身也在同一入口验证，包括 workspace 上下界、关节范围、正的有限速度
上限、非负安全裕量、至少一个路径子步，以及从 100% 降到 0% 的合法缩放序列。

安全过滤器无解、抛出异常、返回错误 shape 或非有限速度时，Replay 只发送六维零
速度，并记录停止原因，绝不会回退执行原 VLA 动作。Mock Robot 更新后如果再次检查
发现不安全，Replay 立即中止并报告安全不变量已破坏。

## 输出与测试

Replay 输出题目要求的：

- `total_steps`、`executed_steps`；
- `modified_steps`、`stopped_steps`；
- `minimum_workspace_margin`；
- `maximum_joint_velocity`；
- `final_joint_state`；
- 名义/实际关节轨迹和逐步安全决策日志。

测试覆盖输入错误、FK 参考值、Mock Robot 数学模型、TCP 安全但中间连杆不安全、
随机有限命令、安全过滤器故障注入、free-space 不修改，以及题目仓库提供的三个
真实边界场景。测试断言的是 Mock Robot **实际执行状态**，而不是求解器状态。

## 能保证什么，不能保证什么

在题目规定的标称 DH、零厚度连杆、固定周期、完美跟踪、零延迟和确定性 Mock
Robot 假设下，本实现保证已执行状态满足配置的关节范围及活动运动链 keep-in
workspace；无法可靠判断安全时停止。

它不能证明真实 UR5 的功能安全。真实部署还需要考虑出厂标定误差、工具和连杆
实体半径、状态估计误差、控制器跟踪误差、通信延迟、加速度与制动距离、动力学、
人员和环境碰撞、硬件 protective stop / 急停以及相应安全认证。

## 代码阅读路线（面试讲解顺序）

1. `validation.py`：为什么非法轨迹不能进入机器人；
2. `kinematics.py`：关节角如何变成空间节点；
3. `geometry.py`：为什么必须检查全身；
4. `safety_filter.py`：如何尽量保留动作以及如何 fail closed；
5. `replay.py`：50 点如何基于实际状态闭环执行；
6. `tests/`：每条安全不变量如何用证据验证。

## 开源方案调研与取舍

- Pinocchio 能从 URDF 计算 FK，但对本题指定的短 DH 链属于额外依赖和抽象；
- RTC-Anything 面向 action chunk 融合与 ROS 真机运行，不提供本题全身 keep-in 保证；
- MuJoCo/cuRobo 类项目提供完整规划或 QP，但远超两天作业范围；
- 完整 agent-to-robot runtime 的 fail-closed 原则值得借鉴，但不适合作为本题代码基底。

因此核心实现为独立编写，没有复制第三方仓库代码。这样也让隐藏测试中的 FK、全身
安全和失败路径可以直接审查。
