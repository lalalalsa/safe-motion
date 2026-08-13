# SafeMotion — VLA 机械臂安全执行器

> Take-home Assignment｜VLA 控制 UR5 在 CR2032 纽扣电池装配工作站中的安全执行原型。
>
> 纯软件实现：拦截 VLA 输出的 50 点关节轨迹，经过输入检查、名义速度生成、
> 全身安全过滤后交给确定性 Mock Robot 执行，保证机械臂全身（TCP + 六关节 +
> 连杆中间点）始终处于允许工作空间内。

```
50-point VLA trajectory
        ↓ 输入检查
        ↓ 逐点生成名义关节速度
        ↓ 全身安全检查 / 安全过滤（速度缩放·回溯）
        ↓ 安全速度
        ↓ Mock Robot
        ↓ 实际执行轨迹
```

---

## 快速开始

```bash
pip install -r requirements.txt

# 运行全部测试（42 项）
pytest -q

# 回放单个场景（终端输出统计指标）
python -m safe_motion scenarios/free_space.json
python -m safe_motion scenarios/real_01_lower_z_boundary.json
python -m safe_motion scenarios/real_03_tcp_safe_mid_link_unsafe.json

# 生成 3D 可视化图
python -m safe_motion scenarios/real_01_lower_z_boundary.json --plot --out artifacts
```

---

## 目录结构

```
safe-motion-deepseek/
├── README.md
├── requirements.txt
├── pyproject.toml
├── generate_scenarios.py        # 生成 free_space / joint_limit 场景
├── safe_motion/
│   ├── config.py                # DH 参数、关节限位、工作空间、安全阈值
│   ├── kinematics.py            # UR5 标称正运动学
│   ├── geometry.py              # 全身几何检查 + 连杆采样 + 带符号 margin
│   ├── safety_filter.py         # 输入检查 + 名义速度 + 速度缩放/回溯
│   ├── mock_robot.py            # 确定性 Mock Robot
│   ├── replay.py                # 闭环 Replay + run_scenario
│   ├── visualize.py             # 3D 可视化
│   └── __main__.py              # CLI 入口
├── scenarios/                   # 5 个测试场景（+ 面试方 real_02 上边界）
├── tests/                       # 42 项 pytest
└── artifacts/                   # 可视化输出
```

---

## 题目关键问答（README 必答 9 问）

### 1. `action_chunk` 的语义是什么？

`action_chunk.shape == (50, 6)`。每一行是未来某时刻六轴**绝对关节目标位置**
`[J1..J6]`（rad），不是关节增量、也不是速度。输出频率 20 Hz，相邻点间隔
0.05 s，50 个点对应约 2.5 s 的未来轨迹。`action_chunk[i]` 是第 `i` 个未来
控制点的绝对目标。

### 2. 50 点轨迹是如何逐步执行的？

采用闭环 Replay：**逐点**取出目标，基于**当前实际状态**生成名义速度，做安全
过滤，再 `MockRobot.step`。下一控制点基于上一步的实际状态继续，而不是假设
前一目标已完美到达：

```python
q = robot.get_joint_state()
for target in action_chunk:            # 恰好 50 点
    q_dot_nom  = nominal_control(q, target)      # (target - q) / dt，再限速
    q_dot_safe = safety_filter(q, q_dot_nom, workspace)
    q = robot.step(q_dot_safe, dt=0.05)
    check_full_body_safety(q)
```

### 3. UR5 正运动学如何实现？

标准 DH（classic DH）前向链：

```
T_i = RotZ(θ_i) · TransZ(d_i) · TransX(a_i) · RotX(α_i)
```

使用题目给定的标称 DH 参数，逐关节累积齐次变换，返回
`[base, joint_1 .. joint_6, tcp]` 共 8 个三维坐标。本题不考虑出厂标定误差与
工具 TCP 偏置，故 `tcp == joint_6`。FK 正确性由测试锁定：零位形参考坐标、
连杆长度不变量（`|a2|=0.425`、`|a3|=0.39225`、`d4=0.10915`、`d5=0.09465`、
`d6=0.0823` 等）逐项断言。

### 4. 如何检查整条机械臂，而不仅是 TCP？

对每个状态检查三部分：**TCP**、**六个关节位置**、以及**各连杆上的中间采样点**。

- 连杆采样方案（方案 A）：沿相邻关节点连成的线段，按 **2 cm** 间距等距采样，
  连同 7 个关节/TCP 点一起送入矩形工作空间判定。
- **为什么 2 cm 足够**：矩形 Keep-in Workspace 是**凸集**，连杆是**直线段**，
  因此「两端点（关节）都在矩形内 ⟹ 整条连杆都在矩形内」。2 cm 采样在几何上
  不产生漏检，采样仅作为实现层防御（浮点边界、未来扩展非凸工作空间）。
- **剩余误差**：若未来把工作空间换成非凸形状，纯采样可能漏检深度小于采样间距
  的浅越界；届时应改用解析线段 vs AABB 相交检测，或加密采样。

### 5. SafeMotion 如何修改不安全动作？

采用**速度缩放 / 回溯**（方法 A）：保持名义速度方向不变，按
`[1.0, 0.75, 0.5, 0.25, 0.0]` 依次尝试缩放系数，对每个系数预测下一状态
`q_next = q + scale·q_dot_nom·dt`，取**第一个（即最大的）使全身安全的系数**。
这等价于「对不安全动作做最小幅度修改」。关节目标越界则先在名义速度生成前把
目标钳位到限位边界。

### 6. 如果安全过滤失败，会发生什么？

**输出零速度（停止）。** 若所有缩放系数（含 `scale=0`）都无法得到全身安全的
下一状态，`safety_filter` 返回全零速度并标记 `stopped`。禁止在安全算法报错、
无解或产生非法结果时继续执行原始 VLA 动作。

### 7. Mock Robot 做了哪些简化？

统一假设 perfect joint tracking、固定时间步长、零通信/传感器/执行器延迟、
零噪声，不建模加速度、力矩、惯量、摩擦及真实底层控制器动力学。状态更新严格
为 `q[t+1] = q[t] + q_dot[t]·dt`，仅保留关节限位作为硬约束。

### 8. 当前实现能够保证什么？

- 所有进入 Mock Robot 的输入都通过了输入合法性检查；
- Mock Robot 永不突破配置的关节限位；
- 执行轨迹每一帧的**全身**（TCP + 六关节 + 连杆采样点）都位于矩形工作空间内；
- 安全过滤失败时输出零速度，绝不回退到原始不安全动作；
- free-space 轨迹不被明显修改。

### 9. 当前实现不能保证什么？

仅覆盖「UR5 标称 DH 模型 + 矩形 Keep-in Workspace + 确定性 Mock Robot +
20 Hz/50 点轨迹」这一边界。**不能**证明真实 UR5 在真实环境中的安全性——真实部署
还需考虑出厂标定误差、工具实际几何、状态估计误差、通信延迟、控制器跟踪误差、
制动距离、电机动力学、感知漏检、急停/Protective Stop 与功能安全认证。

---

## 测试场景

| 场景文件 | 语义 | 关键断言 |
|---|---|---|
| `free_space.json` | 自建安全轨迹（小幅正弦微扰） | 不被修改，margin ≥ 0 |
| `real_01_lower_z_boundary.json` | TCP 下沉破 z_min（面试方数据） | 执行后全身安全 |
| `real_02_upper_z_boundary.json` | TCP 上升破 z_max（面试方数据） | 执行后全身安全 |
| `real_03_tcp_safe_mid_link_unsafe.json` | TCP 安全但肘部越 x_min（面试方数据） | **识别连杆越界** |
| `joint_limit.json` | J3 平滑越出关节下限 | Mock Robot 不越限 |
| （代码级） | NaN / Inf / 错误 shape / 非 50 点 / 突跳 | 拒绝，不进 Mock Robot |

测试重点验证的是**最终执行轨迹是否安全**，而非「函数能跑」。

---

## 一个安全干预示例（便于面试讲解）

`real_01` 中第 28 帧起，VLA 让 J3 持续下沉，TCP 逼近 `z_min = 0.05`：

1. 当前 `q` 的 TCP 已在 `z ≈ 0.06`；
2. VLA 目标继续把 J3 往下压，预测 TCP 将到 `z ≈ 0.04`（越界）；
3. `q_dot_nom = (target - q)/dt` 被限速后仍会带 TCP 穿出 `z_min`；
4. `full_body_check` 预测下一状态 `min_margin < 0` → 不安全；
5. 速度缩放回溯：`scale=1.0/0.75/0.5/0.25` 均越界，最终 `scale=0`；
6. 输出零速度，Mock Robot 停在 `z_min` 边界附近（`min_margin ≈ 0`），不再下沉。

这正是「VLA 想越界，SafeMotion 把它摁在安全区内」的完整因果链。
