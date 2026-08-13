# SafeMotion — VLA 机械臂安全执行器

> Take-home Assignment｜VLA 控制 UR5 在 CR2032 纽扣电池装配工作站的安全执行原型。
>
> 纯软件安全层：拦截 VLA 输出的 50 点关节轨迹，经输入检查、名义速度生成、
> 全身安全检查/过滤后交给确定性 Mock Robot 闭环执行，保证机械臂**全身**
> （TCP + 六关节 + 连杆中间点）始终处于允许工作空间内。

```
50-point VLA trajectory
        ↓ 输入检查（非法整体拒绝）
        ↓ 逐点生成名义关节速度（限速）
        ↓ 全身安全过滤（速度缩放+二分 / QP，fail-closed）
        ↓ step 前独立复核（信任边界）
        ↓ Mock Robot（q += q_dot·dt）
        ↓ 执行后全身安全审计
```

## 快速开始

```bash
pip install -r requirements.txt

pytest -q                                               # 72 项测试

python -m safe_motion.replay scenarios/free_space.json
python -m safe_motion.replay scenarios/workspace_boundary.json
python -m safe_motion.replay scenarios/workspace_boundary.json --explain 28
python -m safe_motion.replay scenarios/workspace_boundary.json --method qp
```

所有命令在仓库根目录运行。Replay 默认把轨迹数据 / 统计 / 图写入 `artifacts/`。

## 结果一览

| 场景 | 语义 | 未过滤 min margin | 执行后 min margin | 干预 |
|---|---|---:|---:|---|
| `free_space` | 自建安全轨迹 | +0.0392 m | **+0.0392 m** | 0 改 0 停（未修改） |
| `workspace_boundary`（real_01） | TCP 破 z_min | −0.0412 m | **+0.0000 m** | 首次干预 step 28 |
| `real_02_upper_z_boundary` | TCP 破 z_max | −0.0248 m | **+0.0000 m** | 首次干预 step 47 |
| `tcp_safe_mid_link_unsafe`（real_03） | TCP 在界内、joint_3 破 x_min | −0.0610 m | **+0.0000 m** | 首次干预 step 46 |
| `joint_limit` | J6 平滑越过 +2π | — | — | **输入检查拒绝**，不进 Mock Robot |
| `invalid_input_nan` | 轨迹中部 NaN | — | — | **输入检查拒绝**，不进 Mock Robot |

**与面试方参考实现交叉验证**：三个 real 场景的首次干预步（28 / 47 / 46）与参考
`first_unsafe_frame` 完全一致；未过滤 min margin 与参考 worst margin 数值一致
（误差 < 1e-6）；FK 的 TCP / 中间关节坐标与参考一致（测试锁定）。

**过滤方法对比（real_01）**：

| | modified | stopped | max TCP 偏离 | 行为 |
|---|---:|---:|---:|---|
| 速度缩放（默认） | 1 | 21 | 0.186 m | 贴边即停，hold 位置 |
| QP（`--method qp`，加分项） | 22 | 0 | 0.125 m | 保持 5 mm 余量，沿边界滑行继续逼近目标 |

## 目录结构

```
safe-motion-kimi/
├── README.md
├── requirements.txt / pyproject.toml
├── safe_motion/
│   ├── config.py          # DH 参数、关节限位、默认工作空间、RobotConfig
│   ├── kinematics.py      # UR5 标称正运动学（标准 DH）
│   ├── geometry.py        # Workspace、连杆采样、全身检查、子步运动检查
│   ├── validation.py      # 输入检查（非法拒绝）
│   ├── safety_filter.py   # 名义速度 + 速度缩放/二分 + fail-closed
│   ├── qp_filter.py       # 加分项：QP 最小修改过滤（SLSQP + 精确验证 + 回退）
│   ├── mock_robot.py      # 确定性 Mock Robot（题目统一定义）
│   ├── replay.py          # 闭环 Replay + CLI（--explain / --method）
│   └── visualize.py       # 3D 工作空间图 + margin 曲线图
├── scenarios/             # 3 个面试方场景 + free_space/joint_limit/invalid_input_nan + 命名拷贝
├── scripts/make_scenarios.py
├── tests/                 # 72 项 pytest
└── artifacts/             # Replay 输出（npz / summary.json / 图）
```

## README 必答 9 问

**1. `action_chunk` 的语义是什么？**
`(50, 6)`，每行是未来某时刻六轴**绝对关节目标位置**（rad），不是增量也不是速度。
20 Hz 输出，相邻点 0.05 s，50 点 ≈ 2.5 s 未来轨迹。

**2. 50 点轨迹是如何逐步执行的？**
闭环 Replay：逐点取目标，基于 Mock Robot **当前实际状态**生成名义速度、过滤、
`step`，下一控制点从上一步的实际状态继续——不开环、不假设前一目标已到达。

**3. UR5 正运动学如何实现？**
标准 DH（Craig）：`A_i = RotZ(θ)·TransZ(d)·TransX(a)·RotX(α)`，逐关节累积，
返回 `base / joint_1..joint_6 / tcp`（tcp == O_6，无工具偏置）。正确性由三类测试
锁定：q=0 手算参考坐标、连杆长度不变量（200 随机位形）、面试方场景文件中
`reference_analysis` 的 TCP/中间关节坐标交叉验证（1e-6 一致）。

**4. 如何检查整条机械臂，而不仅是 TCP？**
每个状态检查：6 个关节点 + TCP + 各连杆 2 cm 等距采样点。
- **为什么 2 cm 足够**：矩形 keep-in 区域是**凸集**，线段两端点在内部 ⟹ 整段在
  内部——对凸盒本检查在数学上是精确的，采样不产生漏检；
- **剩余误差**：采样仅作浮点边界兜底；若换成非凸/keep-out 区域，纯采样可能漏检
  尺度小于采样间距的浅越界，应改解析线段-AABB 相交测试或加密采样。
- **基座处理**：O_0（z=0，在 z_min 之下）是固定安装段，不参与检查。证据：面试方
  参考的 `source_minimum_workspace_margin = 0.039159 = d1 − z_min`，恰为 O_1 的
  margin——参考实现同样不检查 O_0。
- real_03 即「TCP 在界内、joint_3 越 x_min」，测试断言系统识别且最差点为中间关节。

**5. SafeMotion 如何修改不安全动作？**
默认**速度缩放/回溯**（方法 A）：按 `{1.0, 0.75, 0.5, 0.25, 0}` 尝试，对每档预测
`q_next = q + scale·v·dt` 并做**子步插值全身检查**（默认 4 子步，覆盖"一步内穿过
边界"）；取首个安全档，再与失败的上一档之间**二分细化 10 轮**，逼近真正的最小修改
（例：step 28 粗档 0.75 → 细化后 0.7905）。加分项 **QP**（方法 B）：以临近边界节点
的线性化 margin 为约束求 `min||v−v_nom||²`（SLSQP），可沿边界滑行；线性化只是
**提案**，必须过非线性精确验证才下发，失败回退速度缩放。

**6. 如果安全过滤失败，会发生什么？**
输出零速度，受控停止。三层防线：① 过滤器内部任何异常/非法输出 →
`filter_fail_closed` 兜成零速度；② Replay 在 `robot.step` 前对过滤器输出做
**独立复核**，被污染/说谎的过滤器输出会被否决；③ `MockRobot.step` 自身拒绝
非法命令。对应测试：注入崩溃的过滤器 → 50 步全停、机器人原地不动；注入"放行
原动作"的说谎过滤器 → 复核拦截，执行轨迹仍全程安全。

**7. Mock Robot 做了哪些简化？**
`q[t+1] = q[t] + q_dot·dt` 一阶积分；perfect tracking、固定步长、零延迟零噪声；
不建模加速度/力矩/惯量/摩擦/底层控制器动力学；仅关节限位硬约束。

**8. 当前实现能够保证什么？**
- 非法输入（NaN/Inf/错误 shape/非 50 点/突跳/越界目标/非法 hz）不进入 Mock Robot；
- Mock Robot 永不突破关节限位（过滤器预检先于 robot 硬约束）；
- 执行轨迹每一帧**全身**位于 workspace 内（margin ≥ 0，含子步路径）；
- 过滤失败必停（fail-closed），绝不回退原始动作；
- free-space 轨迹零修改（执行 == 名义，偏差 0）。

**9. 当前实现不能保证什么？**
只在「UR5 标称 DH + 矩形 keep-in + 确定性 Mock Robot + 20 Hz/50 点」边界内有效，
**不能**证明真机安全。真实部署还需：出厂标定误差、工具几何、状态估计误差、
通信延迟、跟踪误差、制动距离、电机动力学、感知漏检、急停/Protective Stop、
功能安全认证（ISO 10218 / ISO/TS 15066）。

## 关键设计决策

1. **关节目标越界 → 拒绝（而非钳位）**。题目第 3 节把「关节目标不得超过关节位置
   范围」列为输入检查项，且「非法输入不得进入 Mock Robot」。钳位会让 VLA 的非法
   目标静默变成另一个动作；拒绝把问题显式暴露给上游策略层。两种做法都满足 10.4
   「Mock Robot 不得进入非法关节状态」，本实现选择fail-fast 语义并在此声明。
2. **突跳检查拦不住缓慢漂移的危险**。三个 real 场景的越界偏移是渐进叠加的
   （相邻步 ≤ 0.039 rad < 阈值 0.5 rad），输入检查全部放行——只有基于模型的全身
   预测能拦住。这正是安全过滤存在的理由。
3. **速度限速是逐关节饱和**（`clip`），不改变各关节运动方向符号，但会改变合成
   方向；因随后有全身预测兜底，安全性不受影响。
4. **QP 的线性化误差用 5 mm 缓冲吸收**（`qp_safety_margin`）：0.15 rad 单步的
   一阶泰勒误差可达亚毫米级，缓冲后精确验证（阈值 ~0）稳定通过，同时天然实现
   workspace safety margin 加分项。

## 测试

72 项 pytest，验证**最终执行轨迹的安全性**（不变量），而非「函数能跑」：
FK 交叉验证 / 凸性 / 采样间距 / 15 类非法输入 / 回溯与二分 / 限位预检 /
fail-closed（注入异常、注入说谎过滤器）/ 5 场景端到端 / 12 组随机轨迹
（温和+激进混合）全程全身安全。

## 已知边界

- 速度缩放在「目标持续越界」时会贴边 hold（QP 模式可滑行）；
- 采样间距 / 子步数 / 阈值均为配置项，可按工作站标定收紧；
- 隐藏 50 点轨迹可直接处理：无任何针对场景坐标的硬编码。
