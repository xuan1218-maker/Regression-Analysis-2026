# Week 12 作业 Review：5 位同学的 error_curves / 候选模型问题与源码根因

> **阅读说明**：本文档基于各同学 `summary.md` 中的实际数字与 `main.py` 源码的逐行对照写成。
> 所有表格数据均来自同学的 `results/summary.md`（或 `report.md`），
> 所有源码引用均标注了精确的文件路径和行号。

---

## 目录

1. [12_wxy：error_curves 上 train RMSE 在高阶区锯齿跳跃（数值崩溃）](#1-12_wxyerror_curves-上-train-rmse-在高阶区锯齿跳跃数值崩溃)
2. [21_yyw：error_curves 上 test RMSE 长期低于 train RMSE（split 运气）](#2-21_yywerror_curves-上-test-rmse-长期低于-train-rmsesplit-运气)
3. [18_mxt：Task A 表里 degree=15 test 最低，报告却写 degree=4 最优（自相矛盾）](#3-18_mxttask-a-表里-degree15-test-最低报告却写-degree4-最优自相矛盾)
4. [07_nc：当前版本已修复数值崩溃，表现为标准过拟合](#4-07_nc当前版本已修复数值崩溃表现为标准过拟合)
5. [23_zy：error_curves 既有锯齿（train 不单调），test 也长期低于 train（双问题叠加）](#5-23_zyerror_curves-既有锯齿train-不单调test-也长期低于-train双问题叠加)
6. [总结对照表](#6-总结对照表)

---

## 1. 12_wxy：error_curves 上 train RMSE 在高阶区锯齿跳跃（数值崩溃）

### 1.1 图上实际表现

从 `students/12_wxy/src/week12/results/summary.md` 中 Task B 表格摘取后半段：

| degree | train RMSE | test RMSE |
|--------|-----------|----------|
| 9 | 0.255 | 0.328 |
| 10 | 0.255 | 0.330 |
| 11 | **0.253** | 0.322 |
| 12 | **0.622** ← 跳升 2.5 倍 | 0.716 |
| 13 | **1.055** ← 继续飙升 | 1.238 |
| 14 | **0.555** ← 又掉下来 | 0.705 |
| 15 | **0.318** ← 又掉 | 0.422 |
| 16 | **1.081** ← 又飙 | 1.575 |
| 17 | 0.822 | 0.999 |
| 18 | 1.138 | 1.470 |

**问题判断依据**：多项式回归有一个基本性质——模型是嵌套的（degree=d 的解空间是 degree=d+1 的子集），因此 **train RMSE 必须随 degree 单调不增**。但上表中 degree 11→12 时 train RMSE 从 0.253 跳到 0.622，degree 13→14→15→16 在 0.32 到 1.08 之间反复横跳。error_curves 图上 train 曲线是一条**锯齿状乱线**，而非平滑单调下降。

这不是统计意义上的"过拟合"（过拟合是 train 很低、test 反弹），而是**拟合过程本身在数值上崩溃了**。

### 1.2 源码根因

#### 根因 1：自写多项式特征构造，x 范围过大且无标准化

```python
# students/12_wxy/src/week12/main.py, L20-L25
def polynomial_features(X, degree):
    n_samples = X.shape[0]
    features = [np.ones(n_samples)]
    for d in range(1, degree + 1):
        features.append(X.reshape(-1) ** d)    # ← x ∈ [0,10]，x^15 ≈ 10^15
    return np.column_stack(features[1:])
```

`np.linspace(0, 10, 150)` 使 x 的上界为 10。degree=15 时，设计矩阵的最后一列 `x^15` 量级约 10^15，而第一列 `x^1` 仅约 10。各列量级横跨 15 个数量级 → **设计矩阵 `X^T X` 的条件数约 10^30**，这在 IEEE 754 双精度下接近奇异。

#### 根因 2：纯 OLS 求解，无正则化

```python
# students/12_wxy/src/week12/main.py, L48（run_candidate_models 内）
model = CustomOLS(fit_intercept=True, alpha=0.0)

# 同样的 CustomOLS(alpha=0.0) 也用于 run_error_curve() 内（L79）
model = CustomOLS(alpha=0.0)
```

`alpha=0.0` 即 Ridge 正则化强度为零 → 等价于普通最小二乘。当 `X^T X` 条件数爆炸时，OLS 的闭式解 `(X^T X)^(-1) X^T y` 中矩阵求逆在浮点精度下崩掉。不同 degree 下条件数的恶劣程度不同（取决于各列的具体数值），因此某些 degree（12, 13, 16）崩得厉害，另一些（14, 15）恰好"幸存"，形成了锯齿。

#### 根因 3：两个 Task 各自独立生成数据

```python
# students/12_wxy/src/week12/main.py, L40-L42（run_candidate_models）
x, y, y_true = generate_data()
x_train, x_test, y_train, y_test = train_test_split(x, y)

# L73-L75（run_error_curve）
x, y, _ = generate_data()
x_train, x_test, y_train, y_test = train_test_split(x, y)
```

`run_candidate_models()` 和 `run_error_curve()` 各自独立调用 `generate_data()` → 两个 Task 的 train/test split 不同 → error_curves 图上的锯齿模式不可通过 candidate_models 图来验证。

### 1.3 修复方向

1. 将 x 范围缩小（如 `np.random.uniform(-3, 3, 150)`），或使用 `StandardScaler` 标准化多项式特征；
2. 将 `alpha=0.0` 改为一个小正数（如 `alpha=1e-4`），引入 Ridge 正则化来镇定矩阵求逆。

---

## 2. 21_yyw：error_curves 上 test RMSE 长期低于 train RMSE（split 运气）

### 2.1 图上实际表现

从 `students/21_yyw/src/week12/rrresults/summary.md` 中 Task B 表格摘取前半段：

| degree | train RMSE | test RMSE | gap (= test − train) |
|--------|-----------|----------|---------------------|
| 1 | 0.8860 | **0.8631** | **−0.02** |
| 2 | 0.8856 | **0.8627** | **−0.02** |
| 3 | 0.5903 | **0.5008** | **−0.09** |
| 4 | 0.5902 | **0.4984** | **−0.09** |
| 5 | 0.5228 | **0.4350** | **−0.09** |
| 6 | 0.5218 | **0.4240** | **−0.10** |
| 7 | 0.5198 | **0.4220** | **−0.10** |
| 8 | 0.5191 | **0.4347** | **−0.08** |
| 9 | 0.5191 | **0.4312** | **−0.09** |
| 10 | 0.5179 | **0.4456** | **−0.07** |
| 11 | 0.5170 | **0.4807** | **−0.04** |
| 12 | 0.5170 | **0.4742** | **−0.04** |
| 13 | 0.5113 | 0.6050 | **+0.09** ← gap 终于翻正 |

degree=1 到 degree=12，**gap 全部为负**（test RMSE < train RMSE）。这意味着 error_curves 图上 test 曲线在 train 曲线的**下方**，和教科书上"train 在下、test 在上，test 先降后升"的标准 U 形**完全相反**。

直到 degree=13，gap 才转正（0.09），随后 degree=15（+0.20）、degree=18（+1.10），高阶区又出现断崖式跳跃（`include_bias=False` 且无 `StandardScaler` 导致的数值不稳定，但与本节 issue 无关，此处不展开）。

### 2.2 源码根因

#### 根因：单次 `seed=42` 的随机 split 恰好让测试集比训练集"好拟合"

```python
# students/21_yyw/src/week12/main.py, L28-L31
RANDOM_STATE = 42
N_SAMPLES = 200
NOISE_STD = 0.5
TEST_RATIO = 0.3                # → 60 test, 140 train

# L69-L79
def generate_data(...):
    rng = np.random.default_rng(seed)       # ← 局部随机生成器，seed 固定为 42
    x = rng.uniform(-3, 3, n_samples)
    y_noisy = y_true + rng.normal(0, noise_std, n_samples)

    n_test = int(n_samples * test_ratio)    # = 60
    indices = rng.permutation(n_samples)    # ← seed=42 下的排列
    test_idx = indices[:n_test]             # 前 60 个 → 测试集
    train_idx = indices[n_test:]            # 后 140 个 → 训练集
```

**关键点**：`generate_data()` 只在 `main()` 开头调用一次，Task A 和 Task B 共用同一份 split：

```python
# students/21_yyw/src/week12/main.py, L465-L469 (main 函数内)
x_train, y_train, x_test, y_test = generate_data()

# Task A
task_a = run_model_complexity_demo(x_train, y_train, x_test, y_test)

# Task B —— 同一份 split！
task_b = run_error_curves(x_train, y_train, x_test, y_test)
```

`seed=42` + `permutation(200)` 产生的这次划分中，恰好 60 个测试点的分布或噪声水平使它们比 140 个训练点更容易预测。这不是代码 bug，而是**单次随机 split 的偶然结果**——如果你换一个 seed（比如 `RANDOM_STATE = 123`），test 曲线大概率会回到 train 上方。

### 2.3 为什么这很重要

Week 12 的核心教学目标是让学生在图上看清"过拟合时 test 误差翘起来"的 U 形。但 21 的 error_curves 上 test 长期低于 train，图上的叙事变成了"越复杂 test 越好（直到 degree=13）"——这和你要讲的故事是反的。一旦你把这图投到屏幕上，学生看到的是"test 比 train 还低，那过拟合在哪？"

### 2.4 修复方向

最简单的修复：把 `RANDOM_STATE = 42` 改成另一个值（如 `123`、`2026`），重新跑。或者更稳健的做法：多做几次 random split 取平均，而不是依赖单次 split。

---

## 3. 18_mxt：Task A 表里 degree=15 test 最低，报告却写 degree=4 最优（自相矛盾）

### 3.1 图上实际表现

从 `students/18_mxt/src/week12/results/summary.md` 中 Task A 表格：

| degree | train RMSE | test RMSE |
|--------|-----------|----------|
| 1 | 0.728 | 0.688 |
| 4 | 0.329 | 0.347 |
| **15** | **0.359** | **0.305** ← 三个候选里 test 最低 |

但报告文字写：
> "Degree 15：过拟合……过度学习训练集噪声，测试误差明显上升。"
> "Degree 4：最优平衡，是上线部署的首选。"

**表格里 degree=15 的 test RMSE（0.305）比 degree=4（0.347）低 12%**，文字却说它的"测试误差明显上升"——这不仅自相矛盾，而且直接与数字相反。读者无法从报告里找到任何解释：为什么表里最优的不选？

### 3.2 源码根因

#### 根因 1（次要）：每个 Task 独立生成数据，跨 Task 无法互相验证

```python
# students/18_mxt/src/week12/main.py, L50-L60
def generate_data(n_samples=200):
    x = np.linspace(0, 10, n_samples)
    y_true = np.sin(x) + 0.5 * x
    y = y_true + np.random.normal(0, 0.3, n_samples)
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.3, random_state=42
    )
    return x, y_true, x_train, x_test, y_train, y_test

# L65-L69：run_candidate_models 调用 generate_data() → split A
def run_candidate_models():
    x, y_true, x_train, x_test, y_train, y_test = generate_data()

# L140-L144：run_complexity_sweep 又调用 generate_data() → split B ≠ split A
def run_complexity_sweep():
    x, y_true, x_train, x_test, y_train, y_test = generate_data()
```

虽然 `random_state=42` 固定了 `train_test_split`，但 `np.random.normal` 在两次 `generate_data()` 调用之间使用全局 `np.random` 状态，状态已被第一次调用推进 → 第二次调用的噪声不同 → split 也不同。

#### 根因 2（主要）：报告写作时"先写结论，再填数字"，未做跨图推理

18 的 main.py 末尾写得很清楚：

```python
# students/18_mxt/src/week12/main.py, L345-L347
print("⚠️  This script only runs experiments and generates plots/data")
print("⚠️  Please write your report manually to: results/summary.md")
```

脚本只生成图和 CSV，没有自动写报告。summary.md 是手工写的。同学心里预设了"degree=15 = 过拟合 → 不能选"的结论，在写报告时直接套用了这个判断，**没有检查 Task A 表格中 degree=15 的实际 test RMSE 是否支持这个结论**。

正确的做法应该是在报告里写：
> "本次 split 下 degree=15 的 test RMSE 最低（0.305），但从 variance_demo 和完整 error_curves 来看，高度数模型在不同训练样本下非常不稳定。如果只押一个上线，我更倾向 degree=4，牺牲一点本次 test 表现换取更稳定的泛化。"

但 18 跳过了这一步推理。

### 3.3 修复方向

1. **工程上**：所有 Task 共用同一份 `generate_data()` 的输出（像 21 那样在 `main()` 里只调一次），避免跨 Task 数据不一致。
2. **写作上**：写完报告后逐行核对：表中的数字是否支持我写的结论？如果有矛盾，要么修改结论，要么给出跨图证据来解释为什么"表优 ≠ 选它"。

---

## 4. 07_nc：当前版本已修复数值崩溃，表现为标准过拟合
### 4.1 图上实际表现

> **注：** 初版 Review 中记录该同学 Degree 15 出现了 `train RMSE` 高达 3.18 的拟合崩溃。但在当前仓库最新版本中，该问题已被修复。

从当前 `students/07_nc/week12/results/report.md` 的 Task A 数据看：

| degree | train RMSE | test RMSE |
|--------|-----------|----------|
| 1 | 0.7071 | 0.8136 |
| **4** | **0.3168** | **0.4750** |
| 15 | **0.2112** | 0.6665 |

**最新表现**：Degree 15 的 `train RMSE`（0.2112）极低，甚至远低于 Degree 4 的 0.3168，而其 `test RMSE` 达到了 0.6665，出现反弹。
这完全符合**过拟合的标准特征**（train 低 + test 高），证明模型在极力记忆训练集的局部噪声。

### 4.2 源码状态

该同学最新代码中，核心逻辑如下：

```python
# students/07_nc/week12/main.py, L106
x_train, x_test, y_train, y_test = train_test_split(
    x, y,
    train_size=42,       # ← 依然只用了 42 个训练点
    random_state=seed,
    shuffle=True,
)

# L135-L139
def make_polynomial_model(degree: int) -> Pipeline:
    return Pipeline(
        steps=[
            ("poly", PolynomialFeatures(degree=degree, include_bias=False)),
            ("scale", StandardScaler()),      # ← 核心：特征标准化
            ("linear", LinearRegression()),
        ]
    )
```

尽管训练样本依然极少（42 个），但在 `StandardScaler` 与 `LinearRegression` 的配合下，当前版本成功避免了多项式矩阵求逆过程中的数值溢出与崩溃。极少的样本加上高阶多项式（15阶），使得模型能够轻而易举地穿过这些点，在视觉上和数值上呈现出典型的**高方差与过拟合**，报告中的诊断如今是完全准确的。

---

## 5. 23_zy：error_curves 既有锯齿（train 不单调），test 也长期低于 train（双问题叠加）

### 5.1 图上实际表现

从 `students/23_zy/src/week12/results/summary.md` 中 Task B 表格摘取后半段：

| degree | train RMSE | test RMSE | gap |
|--------|-----------|----------|-----|
| 9 | **0.3448** | 0.3357 | −0.009 |
| 10 | **0.3444** | 0.3366 | −0.008 |
| 11 | **0.3443** | 0.3365 | −0.008 |
| 12 | **0.3436** | 0.3375 | −0.006 |
| 13 | **0.3430** | 0.3367 | −0.006 |
| 14 | **0.3826** ← train 突然跳升 12% | 0.3834 | **+0.0007** |
| 15 | **0.4013** ← 继续上升 | 0.4016 | +0.0003 |
| 16 | **0.4131** ← 还在上升 | 0.4002 | −0.013 |
| 17 | 0.4119 | 0.4014 | −0.010 |
| 18 | 0.4115 | 0.4075 | −0.004 |

**23 的 error_curves 同时出现了两种不该出现的现象：**

**现象 A（锯齿 / 数值崩溃）**：degree 13→14，train RMSE 从 0.3430 **跳升到 0.3826**，再升至 0.4013。嵌套模型下 train RMSE 必须单调递减——这里反升了，说明数值求解崩了（和 12_wxy 同根因）。

**现象 B（test 低于 train / split 运气）**：degree 1–13 以及 16–18，gap 全部为负或接近零。标准 U 形没有出现——test 曲线长期在 train 下方（和 21_yyw 同根因）。

### 5.2 源码根因

#### 根因 A（锯齿）的来源：x ∈ [0, 10] + 无 StandardScaler

```python
# students/23_zy/src/week12/main.py, L99-L106
def true_function(x):
    return np.sin(x) + 0.25 * x

# L109-L119
def generate_data(n_samples=160, noise_std=0.35, random_state=42):
    rng = np.random.default_rng(random_state)
    x = rng.uniform(0, 10, size=n_samples)     # ← x ∈ [0, 10]
    x = np.sort(x)
    ...
    x_train, x_test, y_train, y_test = train_test_split(
        x.reshape(-1, 1), y,
        test_size=0.3, random_state=random_state,
    )

# L142-L145
def build_polynomial_model(degree):
    model = Pipeline(steps=[
        ("poly", PolynomialFeatures(degree=degree, include_bias=False)),
        ("linear", LinearRegression()),         # ← 无 StandardScaler
    ])
```

与 12_wxy 完全相同的模式：`x ∈ [0, 10]` + `include_bias=False` + 无 `StandardScaler` → degree=18 时 `x^18 ≈ 10^18`，各列量级横跨 18 个数量级 → 设计矩阵条件数爆炸 → LinearRegression 的 SVD/QR 求解在高 degree 区间数值崩溃 → train RMSE 不降反升。

#### 根因 B（test 低于 train）的来源：split 运气

23 使用 `random_state=42` + `train_test_split(test_size=0.3)`。和 21_yyw 类似，这次 split 产出的 48 个测试点恰好比 112 个训练点更容易拟合。这不是代码 bug，而是单次 split 的偶然结果。

### 5.3 为什么 23 的情况最"复杂"

23 的 error_curves 是五种问题中最复杂的：它既有 12 的数值崩溃（锯齿），又有 21 的 split 运气问题（test 低于 train）。两个问题叠加在同一张图上 → 图上的 train 和 test 曲线行为都不符合教科书预期 → 无论从哪个角度解读，都无法讲出"bias-variance tradeoff"的标准故事。

### 5.4 修复方向

1. 加 `StandardScaler` 到 pipeline 中，或缩小 x 范围（如 `rng.uniform(-3, 3, ...)`) → 解决锯齿；
2. 换一个 `random_state`（如 `2026`），或做多次 split 取平均 → 解决 test 低于 train 的问题；
3. 两项同时修 → error_curves 应恢复为标准 U 形。

---

## 6. 总结对照表

| 同学 | error_curves 图上表现 | 源码根因 | 本质归类 | 一句话 |
|------|----------------------|----------|----------|--------|
| **12_wxy** | train RMSE 在 degree 12–18 锯齿跳跃（0.25→0.62→1.06→0.32→1.08），train 不单调 | `CustomOLS(alpha=0)` + `linspace(0,10)` + degree 到 18 → `x^15≈10^15` → 矩阵条件数爆炸 | **数值崩溃** | OLS 在病态矩阵上算不准了 |
| **21_yyw** | test RMSE 在 degree 1–12 全程低于 train RMSE（gap 为负），图上下颠倒 | `seed=42` 的 split 恰好让 60 个 test 点比 140 个 train 点更容易拟合 | **split 运气** | 测试集碰巧比训练集"简单" |
| **18_mxt** | Task A 表里 degree=15 test 最低（0.305），报告文字却写 degree=4 最优 | 心里预设"高度数=过拟合" + 没检查表中实际数字 + 未做跨图推理 | **报告自相矛盾** | 先写结论再填表，没回头校对 |
| **07_nc** | degree=15 train=0.21 远低于 degree=4（0.32），但 test 高达 0.67 | 当前版本 `StandardScaler` 和 `LinearRegression` 的配合成功避免了崩溃 | **标准过拟合** | 典型的过拟合现象，此前的崩溃问题已在当前版本中修复 |
| **23_zy** | train 在 degree 14–16 不降反升（锯齿）+ test 全程低于 train（上下颠倒） | x∈[0,10] + 无 StandardScaler → 数值崩溃；seed=42 split 使 test 偏简单 | **锯齿 + 上下颠倒，双问题叠加** | 12 的问题和 21 的问题撞在同一张图上 |

---

## 附录：各同学关键参数速查

| 参数 | 12_wxy | 21_yyw | 18_mxt | 07_nc | 23_zy |
|------|--------|--------|--------|-------|-------|
| 总样本 | 150 | 200 | 200 | 140 | 160 |
| 训练集大小 | ~105 | 140 | ~140 | **42** | ~112 |
| x 生成方式 | **linspace(0,10)** | uniform(-3,3) | **linspace(0,10)** | uniform(-3,3) | uniform(0,10) |
| 真实函数 | sin(x) | sin(1.5x)+0.3x | sin(x)+0.5x | sin(1.4x)+0.35x+**cos(2.2x)** | sin(x)+0.25x |
| 噪声 σ | 0.3 | 0.5 | 0.3 | 0.35 | 0.35 |
| StandardScaler | 无 | 无 | 无 | **有** | 无 |
| include_bias | —（自写 OLS） | False | 默认 True | False | False |
| 主要问题 | 数值崩溃 | split 运气 | 报告矛盾 | 表现正常(已修复) | 双问题叠加 |

---

## 附录 2：导致问题的核心源码位置标注

为便于直接排查和审阅，以下是导致上述现象的核心代码文件与对应行号：

1. **12_wxy (数值崩溃)**
   - `students/12_wxy/src/week12/main.py`, L20-L25：`polynomial_features` 函数内，`X` 直接进行高次方运算（`X ** d`）而未进行标准化。
   - `students/12_wxy/src/week12/main.py`, L48, L79：使用 `CustomOLS(alpha=0.0)` 直接进行 OLS 求解，没有利用正则化项保护病态矩阵求逆。
   - `students/12_wxy/src/week12/main.py`, L40-42, L73-75：`generate_data()` 在多个实验函数中独立调用，导致 Task A 与 B 并非基于同一份切分数据进行比较。

2. **21_yyw (split 运气导致 test 全程偏低)**
   - `students/21_yyw/src/week12/main.py`, L28-L31：设置固定随机数种子 `RANDOM_STATE = 42`，单次切分产生了一个使得测试集非常“好拟合”的数据分布。
   - `students/21_yyw/src/week12/main.py`, L465-469：所有任务仅依靠这一组包含运气成分的 `x_train, y_train, x_test, y_test` 进行评价，未能通过交叉验证等方式消解误差。

3. **18_mxt (报告表述与数据自相矛盾)**
   - `students/18_mxt/src/week12/main.py`, L65-69, L140-144：在不同的分析函数（`run_candidate_models` 与 `run_complexity_sweep`）中多次独立调用 `generate_data()`。
   - 导致该问题的关键是：受全局 `np.random` 状态影响，两次调用的噪声不同，切分数据不一致。这种数据状态下，写作者强行套用定式结论（高次等于过拟合），而忽略了 Task A 表格里自己实际跑出的最优 `test RMSE`。

4. **07_nc (曾引发拟合崩溃，现已表现为过拟合)**
   - `students/07_nc/week12/main.py`, L106-L111：`train_test_split` 中显式指定极小的 `train_size=42`。
   - `students/07_nc/week12/main.py`, L135-L139：`Pipeline` 的特征构造环节。在未添加 `StandardScaler` 的极早期版本中曾因样本极少且特征跨度极大引发崩溃；当前版本加入 `StandardScaler` 配合 `LinearRegression` 后已被完美修复，展现正常的过拟合现象。

5. **23_zy (数值崩溃与 split 运气双叠加)**
   - `students/23_zy/src/week12/main.py`, L109-119：使用 `rng.uniform(0, 10)` 在极大区间上采样且最终并未标准化。
   - `students/23_zy/src/week12/main.py`, L114-L117：`train_test_split` 中 `random_state=42` 导致测试集的运气效应（与 21_yyw 类似）。
   - `students/23_zy/src/week12/main.py`, L142-145：`PolynomialFeatures(degree=degree, include_bias=False)` 后直接衔接 `LinearRegression()` 导致 14阶以上矩阵求逆出现数值跳跃。

