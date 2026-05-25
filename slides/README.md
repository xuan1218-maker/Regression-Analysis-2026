# Slides / Notebook Workspace

这个目录不再把 `slides` 作为主要课堂媒介，而是把 **Jupyter Notebook** 作为课堂主界面。

核心思路：
- `*.md`：承载课程大纲、规范和 AI-friendly 说明；
- `*.py`（Jupytext percent format）：作为 notebook 的可维护源文件；
- `*.ipynb`：作为课堂运行和展示用 notebook；
- 图、表、代码输出：承担课堂里的“戏剧时刻”。

## 目录结构

```text
slides/
├── .venv/
├── pyproject.toml
├── uv.lock
├── README.md
├── ipynb_agent_spec.md
├── template/
├── week12/
└── week12_new/
    ├── figures/
    ├── week12_class.py
    └── week12_class.ipynb
```

## 环境准备

本目录使用 `uv` 管理 Python 环境。

如果仓库里已经有 `slides/.venv`，通常不需要重新初始化；否则可以在仓库根目录执行：

```bash
uv init --bare --vcs none --no-readme --no-workspace slides
uv venv slides/.venv
uv add --directory slides jupyter jupytext ipykernel nbformat numpy pandas matplotlib scikit-learn seaborn
```

## 激活环境

### macOS / Linux
```bash
source slides/.venv/bin/activate
```

### Windows PowerShell
```powershell
slides\.venv\Scripts\Activate.ps1
```

如果你不想手动激活，也可以直接使用 `uv run --directory slides ...`。

## 最常用命令

### 1. 启动 Jupyter Lab
在仓库根目录执行：

```bash
uv run --directory slides jupyter lab
```

### 2. 从 `.py` 同步到 `.ipynb`
推荐把 `.py` 当作 source of truth。

```bash
uv run --directory slides jupytext --sync week12_new/week12_class.py
```

### 3. 明确地把 `.py` 转成 `.ipynb`
```bash
uv run --directory slides jupytext --to ipynb week12_new/week12_class.py -o week12_new/week12_class.ipynb
```

### 4. 执行 notebook（验证从头到尾可运行）
```bash
uv run --directory slides jupyter nbconvert --to notebook --execute --inplace week12_new/week12_class.ipynb
```

### 5. 只打开某个 notebook 进行编辑
先启动 Jupyter：

```bash
uv run --directory slides jupyter lab
```

然后在浏览器里打开：
- `week12_new/week12_class.ipynb`

## VS Code 使用建议

如果你在 VS Code 中直接打开 notebook：

1. 打开 `slides/week12_new/week12_class.ipynb`；
2. 选择 Python 内核；
3. 推荐内核对应解释器：
   - `slides/.venv/bin/python`

这样能保证 VS Code 与 `uv` 的环境一致。

## 推荐工作流

### 教师 / 助教
1. 编辑 `week12_new/week12_class.py`
2. 同步生成 `week12_new/week12_class.ipynb`
3. 执行 notebook 检查是否能从头跑通
4. 上课时使用 `.ipynb`

对应命令：

```bash
uv run --directory slides jupytext --sync week12_new/week12_class.py
uv run --directory slides jupyter nbconvert --to notebook --execute --inplace week12_new/week12_class.ipynb
```

### 学生
1. 拉取仓库；
2. 启动 Jupyter；
3. 打开 notebook；
4. 从上到下运行；
5. 修改参数、观察图像和表格变化。

## 为什么同时保留 `.py` 和 `.ipynb`

保留两者的原因：

- `.py` 更适合 Git diff；
- `.py` 更适合 AI 生成和重构；
- `.ipynb` 更适合课堂展示和交互；
- 二者配合可以兼顾“可维护性”和“可运行性”。

## 规范文件

请同时参考：

- `slides/ipynb_agent_spec.md`

这个文件定义了：
- notebook 的 cell 类型；
- tags 规范；
- teacher-only script 的写法；
- agent 应如何生成和同步 notebook。

## 课堂 notebook 的基本理念

一个课堂 notebook 不应只是代码草稿，而应同时承担：
- 提问；
- 演示；
- 解释；
- 讨论；
- 留白；
- 教师隐藏剧本。

推荐教学节奏：

```text
cue -> stage -> explain -> checkpoint
```

## Week 12 示例

当前示例：
- `slides/week12_new/week12_class.py`
- `slides/week12_new/week12_class.ipynb`

它演示了：
- 模型复杂度变化下的 underfit / overfit；
- repeated sampling 展示 variance；
- outlier 对 `RMSE` / `MAE` 的影响。
