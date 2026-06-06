# Notebook Authoring Spec for Agents

## Status
- document_type: notebook_authoring_spec
- scope: `slides/` subtree
- target_files:
  - `slides/**/*.ipynb`
  - `slides/**/*.py` (Jupytext percent format)
- primary_audience: AI agents
- secondary_audience: human instructors
- source_of_truth: `.py` Jupytext file
- derived_artifact: `.ipynb`
- python_env: `slides/.venv`
- revealjs_capable: notebooks are authored to support `nbconvert --to revealjs` for slide-based classroom presentation

## Core Rule
在 `slides/` 下编写课程 notebook 时，**优先维护 Jupytext percent-format 的 `.py` 文件**，再由该 `.py` 文件同步生成 `.ipynb`。

Do:
- edit: source `.py` files (e.g., `topic_name.py`)
- generate/sync: derived `.ipynb` files (e.g., `topic_name.ipynb`)

Do not:
- 只改 `.ipynb` 而不回写 `.py`
- 把 `.ipynb` 当作唯一真实来源

## Human-in-the-Loop Workflow
This project uses a collaborative authoring flow between instructors and AI agents.

The default workflow is:

1. **Discuss scenes first**
   - human and agent first co-design the lesson outline;
   - decide how many scenes the class should have;
   - for each scene, decide the core question, the key visual/table/demo, the strict definition/formula that must appear, and the intended takeaway;
   - the unit of design is the **scene**, not the individual cell.
2. **Generate notebook draft**
   - the agent writes the Jupytext `.py` source;
   - the agent syncs it into `.ipynb`;
   - the notebook is treated as a draft for classroom inspection.
3. **Human reviews in notebook form**
   - the instructor opens the `.ipynb` locally, runs it, inspects layout, and makes adjustments;
   - edits may happen in notebook view during this stage for speed and teaching judgment.
4. **Sync back to source of truth**
   - after notebook-side edits, sync changes back into the `.py` file;
   - the `.py` file remains the long-term source of truth.
5. **Build presentation artifacts**
   - generate classroom HTML slides;
   - optionally generate a lightweight teacher-notes artifact in HTML or Markdown.

Operationally, this means:
- early-stage authoring is scene-first;
- notebook review is expected and supported;
- final build automation should stabilize the path from edited notebook back to `.py`, then to HTML outputs.

## Rationale
使用 `.py` 作为 source of truth 的原因：
- 更适合 Git diff；
- 更适合 AI 生成和重写；
- 更适合批量重构 notebook 结构；
- 更容易显式维护 cell tags 和教学结构。

## Required Tooling
Assume the `slides/` project uses:
- `uv`
- `jupyter`
- `jupytext`
- `ipykernel`
- `nbformat`
- `nbconvert` (for reveal.js slide export)

## Directory Convention
For each course topic or module, use a dedicated folder:

```text
slides/
├── pyproject.toml
├── .venv/
├── ipynb_agent_spec.md
└── topic_name/
    ├── figures/
    ├── notebook_name.py
    └── notebook_name.ipynb
```

## File Naming Convention
*Currently, there are no strict file naming restrictions.*

Please name your folders and files clearly according to the semantic topic or pedagogical purpose (e.g., `decision_trees/01_intro.py`, `linear_regression/main_class.py`, or `lesson01/lab.py`).

If there are multiple notebooks in the same topic, use explicit suffixes to distinguish their roles:
- `topic_main.py`
- `topic_demo.py`
- `topic_lab.py`

## Required Notebook Format
The `.py` source file should use Jupytext percent format.

Expected header pattern:

```python
# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---
```

## Teaching Model
Each notebook is a classroom artifact, not just a code scratchpad.

Notebook responsibilities:
1. carry the classroom flow;
2. trigger questions before explanations;
3. show phenomena through plots/tables/code execution;
4. preserve teacher-only script notes when needed;
5. keep text short and stage-like.

## Cell Taxonomy
Every major notebook should use these 5 core cell roles.
Each role maps to a reveal.js structural tag (see [Required Tags](#required-tags)).

### 1. `meta`
Purpose:
- notebook title
- week number
- learning goals
- usage note

Typical format:
- markdown cell, tagged `slide` (opening title slide)

### 2. `prompt`
Purpose:
- ask a question before running code (predict/create tension)
- or force a judgment/pause after an output (discuss/reflect)
- drives all student interaction

Typical format:
- markdown cell, tagged `slide` or `sub-slide`

### 3. `stage`
Purpose:
- produce the main classroom event
- code + figure/table/output

Typical format:
- code cell, tagged `slide` or `sub-slide`

### 4. `takeaway`
Purpose:
- give a short interpretation of the output (name the phenomenon)
- elevate from "what we just saw" to "what this means"
- transform a phenomenon into a transferable judgment
- give a clear sentence framed as “这里，大家需要记住的是 / 理解的是”

Typical format:
- markdown cell, tagged `fragment` (incremental reveal) or `sub-slide`

### 5. `script`
Purpose:
- teacher-only speaking notes
- reminders about pacing, likely confusion points, or follow-up prompts

Typical format:
- markdown cell, tagged `script` and `skip`
- excluded from reveal.js slides
- eligible for export into a lightweight teacher-notes artifact

## Required Tags
Use reveal.js-native structural tags consistently.

Tag set:
- `slide` — start a new horizontal slide
- `sub-slide` — start a vertical sub-slide (nested within the current slide)
- `fragment` — reveal content incrementally within the current slide
- `skip` — exclude cell from reveal.js output
- `script` — mark teacher-note cells for downstream notes export
- `hide-input` — hide the code input but keep the output visible in slides (use on all `# %%` code cells)

Typical role-to-tag mapping:

| Role | Typical Tag | Notes |
|---|---|---|
| `meta` | `slide` | Opening title slide |
| `prompt` | `slide` | A prompt typically starts a new slide |
| `stage` | `slide` or `sub-slide` | Demo content; use `sub-slide` when nested under a prompt |
| `takeaway` | `fragment` or `sub-slide` | Use `fragment` for point-by-point reveal within a slide |
| `script` | `script` + `skip` | Teacher-only notes, hidden from slides and available for notes export |

This mapping is conventional, not rigid. Use judgment to structure the slide deck effectively while preserving the `prompt → stage → formalize → takeaway` rhythm whenever the topic has a mathematical definition.

### Slideshow Metadata (Required for nbconvert)

Cell tags alone are NOT sufficient for `nbconvert --to slides`. Each tagged cell marker MUST also carry `slideshow={"slide_type": "..."}` metadata.

The Jupytext percent-format cell marker syntax:

```python
# %% [markdown] tags=["slide"] slideshow={"slide_type": "slide"}
# %% tags=["sub-slide"] slideshow={"slide_type": "subslide"}
# %% [markdown] tags=["skip"] slideshow={"slide_type": "skip"}
# %% [markdown] tags=["script","skip"] slideshow={"slide_type": "skip"}
```

Tag-to-slide_type mapping:

| Tag | `slideshow.slide_type` |
|---|---|
| `slide` | `slide` |
| `sub-slide` | `subslide` |
| `fragment` | `fragment` |
| `skip` | `skip` |

Note: the `slideshow` key uses `subslide` (not `sub-slide`) as the value.

## Structure Rule
The default teaching rhythm inside each scene should follow this pattern:

```text
prompt (slide) → stage (sub-slide) → formalize (sub-slide, if needed) → takeaway (fragment)
```

A typical scene should occupy **one horizontal `slide` container** with nested `sub-slide` and `fragment` reveals.
In other words:
- use one `slide` to announce/start the scene;
- place the scene's prompt, stage, formalization, and visible follow-up content underneath it as `sub-slide`s whenever possible;
- do not create an extra horizontal `slide` immediately after a scene title unless there is a deliberate chapter-level break.

Larger scenes may still span multiple horizontal `slide`s, but that should be the exception rather than the default; prefer `sub-slide` for vertical drill-down.

Across scenes, the notebook should be woven by standard markdown narrative text (no special tags needed) that explicitly:
- restates the core thread;
- connects the dots;
- prepares the semantic shift to the next section.

## Scene-First Authoring Rule
Agents should design at the **scene level before writing cells**.

For each scene, define first:
- the teaching question;
- the key visible event (usually one figure, one table, or one comparison);
- the strict mathematical object that must be named after the intuition is visible;
- the intended takeaway sentence;
- whether the scene fits in one horizontal `slide` with a small vertical stack of `sub-slide`s.

Only after that should the agent expand the scene into notebook cells.

Recommended translation:

```text
scene brief
  -> prompt cell(s)
  -> setup cell(s), if needed
  -> stage cell
  -> formalization cell(s), if needed
  -> takeaway cell
  -> script cell, if needed
```

Do not start from a loose pile of cells and hope a scene emerges later.

## Content Density Rule
- 每个 markdown cell 只承载一个主要功能；
- 每个 prompt cell 最多 1~3 个问题；
- 每个 takeaway cell 必须兼顾“解释现象”和“业务升华”，但尽量精简，并优先使用“这里，大家需要记住的是 / 理解的是”这一类直接面向学生的表述；
- 每个 stage cell 尽量只产生一个主要图或一个主要表；
- 如果概念有严格数学定义，必须在直观演示之后补一个简短的 formalization markdown cell，把图上的对象和公式里的对象一一对应；
- 如果代码太长，应拆分为“setup cell + stage cell”。

## Prompt Writing Rule
Prompts should not be "terminology questions" that only make sense after you already understand the concept.

A good prompt must provide enough situational context that a student can form a prediction or reflection.

Patterns to use:
- **Business scenario**: "假设你是一家超市的数据负责人，老板给你三个模型..."
- **Concrete analogy**: "同一个考试大纲，但每次给你不同的练习题，好学生应该每次考出接近的分数..."
- **Economic stakes**: "如果今天必须选一个上线，选错了会怎样？"
- **Visual prediction**: "不看后面的图，你觉得训练误差会怎么走？"

Avoid:
- Prompts that are answerable only by students who already know the concept ("> 谁像欠拟合？谁像过拟合？" — too flat)
- Prompts that are too broad or open-ended without a concrete anchor

## Takeaway Writing Rule
A `takeaway` cell must do two things at once:

1. **Name the phenomenon**: "This is why we call it high variance. The left panel is tighter because..."
2. **Elevate to judgment**: "So what does this mean for model selection? In any problem where training samples are limited..."

It should connect to risk, decision-making, or real-world stakes.
A scene that only names the phenomenon feels unfinished.
A scene that includes a transferable judgment makes students feel they learned something they can use.

Visible wording should be student-facing rather than teacher-facing.
Prefer phrases such as:
- “这里，大家需要记住的是……”
- “这里，大家需要理解的是……”

Avoid teacher-internal formulations such as:
- “学生要带走的是……”
- “这一幕要让学生明白……”

## Formalization Rule
The notebook should not stop at intuition when the topic has a standard mathematical definition.

After the main visual or empirical event, add a short visible markdown cell that:
- writes the core formula or definition;
- maps symbols back to the just-seen plot/table/demo;
- explains what part of the intuition the formula is formalizing.

Typical examples:
- after showing `p > n` instability, write the `OLS` estimator and note when `X^TX` is singular or ill-conditioned;
- after a PCA geometry plot, write the variance-maximization definition of the first principal component;
- after a PCR performance plot, write the “PCA scores + regression” pipeline in symbols.

The goal is not to turn the slide into a proof.
The goal is to explicitly connect “what we saw” and “what the formal definition says”.

## Narrative Arc Design Rule
A classroom notebook is not a flat list of demos.
It is a deliberately designed cognitive journey.

A 90-minute session should have roughly:
- 5–7 scenes;
- 1–2 synthesis sections;
- a clear prologue that sets stakes;
- a clear synthesis before transitioning to the next topic.

The high-level narrative arc should follow:

```text
prologue (stakes) → scenes (prompt → stage → formalize → takeaway) → synthesis (anchor) → scenes → synthesis (final) → next-topic transition
```

"More scenes" does not mean "more code."
It means more distinct cognitive beats—each scene should produce one main observation that feeds into one takeaway.

## Plotting Rule
For classroom plots:
- prefer one strong figure over many small ones;
- set titles and axis labels explicitly;
- annotate the interpretation if needed;
- avoid tiny unreadable defaults;
- prefer deterministic random seeds.

## Math Rendering Rule
When a visible markdown cell contains mathematical notation intended for slide rendering, use MathJax-compatible markdown syntax:
- inline math: `$...$`
- display math: `$$...$$`

Do not wrap real formulas in backticks such as `` `\hat\beta = ...` `` when the intent is mathematical rendering, because that produces literal code text in exported HTML slides.

## Data / Reproducibility Rule
- always set random seed in demo notebooks;
- keep datasets lightweight unless explicitly needed;
- do not require secret credentials;
- do not depend on network downloads during class unless explicitly planned.

## Output Rule
Default policy:
- commit the `.ipynb` file for convenience;
- do not rely on embedded outputs as the only source of truth;
- prefer notebooks that can be re-executed from top to bottom.

If not explicitly requested:
- avoid embedding excessive binary output;
- avoid extremely large notebook outputs.

## Code Cell Visibility Rule
All `# %%` code cells (non-markdown) that produce classroom output should be tagged `hide-input`.

This tells `nbconvert --to slides` to:
- **hide** the Python source code from the slide
- **keep** the cell's output (plots, tables, printed results) visible

The code remains visible in the `.ipynb` notebook view and in the `.py` source file.

Cell marker format:
```python
# %% tags=["hide-input"]
# code that produces a plot or table...
```

Pure setup/import cells that produce no visible output may use `skip` instead.

## Teacher-Only Script Rule
Teacher notes should:
- be short;
- focus on pacing or prompting;
- not duplicate the visible explanation cells;
- add the missing bridge between “图上看到了什么” and “为什么这个现象能推出后面的结论”;
- explicitly remind the instructor where to translate intuition into strict notation or definitions;
- be tagged with both `script` and `skip`.

Use the two hidden-note tags with different intent:
- `script`: teacher-facing notes that should be collected into a lightweight notes export;
- `skip`: hidden-from-slides content in general, including non-note utility cells or internal comments.

In other words:
- all `script` cells should also carry `skip`;
- not all `skip` cells are `script` cells.

Example content:
- "先让学生猜 20 秒，不要立刻运行。"
- "如果学生答不出 variance，就先说‘模型不稳定’。"
- "这里强调：训练误差下降不代表泛化能力提高。"
- "先用图讲‘最长方向’，再补一句：这正对应最大化投影方差的优化问题。"

## Preferred Lesson/Module Notebook Skeleton
Recommended high-level structure:

```text
00_meta                                                     [slide]
00_prologue (stakes, business scenario)                     [slide]
01_scene_1 (title/prompt → stage → formalize → takeaway)   [slide → sub-slide → sub-slide → fragment]
02_scene_2
03_scene_3
04_synthesis_1 (cognitive anchor)                           [slide]
05_scene_4
06_scene_5
07_scene_6
08_synthesis_2 (final synthesis)                            [slide]
09_transition (natural question → next topic)               [slide]
```

The exact number of scenes can vary by module content, but the arc should always include:
- a prologue;
- at least one synthesis section;
- a final synthesis that collects all takeaways;
- a transition that makes the next topic feel like a natural answer to an open question.

Default layout preference:
- one scene = one horizontal `slide`;
- scene internals = `sub-slide`s and `fragment`s;
- instructor pacing should usually be controlled manually, not by inserting an extra horizontal title slide before the real content.

## Commands
### Sync notebook edits back to source
From repository root:

```bash
uv run --directory slides jupytext --sync topic_dir/notebook_name.ipynb
```

### Create/update notebook from source
From repository root:

```bash
uv run --directory slides jupytext --sync topic_dir/notebook_name.py
```

### Convert source file into notebook explicitly
```bash
uv run --directory slides jupytext --to ipynb topic_dir/notebook_name.py -o topic_dir/notebook_name.ipynb
```

### Execute notebook for validation
```bash
uv run --directory slides jupyter nbconvert --to notebook --execute --inplace topic_dir/notebook_name.ipynb
```

### Export to reveal.js HTML slides
```bash
uv run --directory slides jupyter nbconvert --to slides topic_dir/notebook_name.ipynb --output-dir topic_dir/ --TagRemovePreprocessor.remove_input_tags="{'hide-input'}"
```

The `--TagRemovePreprocessor.remove_input_tags` option hides the code input of cells tagged `hide-input`, while keeping their output (plots, tables) visible in the slides.

### Preferred final build command
For the human-reviewed final build step, prefer the project build wrapper:

```bash
uv run --directory slides python -m slide_builder build topic_dir/notebook_name.ipynb
```

This wrapper is responsible for:
- syncing notebook edits back to `.py`;
- running structural checks;
- exporting reveal.js slides HTML;
- exporting lightweight teacher notes Markdown.

## Final Build Automation Rule
The final automation stage should be optimized for reliability, not for one-shot generation.

The recommended build sequence is:

1. sync the edited `.ipynb` back to `.py`;
2. run structural checks;
3. export reveal.js HTML slides;
4. export lightweight teacher notes from `script` cells.

The purpose of this stage is to stabilize the path:

```text
edited notebook -> synced .py -> validated notebook structure -> classroom slides.html + teacher_notes.md/html
```

### Structural checks
Before slide export, the build step should check at least:
- every slide-structured cell has valid `slideshow.slide_type` metadata;
- every classroom code-output cell that should hide source is tagged `hide-input`;
- teacher notes use `script` + `skip`, not just `skip`;
- scenes are not overstuffed with too many visible elements;
- obviously oversized markdown/table/figure sections are flagged for splitting into additional vertical slides.

These checks should be split into two severity levels:
- `warning`: probable density/layout problems, recommended scene splitting, likely projection issues;
- `error`: metadata missing, sync inconsistency, or export-blocking issues.

### Slide layout policy
For classroom projection:
- prefer more vertical slides over overstuffed single slides;
- do not rely on scrolling as the default presentation behavior;
- do not solve density problems primarily by shrinking fonts;
- if a slide is too tall, split the scene rather than compressing it aggressively.

### Teacher notes export
Teacher notes do not need to be a heavy product.

A simple Markdown or HTML artifact is sufficient, as long as it:
- extracts content from `script` cells;
- preserves notebook/scene order;
- records the instructor-facing bridge from phenomenon to conclusion, not just pacing reminders;
- remains easy for instructors to read before or during class.

## Agent Checklist
Before finishing notebook generation, an agent should verify:
1. Is the `.py` file present and readable as the source of truth?
2. Does the notebook use `slide`, `sub-slide`, `fragment`, `skip`, and `hide-input` tags to structure the slide deck?
3. Is the narrative structured around `prompt → stage → formalize → takeaway` when formal math is part of the topic?
4. Does every major scene include a `takeaway` cell that both names the phenomenon and elevates to judgment, using student-facing wording?
5. Are narrative syntheses placed at key transitions, not skipped?
6. Are prompts written with situational context, not just terminology?
7. Are teacher-only notes separated from student-visible text, with `script` distinct from generic `skip` content?
8. Does the notebook run top-to-bottom without external hidden state?
9. Has the `.ipynb` been synced from the `.py` source, or have notebook edits been synced back into `.py` before final export?
10. Are plots/tables legible enough for classroom projection?
11. Is each scene small enough to fit cleanly into one slide or a short vertical stack?
11. Is each scene, by default, implemented as one horizontal `slide` with a short vertical stack rather than as multiple consecutive horizontal title/content slides?
12. Does the notebook end with a transition question that makes the next topic feel like a natural continuation?
13. Does each mathematically defined concept connect its intuition to a visible formula/definition?
14. Do `script` cells help the instructor bridge from figure to interpretation to conclusion, rather than only repeating visible text?
