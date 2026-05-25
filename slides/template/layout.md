# Beamer Slide Layout Spec

## Status
- document_type: slide_layout_spec
- target_renderer: LaTeX Beamer
- template_entrypoint: `slides/template/course_beamer_template.tex`
- theme: `Madrid` (built-in beamer theme)
- theme_file: none_required_by_default
- compile_engine: `xelatex`
- audience_priority: AI_first_human_readable_second
- intended_usage: future lecture slide generation

## Goal
本文件定义未来生成课程课件时应遵循的版式、结构和内容密度规则。

它的用途不是讲授知识点，而是告诉 AI 或人类作者：
- 一套标准课件应包含哪些页面类型；
- 每类页面应放什么内容；
- 内容密度和表达粒度如何控制；
- 生成 LaTeX Beamer 课件时优先遵循什么规范。

## File Convention
每次新课件建议基于模板复制一份新文件，例如：
- `slides/week12_bias_variance.tex`
- `slides/week13_regularization.tex`
- `slides/week14_pca_pcr.tex`
- `slides/week15_logistic_regression.tex`
- `slides/week16_model_selection.tex`

## Recommended Slide Sequence
一份标准课程课件建议遵循以下顺序：

1. `title_page`
2. `agenda`
3. `learning_objectives`
4. `motivation_or_problem_setup`
5. `core_concept_1`
6. `core_formula_or_geometry`
7. `comparison_or_taxonomy`
8. `figure_or_visual_intuition`
9. `case_study_or_workflow`
10. `code_or_pseudocode_optional`
11. `discussion_question`
12. `summary`
13. `assignment_or_after_class`
14. `qa`

## Required Frame Types
以下页面类型建议至少覆盖大部分：

### 1. `title_page`
Purpose:
- 给出课程名、周次、主题、授课人、日期。

Content rules:
- 标题不要过长；
- 副标题用于补充英文主题或学期信息；
- 不要在标题页堆叠过多说明文字。

### 2. `agenda`
Purpose:
- 告诉学生本讲结构。

Content rules:
- 3~6 个 section 为宜；
- section 名称尽量是概念词组，而不是长句。

### 3. `learning_objectives`
Purpose:
- 明确学生学完后应该会什么。

Content rules:
- 2~4 条即可；
- 每条目标最好以“理解 / 解释 / 比较 / 应用 / 判断”开头。

### 4. `motivation_or_problem_setup`
Purpose:
- 建立“为什么要学这个”的问题意识。

Content rules:
- 优先用真实建模问题引入；
- 不要直接堆公式；
- 说明旧方法哪里不够。

### 5. `core_concept`
Purpose:
- 给出本讲最核心的概念定义。

Content rules:
- 一页一个概念，避免一页塞太多新术语；
- 可以使用 `block`, `alertblock`, `exampleblock`。

### 6. `core_formula_or_geometry`
Purpose:
- 展示公式、几何图像或概率解释。

Content rules:
- 先给公式，再解释每个项；
- 公式页必须配口语化解释；
- 如果图比公式更有效，优先图。

### 7. `comparison_or_taxonomy`
Purpose:
- 让学生看到多个方法之间的差异与联系。

Content rules:
- 推荐使用表格；
- 推荐维度：目标、优点、风险、适用场景；
- 不要在同一页比较超过 4 个对象。

### 8. `figure_or_visual_intuition`
Purpose:
- 用图像替代大段抽象说明。

Content rules:
- 每张图必须带一句“该图想让学生看懂什么”；
- 图页最好只保留 2~4 个讲解点；
- 推荐图：bias-variance 曲线、regularization path、PCA 投影图、ROC 曲线。

### 9. `case_study_or_workflow`
Purpose:
- 把理论放入真实建模流程。

Content rules:
- 写清数据、任务类型、挑战和方法选择理由；
- 如果是实验页，至少说明输入、输出、评价指标。

### 10. `code_or_pseudocode_optional`
Purpose:
- 展示最小可用实现或流程骨架。

Content rules:
- 不要贴太长代码；
- 优先展示 8~20 行的关键片段；
- 如果代码不是本讲重点，可以改成伪代码。

### 11. `discussion_question`
Purpose:
- 促进课堂互动与方法比较。

Content rules:
- 优先提“取舍型”问题；
- 例如：`Lasso vs PCR`、`RMSE vs MAE`、`解释性 vs 预测性`。

### 12. `summary`
Purpose:
- 回收本讲主线。

Content rules:
- 最好压缩成 3 条关键结论；
- 其中 1 条负责连接下一讲。

### 13. `assignment_or_after_class`
Purpose:
- 给出课后延伸或作业建议。

Content rules:
- 题目不要只有推导；
- 最好包含“解释 + 实验 + 比较”三种任务之一。

### 14. `qa`
Purpose:
- 结束页。

Content rules:
- 尽量简洁。

## Content Density Rules
适用于大多数页面：
- 每页主标题只表达一个中心主题；
- 单页 bullet 建议 `3~5` 条；
- 每条 bullet 尽量不超过两行；
- 一页如果已经有公式或图，文字就要减少；
- 如果一页内容必须很多，优先拆成两页。

## Language Rules
- 默认使用中文讲述；
- 英文术语保留原词，并尽量第一次出现时给中文解释；
- 数学符号使用标准记号；
- 业务解释优先用短句，不用长段落。

## Visual Rules
- 默认使用 `16:9` 比例；
- 默认使用 `Madrid` 主题，不要每份课件随意切换主题；
- 尽量避免一页超过两层嵌套 bullet；
- 如果使用表格，列数建议不超过 4；
- 如果使用图片，优先保证“图意明确”而不是装饰性。

## AI Generation Checklist
当 AI 基于模板生成新课件时，建议逐项自检：

1. 是否明确填写了标题、周次、主题、日期？
2. 是否有清晰的 `learning objectives`？
3. 是否先解释问题，再给公式？
4. 是否至少有一页对比页？
5. 是否至少有一页图示或可视化直觉页？
6. 是否至少有一页案例或 workflow 页？
7. 是否最后回收到 3 条关键结论？
8. 是否避免单页信息过载？
9. 是否避免把整段讲义原文直接贴成 bullet？
10. 是否让学生能看出“这一讲为什么存在”？

## Font Note
- 中文正文字体默认使用 `楷体` 风格；
- 中文标题字体默认使用 `行楷` 风格；
- 如果本机缺少 `行楷` 字体，模板会自动回退到 `楷体` 风格。

## Compile Note
推荐编译方式：

```bash
xelatex slides/template/course_beamer_template.tex
```

如果后续新增图片资源，建议统一放在：
- `slides/template/figures/`
- 或每周课件自己的 `slides/weekXX_figures/`
