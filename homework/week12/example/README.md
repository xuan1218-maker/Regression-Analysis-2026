# Week 12 Example

这个目录提供一个 **参考实现骨架**，把 Week 12 的 bias-variance / RMSE-MAE 演示整理成了一个可提交的脚本工作流。

## 文件说明
- `main.py`：唯一执行入口
- `figures/`：运行后生成的图像目录
- `results/summary.md`：运行后生成的 Markdown 总结

## 参考价值
这个 example 主要演示三件事：
1. 如何把课堂里的“现象演示”翻译成 `main.py` 工作流；
2. 如何让代码按照阶段输出图和结果；
3. 如何把实验结论整理成 `summary.md`。

## 注意
- 这是一个参考实现，不要求学生逐字照抄；
- 学生正式提交时，仍然应按照自己的目录组织和已有 `utils/` 能力进行适配；
- 如果学生已经在 `utils/metrics.py` 中实现了 `RMSE` / `MAE`，正式作业中应优先复用自己的实现。

## 运行方式
如果你的环境已经安装了：
- `numpy`
- `pandas`
- `matplotlib`
- `scikit-learn`

那么可以在该目录下执行：

```bash
python main.py
```

运行后将自动生成：
- `figures/candidate_models.png`
- `figures/error_curves.png`
- `figures/variance_demo.png`
- `figures/loss_outlier_comparison.png`
- `results/summary.md`
