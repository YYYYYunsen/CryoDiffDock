# 论文图件深海配色

`recolor_paper_figures.py` 会读取当前论文图件目录，将三种配色分别生成到
`visualizations/paper_figures_deep_sea/`。原图不会被覆盖。

双击 `scripts/recolor_paper_figures.command` 可一次生成三套方案。命令行也可只生成一种：

```bash
python scripts/recolor_paper_figures.py --palette style3_blue_green
```

可用主题：

- `style1_blue`：`#6CBADB #96C2D4 #BAD2E1 #D8E5F7 #DBF1FA`。同一蓝色梯度，适合连续值或强调单一主线。
- `style2_green`：`#519D78 #88CF88 #AADCA9`。三类绿色，适合三组以内对照。
- `style3_blue_green`：从参考图的 12 个色块中挑出 8 个，选择兼顾深浅与蓝绿层次的颜色：`#045798 #3492B2 #58B8D1 #6FC8CA #92C2A6 #BCF4C5 #ACEEFE #D6F6FF`。

参考图实际显示了 12 个色块；第三套按请求收敛为 8 色，去掉了相近的浅绿，避免多类别作图时颜色太接近。浅色适合填充，深蓝和中间蓝适合主线、强调项及小面积标记。图中文字、灰色元素和白底保留。

程序默认处理 Figure 1–6 当前的成稿导出目录，优先使用 SVG；会跨图统一旧色到新色的映射，并同步处理 SVG 中嵌入的 PNG。每张图会输出可编辑 SVG、矢量 PDF 和 PNG 预览。Figure 1 当前只有 PDF 成稿，因此会以 180 dpi 转成位图后换色，再写出 PNG 和 PDF；其清晰度受现有 PDF 中位图内容限制。输出清单和每张图的颜色映射记录在 `manifest.csv`。

运行依赖：Python 3、Pillow、NumPy、`rsvg-convert` 和 Ghostscript（`gs`）。如要处理自己的图件文件夹：

```bash
python scripts/recolor_paper_figures.py --input-dir path/to/figures --palette all
```

`--output-dir` 可指定另一个输出目录，`--png-dpi` 可调整预览分辨率。若后续有更新的定稿导出，请在脚本顶部 `DEFAULT_SOURCES` 中调整对应目录。
