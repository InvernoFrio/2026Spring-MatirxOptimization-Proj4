"""Build the project presentation without external PowerPoint dependencies."""

from __future__ import annotations

import csv
import html
import struct
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "output" / "figures"
RESULT_DIR = ROOT / "output" / "results"
PPTX_PATH = ROOT / "Pegasos_SVM_Report.pptx"

EMU_PER_INCH = 914400
SLIDE_W = int(13.333333 * EMU_PER_INCH)
SLIDE_H = int(7.5 * EMU_PER_INCH)

BG = "F7F9FB"
NAVY = "0B1F33"
TEAL = "2A9D8F"
CORAL = "E76F51"
BLUE = "457B9D"
GRAY = "536271"
LIGHT = "E7EEF5"
WHITE = "FFFFFF"


def emu(inches: float) -> int:
    return int(inches * EMU_PER_INCH)


def esc(text) -> str:
    return html.escape(str(text), quote=False)


def png_size(path: Path) -> tuple[int, int]:
    data = path.read_bytes()
    if data[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError(f"Only PNG images are supported: {path}")
    return struct.unpack(">II", data[16:24])


def fit_image(path: Path, x: float, y: float, w: float, h: float):
    img_w, img_h = png_size(path)
    box_w, box_h = emu(w), emu(h)
    scale = min(box_w / img_w, box_h / img_h)
    final_w = int(img_w * scale)
    final_h = int(img_h * scale)
    final_x = emu(x) + (box_w - final_w) // 2
    final_y = emu(y) + (box_h - final_h) // 2
    return final_x, final_y, final_w, final_h


def solid_rect(shape_id: int, x: float, y: float, w: float, h: float, color: str, radius=False):
    prst = "roundRect" if radius else "rect"
    return f"""
      <p:sp>
        <p:nvSpPr><p:cNvPr id="{shape_id}" name="Rectangle {shape_id}"/><p:cNvSpPr/><p:nvPr/></p:nvSpPr>
        <p:spPr>
          <a:xfrm><a:off x="{emu(x)}" y="{emu(y)}"/><a:ext cx="{emu(w)}" cy="{emu(h)}"/></a:xfrm>
          <a:prstGeom prst="{prst}"><a:avLst/></a:prstGeom>
          <a:solidFill><a:srgbClr val="{color}"/></a:solidFill>
          <a:ln><a:noFill/></a:ln>
        </p:spPr>
      </p:sp>"""


def run_xml(text: str, size: int, color: str, bold=False):
    b = ' b="1"' if bold else ""
    return (
        f'<a:r><a:rPr lang="zh-CN" sz="{size}"{b}>'
        f'<a:solidFill><a:srgbClr val="{color}"/></a:solidFill>'
        '<a:latin typeface="Microsoft YaHei"/><a:ea typeface="Microsoft YaHei"/>'
        f'</a:rPr><a:t>{esc(text)}</a:t></a:r>'
    )


def textbox(
    shape_id: int,
    x: float,
    y: float,
    w: float,
    h: float,
    lines,
    size=2200,
    color=NAVY,
    bold=False,
    align="l",
    fill=None,
    radius=False,
):
    if isinstance(lines, str):
        lines = [lines]
    bg_xml = '<a:noFill/>' if fill is None else f'<a:solidFill><a:srgbClr val="{fill}"/></a:solidFill>'
    geom = "roundRect" if radius else "rect"
    paras = []
    for idx, line in enumerate(lines):
        line_text, line_size, line_color, line_bold = line, size, color, bold
        if isinstance(line, dict):
            line_text = line["text"]
            line_size = line.get("size", size)
            line_color = line.get("color", color)
            line_bold = line.get("bold", bold)
        paras.append(
            f'<a:p><a:pPr algn="{align}"/>{run_xml(line_text, line_size, line_color, line_bold)}'
            f'<a:endParaRPr lang="zh-CN" sz="{line_size}"/></a:p>'
        )
    return f"""
      <p:sp>
        <p:nvSpPr><p:cNvPr id="{shape_id}" name="TextBox {shape_id}"/><p:cNvSpPr txBox="1"/><p:nvPr/></p:nvSpPr>
        <p:spPr>
          <a:xfrm><a:off x="{emu(x)}" y="{emu(y)}"/><a:ext cx="{emu(w)}" cy="{emu(h)}"/></a:xfrm>
          <a:prstGeom prst="{geom}"><a:avLst/></a:prstGeom>
          {bg_xml}
          <a:ln><a:noFill/></a:ln>
        </p:spPr>
        <p:txBody>
          <a:bodyPr wrap="square" lIns="91440" tIns="68580" rIns="91440" bIns="68580"/>
          <a:lstStyle/>
          {''.join(paras)}
        </p:txBody>
      </p:sp>"""


def image_xml(shape_id: int, rid: str, path: Path, x: float, y: float, w: float, h: float):
    ix, iy, iw, ih = fit_image(path, x, y, w, h)
    return f"""
      <p:pic>
        <p:nvPicPr>
          <p:cNvPr id="{shape_id}" name="{esc(path.name)}"/>
          <p:cNvPicPr><a:picLocks noChangeAspect="1"/></p:cNvPicPr>
          <p:nvPr/>
        </p:nvPicPr>
        <p:blipFill><a:blip r:embed="{rid}"/><a:stretch><a:fillRect/></a:stretch></p:blipFill>
        <p:spPr>
          <a:xfrm><a:off x="{ix}" y="{iy}"/><a:ext cx="{iw}" cy="{ih}"/></a:xfrm>
          <a:prstGeom prst="rect"><a:avLst/></a:prstGeom>
        </p:spPr>
      </p:pic>"""


def slide_xml(elements):
    body = "\n".join(elements)
    return f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<p:sld xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main"
       xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"
       xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main">
  <p:cSld>
    <p:bg><p:bgPr><a:solidFill><a:srgbClr val="{BG}"/></a:solidFill><a:effectLst/></p:bgPr></p:bg>
    <p:spTree>
      <p:nvGrpSpPr><p:cNvPr id="1" name=""/><p:cNvGrpSpPr/><p:nvPr/></p:nvGrpSpPr>
      <p:grpSpPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="0" cy="0"/><a:chOff x="0" y="0"/><a:chExt cx="0" cy="0"/></a:xfrm></p:grpSpPr>
      {body}
    </p:spTree>
  </p:cSld>
  <p:clrMapOvr><a:masterClrMapping/></p:clrMapOvr>
</p:sld>"""


def title_bar(title: str, subtitle: str | None = None):
    elements = [
        solid_rect(2, 0, 0, 13.333, 0.18, TEAL),
        textbox(3, 0.55, 0.35, 9.5, 0.55, title, size=3000, color=NAVY, bold=True),
    ]
    if subtitle:
        elements.append(textbox(4, 10.15, 0.39, 2.6, 0.45, subtitle, size=1250, color=GRAY, align="r"))
    return elements


def metric_card(shape_id, x, title, value, note, color):
    return [
        textbox(shape_id, x, 5.75, 2.35, 0.95, [
            {"text": value, "size": 2300, "color": color, "bold": True},
            {"text": title, "size": 1050, "color": NAVY, "bold": True},
            {"text": note, "size": 850, "color": GRAY},
        ], fill=WHITE, radius=True)
    ]


def read_csv(path: Path):
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def build_slides():
    real_rows = read_csv(RESULT_DIR / "real_dataset_results.csv")
    nonlinear_rows = read_csv(RESULT_DIR / "nonlinear_svm_results.csv")
    rcv1 = real_rows[0] if real_rows else {}
    linear = next((r for r in nonlinear_rows if r.get("model") == "linear_pegasos"), {})
    rbf = next((r for r in nonlinear_rows if r.get("model") == "rbf_kernel_smo"), {})

    slides = []

    slides.append({
        "elements": [
            solid_rect(2, 0, 0, 13.333, 7.5, BG),
            solid_rect(3, 0, 0, 13.333, 0.22, TEAL),
            textbox(4, 0.72, 1.05, 8.4, 1.15, "大规模文本分类中的 Pegasos SVM", size=3600, color=NAVY, bold=True),
            textbox(5, 0.75, 2.15, 8.2, 0.55, "从二分类 SVM 到多分类扩展，以及真实数据实验分析", size=1650, color=GRAY),
            textbox(6, 0.78, 3.05, 4.9, 1.3, [
                "实际任务：新闻主题识别、评论情感分类",
                "核心方法：线性 SVM + Pegasos 随机优化",
                "实验数据：RCV1、Amazon Review Polarity",
            ], size=1450, color=NAVY, fill=WHITE, radius=True),
            {"image": FIG_DIR / "decision_boundary.png", "x": 7.05, "y": 1.05, "w": 5.55, "h": 4.35},
            textbox(7, 0.78, 6.55, 11.8, 0.32, "课程项目汇报 | Pegasos Linear SVM", size=1050, color=GRAY),
        ],
    })

    slides.append({
        "elements": title_bar("1. 实际问题引入", "为什么需要大规模线性 SVM") + [
            textbox(5, 0.72, 1.1, 4.9, 4.25, [
                "• 文本分类常见于新闻分发、搜索过滤、舆情分析和商品评论情感识别。",
                "• 文本向量化后维度很高，但每条文本只包含少量非零词特征。",
                "• 传统核 SVM 在样本多时核矩阵开销大，难以直接扩展到百万级。",
                "• Pegasos 直接优化线性 SVM 的 primal 目标，适合高维稀疏大规模数据。",
            ], size=1450, color=NAVY),
            *metric_card(6, 6.05, "RCV1 实验抽样", "50k / 10k", "训练 / 测试", TEAL),
            *metric_card(7, 8.65, "特征维度", "47,236", "稀疏文本特征", BLUE),
            *metric_card(8, 11.25, "Amazon 全量", "3.6M / 400k", "训练 / 测试", CORAL),
            textbox(9, 6.05, 1.15, 6.1, 3.75, [
                {"text": "本项目回答的问题", "size": 1700, "color": NAVY, "bold": True},
                "• 如何从 SVM 的最大间隔思想落到可运行代码？",
                "• 如何用 Pegasos 避免大规模文本上的高训练成本？",
                "• 如何把二分类 SVM 扩展到多分类？",
                "• 非线性核 SVM 在什么场景下更合适？",
            ], size=1350, color=NAVY, fill=WHITE, radius=True),
        ],
    })

    slides.append({
        "elements": title_bar("2. 整体流程", "数据到图像结果") + [
            textbox(5, 0.75, 1.15, 2.35, 1.15, ["真实数据", "RCV1 / Amazon"], size=1500, color=WHITE, bold=True, fill=TEAL, radius=True, align="c"),
            textbox(6, 3.65, 1.15, 2.35, 1.15, ["文本向量化", "Hashing / sparse"], size=1450, color=WHITE, bold=True, fill=BLUE, radius=True, align="c"),
            textbox(7, 6.55, 1.15, 2.35, 1.15, ["Pegasos SVM", "mini-batch SGD"], size=1450, color=WHITE, bold=True, fill=CORAL, radius=True, align="c"),
            textbox(8, 9.45, 1.15, 2.35, 1.15, ["评估输出", "CSV + 图像"], size=1450, color=WHITE, bold=True, fill=NAVY, radius=True, align="c"),
            textbox(9, 3.1, 1.42, 0.5, 0.4, "→", size=2200, color=NAVY, bold=True, align="c"),
            textbox(10, 6.0, 1.42, 0.5, 0.4, "→", size=2200, color=NAVY, bold=True, align="c"),
            textbox(11, 8.9, 1.42, 0.5, 0.4, "→", size=2200, color=NAVY, bold=True, align="c"),
            textbox(12, 0.9, 3.1, 5.2, 2.85, [
                {"text": "代码对应关系", "size": 1600, "color": NAVY, "bold": True},
                "• src/pegasos.py：BinaryPegasosSVM 与 OneVsRestPegasosSVM",
                "• scripts/run_real_datasets.py：下载、抽样、训练、保存图表",
                "• output/figures：汇报图像素材",
                "• output/results：实验指标 CSV",
            ], size=1230, color=NAVY, fill=WHITE, radius=True),
            {"image": FIG_DIR / "real_dataset_training_curves.png", "x": 6.55, "y": 2.75, "w": 5.8, "h": 3.55},
        ],
    })

    slides.append({
        "elements": title_bar("3. SVM 二分类数学原理", "最大间隔与 hinge loss") + [
            textbox(5, 0.7, 1.05, 5.3, 4.75, [
                {"text": "线性分类器", "size": 1500, "color": TEAL, "bold": True},
                "f(x) = wᵀx + b,    ŷ = sign(f(x))",
                "",
                {"text": "最大间隔目标", "size": 1500, "color": TEAL, "bold": True},
                "min  1/2 ||w||²",
                "s.t.  yᵢ(wᵀxᵢ + b) ≥ 1",
                "",
                {"text": "软间隔 / hinge loss", "size": 1500, "color": TEAL, "bold": True},
                "min  λ/2||w||² + 1/n Σ max(0, 1 - yᵢ(wᵀxᵢ+b))",
            ], size=1250, color=NAVY, fill=WHITE, radius=True),
            {"image": FIG_DIR / "decision_boundary.png", "x": 6.35, "y": 1.0, "w": 6.25, "h": 5.35},
            textbox(6, 0.85, 6.05, 11.65, 0.45, "图中实线是决策边界，虚线是 margin；违反 margin 的样本会产生 hinge loss。", size=1150, color=GRAY),
        ],
    })

    slides.append({
        "elements": title_bar("4. 优化算法：SMO 与 Pegasos", "为什么主线选择 Pegasos") + [
            textbox(5, 0.8, 1.1, 5.55, 2.45, [
                {"text": "SMO：优化对偶问题", "size": 1650, "color": BLUE, "bold": True},
                "• 变量是 αᵢ，每次选择两个 α 更新。",
                "• 可以配合核函数处理非线性边界。",
                "• 核矩阵通常需要 O(n²) 内存，不适合百万级文本。",
            ], size=1300, color=NAVY, fill=WHITE, radius=True),
            textbox(6, 0.8, 3.85, 5.55, 2.45, [
                {"text": "Pegasos：优化 primal 问题", "size": 1650, "color": TEAL, "bold": True},
                "ηₜ = 1 / (λt)",
                "若 yᵢ(wᵀxᵢ+b) < 1：",
                "w ← (1-ηₜλ)w + ηₜyᵢxᵢ",
                "否则只做正则收缩。",
            ], size=1300, color=NAVY, fill=WHITE, radius=True),
            {"image": FIG_DIR / "loss_curve.png", "x": 6.8, "y": 1.25, "w": 5.35, "h": 3.55},
            textbox(7, 6.9, 5.15, 5.05, 0.75, "目标函数整体下降，说明 mini-batch Pegasos 在逐步降低 hinge loss 与正则项。", size=1150, color=GRAY),
        ],
    })

    slides.append({
        "elements": title_bar("5. Pegasos 在代码中的应用", "训练日志与稀疏矩阵") + [
            textbox(5, 0.72, 1.1, 5.6, 4.65, [
                {"text": "核心训练逻辑", "size": 1600, "color": NAVY, "bold": True},
                "margins = yb * (Xb @ w + b)",
                "active = margins < 1.0",
                "correction = X_active.T @ y_active / batch_size",
                "",
                {"text": "调试输出", "size": 1600, "color": NAVY, "bold": True},
                "• active_rate：违反 margin 的比例",
                "• objective：目标函数值",
                "• sample_acc：训练子样本准确率",
                "• epoch_seconds：每轮训练耗时",
            ], size=1180, color=NAVY, fill=WHITE, radius=True),
            {"image": FIG_DIR / "rcv1_ccat_training_curve.png", "x": 6.55, "y": 1.05, "w": 5.65, "h": 4.15},
            textbox(6, 6.65, 5.55, 5.5, 0.65, "RCV1 训练曲线用于展示目标函数和训练子样本准确率的变化。", size=1100, color=GRAY),
        ],
    })

    slides.append({
        "elements": title_bar("6. 从二分类到多分类", "One-vs-Rest 扩展") + [
            textbox(5, 0.75, 1.05, 5.35, 4.8, [
                {"text": "思想", "size": 1600, "color": TEAL, "bold": True},
                "K 个类别训练 K 个二分类器。",
                "",
                "第 k 个分类器：",
                "yᵢ(k)=+1,  当 yᵢ=k",
                "yᵢ(k)=-1,  当 yᵢ≠k",
                "",
                "预测时：",
                "ŷ = argmaxₖ (wₖᵀx + bₖ)",
                "",
                "代码：OneVsRestPegasosSVM",
            ], size=1300, color=NAVY, fill=WHITE, radius=True),
            {"image": FIG_DIR / "multiclass_confusion_matrix.png", "x": 6.25, "y": 1.05, "w": 5.75, "h": 4.9},
            textbox(6, 6.5, 6.0, 5.2, 0.45, "混淆矩阵对角线越集中，多分类预测越稳定。", size=1120, color=GRAY),
        ],
    })

    rcv1_acc = float(rcv1.get("test_accuracy", 0) or 0)
    rcv1_time = float(rcv1.get("training_seconds", 0) or 0)
    slides.append({
        "elements": title_bar("7. 真实数据实验结果", "RCV1 与 Amazon 文本分类") + [
            {"image": FIG_DIR / "real_dataset_report.png", "x": 0.45, "y": 1.0, "w": 8.1, "h": 5.65},
            textbox(5, 8.85, 1.15, 3.65, 1.05, [
                {"text": f"{rcv1_acc:.3f}", "size": 2450, "color": TEAL, "bold": True},
                "RCV1 测试准确率",
            ], size=1150, color=NAVY, fill=WHITE, radius=True, align="c"),
            textbox(6, 8.85, 2.45, 3.65, 1.05, [
                {"text": f"{rcv1_time:.2f}s", "size": 2450, "color": CORAL, "bold": True},
                "5 epoch 训练耗时",
            ], size=1150, color=NAVY, fill=WHITE, radius=True, align="c"),
            textbox(7, 8.85, 3.75, 3.65, 1.85, [
                "• 当前 CSV 中记录的是 RCV1 抽样实验。",
                "• Amazon 图像已生成训练曲线。",
                "• 若使用 --amazon-full，可跑 3.6M 训练样本。",
            ], size=1080, color=NAVY, fill=WHITE, radius=True),
        ],
    })

    slides.append({
        "elements": title_bar("8. Amazon Review Polarity 说明", "百万级数据与默认抽样") + [
            {"image": FIG_DIR / "amazon_polarity_training_curve.png", "x": 0.75, "y": 1.05, "w": 6.2, "h": 4.75},
            textbox(5, 7.3, 1.05, 4.85, 4.75, [
                {"text": "为什么之前只训练 5 万？", "size": 1650, "color": NAVY, "bold": True},
                "• Amazon Polarity 本身约 3.6M 训练样本、400k 测试样本。",
                "• 脚本默认 50k / 10k 是快速演示配置，便于首次下载和调试。",
                "• 全量运行可使用 --amazon-full。",
                "• 文本特征保持稀疏矩阵，训练复杂度主要与非零词项数量相关。",
                "",
                "推荐汇报说法：默认抽样用于快速验证；项目支持全量百万级实验。",
            ], size=1170, color=NAVY, fill=WHITE, radius=True),
        ],
    })

    rbf_acc = float(rbf.get("test_accuracy", 0) or 0)
    linear_acc = float(linear.get("test_accuracy", 0) or 0)
    slides.append({
        "elements": title_bar("9. K-SVM 一笔带过", "RBF kernel 用于非线性数据") + [
            {"image": FIG_DIR / "nonlinear_svm_decision_boundary.png", "x": 0.55, "y": 1.05, "w": 8.0, "h": 5.2},
            textbox(5, 8.85, 1.2, 3.65, 4.55, [
                {"text": "补充结论", "size": 1650, "color": NAVY, "bold": True},
                f"• 线性 Pegasos：acc={linear_acc:.3f}",
                f"• RBF K-SVM：acc={rbf_acc:.3f}",
                "• RBF kernel 能形成弯曲边界。",
                "• 但核矩阵开销约 O(n²)，因此只作为小规模非线性任务补充。",
                "• 大规模文本主线仍采用线性 Pegasos。",
            ], size=1120, color=NAVY, fill=WHITE, radius=True),
        ],
    })

    slides.append({
        "elements": title_bar("10. 总结", "方法选择与项目成果") + [
            textbox(5, 0.8, 1.1, 5.8, 4.65, [
                {"text": "项目产出", "size": 1700, "color": TEAL, "bold": True},
                "• 实现 BinaryPegasosSVM：线性二分类 SVM。",
                "• 实现 OneVsRestPegasosSVM：二分类到多分类。",
                "• 支持 scipy.sparse：适配高维文本数据。",
                "• 添加训练调试输出和自动汇报图。",
                "• 添加 K-SVM/RBF/SMO 作为非线性补充。",
            ], size=1280, color=NAVY, fill=WHITE, radius=True),
            textbox(6, 7.0, 1.1, 4.9, 4.65, [
                {"text": "汇报主结论", "size": 1700, "color": CORAL, "bold": True},
                "• SVM 的核心是最大化 margin。",
                "• Pegasos 把 SVM 训练变成高效随机优化。",
                "• 稀疏文本上训练快，是算法和数据结构共同作用的结果。",
                "• K-SVM 更适合展示非线性边界，不适合直接跑百万级文本。",
                "",
                "Q & A",
            ], size=1280, color=NAVY, fill=WHITE, radius=True),
        ],
    })

    return slides


def content_types(slide_count: int, image_exts):
    overrides = [
        '<Override PartName="/ppt/presentation.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.presentation.main+xml"/>',
        '<Override PartName="/ppt/slideMasters/slideMaster1.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.slideMaster+xml"/>',
        '<Override PartName="/ppt/slideLayouts/slideLayout1.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.slideLayout+xml"/>',
        '<Override PartName="/ppt/theme/theme1.xml" ContentType="application/vnd.openxmlformats-officedocument.theme+xml"/>',
        '<Override PartName="/docProps/core.xml" ContentType="application/vnd.openxmlformats-package.core-properties+xml"/>',
        '<Override PartName="/docProps/app.xml" ContentType="application/vnd.openxmlformats-officedocument.extended-properties+xml"/>',
    ]
    for idx in range(1, slide_count + 1):
        overrides.append(
            f'<Override PartName="/ppt/slides/slide{idx}.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.slide+xml"/>'
        )
    defaults = [
        '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>',
        '<Default Extension="xml" ContentType="application/xml"/>',
    ]
    if "png" in image_exts:
        defaults.append('<Default Extension="png" ContentType="image/png"/>')
    return f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  {''.join(defaults)}
  {''.join(overrides)}
</Types>"""


def rels(items):
    rows = []
    for rid, rel_type, target in items:
        rows.append(f'<Relationship Id="{rid}" Type="{rel_type}" Target="{target}"/>')
    return f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  {''.join(rows)}
</Relationships>"""


def presentation_xml(slide_count: int):
    ids = []
    for idx in range(1, slide_count + 1):
        ids.append(f'<p:sldId id="{255 + idx}" r:id="rId{idx + 1}"/>')
    return f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<p:presentation xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main"
                xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"
                xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main">
  <p:sldMasterIdLst><p:sldMasterId id="2147483648" r:id="rId1"/></p:sldMasterIdLst>
  <p:sldIdLst>{''.join(ids)}</p:sldIdLst>
  <p:sldSz cx="{SLIDE_W}" cy="{SLIDE_H}" type="wide"/>
  <p:notesSz cx="6858000" cy="9144000"/>
</p:presentation>"""


def slide_master_xml():
    return f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<p:sldMaster xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main"
             xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"
             xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main">
  <p:cSld><p:spTree><p:nvGrpSpPr><p:cNvPr id="1" name=""/><p:cNvGrpSpPr/><p:nvPr/></p:nvGrpSpPr><p:grpSpPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="0" cy="0"/><a:chOff x="0" y="0"/><a:chExt cx="0" cy="0"/></a:xfrm></p:grpSpPr></p:spTree></p:cSld>
  <p:clrMap bg1="lt1" tx1="dk1" bg2="lt2" tx2="dk2" accent1="accent1" accent2="accent2" accent3="accent3" accent4="accent4" accent5="accent5" accent6="accent6" hlink="hlink" folHlink="folHlink"/>
  <p:sldLayoutIdLst><p:sldLayoutId id="2147483649" r:id="rId1"/></p:sldLayoutIdLst>
  <p:txStyles><p:titleStyle/><p:bodyStyle/><p:otherStyle/></p:txStyles>
</p:sldMaster>"""


def slide_layout_xml():
    return """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<p:sldLayout xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main"
             xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"
             xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main"
             type="blank" preserve="1">
  <p:cSld name="Blank"><p:spTree><p:nvGrpSpPr><p:cNvPr id="1" name=""/><p:cNvGrpSpPr/><p:nvPr/></p:nvGrpSpPr><p:grpSpPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="0" cy="0"/><a:chOff x="0" y="0"/><a:chExt cx="0" cy="0"/></a:xfrm></p:grpSpPr></p:spTree></p:cSld>
  <p:clrMapOvr><a:masterClrMapping/></p:clrMapOvr>
</p:sldLayout>"""


def theme_xml():
    return f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<a:theme xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" name="PegasosTheme">
  <a:themeElements>
    <a:clrScheme name="Pegasos">
      <a:dk1><a:srgbClr val="{NAVY}"/></a:dk1><a:lt1><a:srgbClr val="{WHITE}"/></a:lt1>
      <a:dk2><a:srgbClr val="{GRAY}"/></a:dk2><a:lt2><a:srgbClr val="{BG}"/></a:lt2>
      <a:accent1><a:srgbClr val="{TEAL}"/></a:accent1><a:accent2><a:srgbClr val="{CORAL}"/></a:accent2>
      <a:accent3><a:srgbClr val="{BLUE}"/></a:accent3><a:accent4><a:srgbClr val="F4A261"/></a:accent4>
      <a:accent5><a:srgbClr val="6C757D"/></a:accent5><a:accent6><a:srgbClr val="2D6A4F"/></a:accent6>
      <a:hlink><a:srgbClr val="{BLUE}"/></a:hlink><a:folHlink><a:srgbClr val="{CORAL}"/></a:folHlink>
    </a:clrScheme>
    <a:fontScheme name="PegasosFonts">
      <a:majorFont><a:latin typeface="Microsoft YaHei"/><a:ea typeface="Microsoft YaHei"/></a:majorFont>
      <a:minorFont><a:latin typeface="Microsoft YaHei"/><a:ea typeface="Microsoft YaHei"/></a:minorFont>
    </a:fontScheme>
    <a:fmtScheme name="PegasosFmt">
      <a:fillStyleLst>
        <a:solidFill><a:schemeClr val="phClr"/></a:solidFill>
        <a:gradFill rotWithShape="1"><a:gsLst><a:gs pos="0"><a:schemeClr val="phClr"/></a:gs><a:gs pos="100000"><a:schemeClr val="phClr"/></a:gs></a:gsLst><a:lin ang="5400000" scaled="0"/></a:gradFill>
        <a:solidFill><a:schemeClr val="phClr"/></a:solidFill>
      </a:fillStyleLst>
      <a:lnStyleLst>
        <a:ln w="6350"><a:solidFill><a:schemeClr val="phClr"/></a:solidFill><a:prstDash val="solid"/></a:ln>
        <a:ln w="12700"><a:solidFill><a:schemeClr val="phClr"/></a:solidFill><a:prstDash val="solid"/></a:ln>
        <a:ln w="19050"><a:solidFill><a:schemeClr val="phClr"/></a:solidFill><a:prstDash val="solid"/></a:ln>
      </a:lnStyleLst>
      <a:effectStyleLst><a:effectStyle><a:effectLst/></a:effectStyle><a:effectStyle><a:effectLst/></a:effectStyle><a:effectStyle><a:effectLst/></a:effectStyle></a:effectStyleLst>
      <a:bgFillStyleLst>
        <a:solidFill><a:schemeClr val="phClr"/></a:solidFill>
        <a:solidFill><a:schemeClr val="phClr"/></a:solidFill>
        <a:solidFill><a:schemeClr val="phClr"/></a:solidFill>
      </a:bgFillStyleLst>
    </a:fmtScheme>
  </a:themeElements>
</a:theme>"""


def core_props_xml():
    return """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<cp:coreProperties xmlns:cp="http://schemas.openxmlformats.org/package/2006/metadata/core-properties"
                   xmlns:dc="http://purl.org/dc/elements/1.1/"
                   xmlns:dcterms="http://purl.org/dc/terms/"
                   xmlns:dcmitype="http://purl.org/dc/dcmitype/"
                   xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">
  <dc:title>Pegasos SVM 项目汇报</dc:title>
  <dc:subject>大规模文本分类中的 Pegasos SVM</dc:subject>
  <dc:creator>Codex</dc:creator>
  <cp:lastModifiedBy>Codex</cp:lastModifiedBy>
</cp:coreProperties>"""


def app_props_xml(slide_count: int):
    return f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Properties xmlns="http://schemas.openxmlformats.org/officeDocument/2006/extended-properties"
            xmlns:vt="http://schemas.openxmlformats.org/officeDocument/2006/docPropsVTypes">
  <Application>Microsoft PowerPoint</Application>
  <PresentationFormat>宽屏</PresentationFormat>
  <Slides>{slide_count}</Slides>
  <Notes>0</Notes>
</Properties>"""


def build_pptx():
    slides = build_slides()
    media = []
    image_exts = set()
    rendered_slides = []

    for slide in slides:
        elements = []
        rel_items = [("rId1", "http://schemas.openxmlformats.org/officeDocument/2006/relationships/slideLayout", "../slideLayouts/slideLayout1.xml")]
        shape_id = 20
        image_no = 0
        for element in slide["elements"]:
            if isinstance(element, dict) and "image" in element:
                image_no += 1
                shape_id += 1
                path = element["image"]
                if not path.exists():
                    continue
                media_name = f"image{len(media) + 1}{path.suffix.lower()}"
                media.append((path, media_name))
                image_exts.add(path.suffix.lower().lstrip("."))
                rid = f"rId{image_no + 1}"
                rel_items.append((rid, "http://schemas.openxmlformats.org/officeDocument/2006/relationships/image", f"../media/{media_name}"))
                elements.append(image_xml(shape_id, rid, path, element["x"], element["y"], element["w"], element["h"]))
            else:
                elements.append(element)
        rendered_slides.append((slide_xml(elements), rel_items))

    PPTX_PATH.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(PPTX_PATH, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("[Content_Types].xml", content_types(len(slides), image_exts))
        zf.writestr("_rels/.rels", rels([
            ("rId1", "http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument", "ppt/presentation.xml"),
            ("rId2", "http://schemas.openxmlformats.org/package/2006/relationships/metadata/core-properties", "docProps/core.xml"),
            ("rId3", "http://schemas.openxmlformats.org/officeDocument/2006/relationships/extended-properties", "docProps/app.xml"),
        ]))
        zf.writestr("docProps/core.xml", core_props_xml())
        zf.writestr("docProps/app.xml", app_props_xml(len(slides)))
        pres_rels = [("rId1", "http://schemas.openxmlformats.org/officeDocument/2006/relationships/slideMaster", "slideMasters/slideMaster1.xml")]
        for idx in range(1, len(slides) + 1):
            pres_rels.append((f"rId{idx + 1}", "http://schemas.openxmlformats.org/officeDocument/2006/relationships/slide", f"slides/slide{idx}.xml"))
        zf.writestr("ppt/presentation.xml", presentation_xml(len(slides)))
        zf.writestr("ppt/_rels/presentation.xml.rels", rels(pres_rels))
        zf.writestr("ppt/slideMasters/slideMaster1.xml", slide_master_xml())
        zf.writestr("ppt/slideMasters/_rels/slideMaster1.xml.rels", rels([
            ("rId1", "http://schemas.openxmlformats.org/officeDocument/2006/relationships/slideLayout", "../slideLayouts/slideLayout1.xml"),
            ("rId2", "http://schemas.openxmlformats.org/officeDocument/2006/relationships/theme", "../theme/theme1.xml"),
        ]))
        zf.writestr("ppt/slideLayouts/slideLayout1.xml", slide_layout_xml())
        zf.writestr("ppt/slideLayouts/_rels/slideLayout1.xml.rels", rels([
            ("rId1", "http://schemas.openxmlformats.org/officeDocument/2006/relationships/slideMaster", "../slideMasters/slideMaster1.xml")
        ]))
        zf.writestr("ppt/theme/theme1.xml", theme_xml())

        for idx, (xml, rel_items) in enumerate(rendered_slides, start=1):
            zf.writestr(f"ppt/slides/slide{idx}.xml", xml)
            zf.writestr(f"ppt/slides/_rels/slide{idx}.xml.rels", rels(rel_items))
        for src, media_name in media:
            zf.writestr(f"ppt/media/{media_name}", src.read_bytes())

    print(PPTX_PATH)


if __name__ == "__main__":
    build_pptx()
