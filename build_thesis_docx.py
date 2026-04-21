from __future__ import annotations

import re
from pathlib import Path

from docx import Document
from docx.enum.section import WD_SECTION_START
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Pt


ROOT = Path(r"C:\Coding\tourist-comments-thesis")
TEMPLATE = ROOT / r"2026届毕业论文模板\人文社科类\1.毕业论文模板 .docx"
SOURCE = ROOT / "本科毕业论文初稿-大数据背景下旅游评论挖掘及景区推荐研究.md"
OUTPUT = ROOT / "本科毕业论文-正式排版稿.docx"


TITLE = "大数据背景下旅游评论挖掘及景区推荐研究"
SUBTITLE = "以去哪儿为例"
LEVEL3_COUNTER = {}


def clear_document(doc: Document) -> None:
    body = doc._element.body
    sect_pr = body.sectPr
    for child in list(body):
        if child is not sect_pr:
            body.remove(child)


def set_run_font(run, name: str, size: int, bold: bool = False) -> None:
    run.font.name = name
    if run._element.rPr is None:
        run._element.get_or_add_rPr()
    run._element.rPr.rFonts.set(qn("w:eastAsia"), name)
    run.font.size = Pt(size)
    run.bold = bold


def set_paragraph_style(paragraph, style_name: str | None) -> None:
    if style_name:
        try:
            paragraph.style = style_name
        except KeyError:
            pass


def add_paragraph(
    doc: Document,
    text: str = "",
    *,
    align=WD_ALIGN_PARAGRAPH.JUSTIFY,
    font_name: str = "宋体",
    font_size: int = 12,
    bold: bool = False,
    first_line_chars: int | None = None,
    line_spacing: float | None = 1.25,
    space_before: int = 0,
    space_after: int = 0,
    style_name: str | None = None,
):
    p = doc.add_paragraph()
    set_paragraph_style(p, style_name)
    p.alignment = align
    p.paragraph_format.space_before = Pt(space_before)
    p.paragraph_format.space_after = Pt(space_after)
    if line_spacing is not None:
        p.paragraph_format.line_spacing = line_spacing
    if first_line_chars is not None:
        p.paragraph_format.first_line_indent = Pt(first_line_chars * 10)
    run = p.add_run(text)
    set_run_font(run, font_name, font_size, bold=bold)
    return p


def add_field(paragraph, instruction: str) -> None:
    run = paragraph.add_run()
    fld_char_begin = OxmlElement("w:fldChar")
    fld_char_begin.set(qn("w:fldCharType"), "begin")
    run._r.append(fld_char_begin)

    instr_run = paragraph.add_run()
    instr_text = OxmlElement("w:instrText")
    instr_text.set(qn("xml:space"), "preserve")
    instr_text.text = instruction
    instr_run._r.append(instr_text)

    sep_run = paragraph.add_run()
    fld_char_sep = OxmlElement("w:fldChar")
    fld_char_sep.set(qn("w:fldCharType"), "separate")
    sep_run._r.append(fld_char_sep)

    end_run = paragraph.add_run()
    fld_char_end = OxmlElement("w:fldChar")
    fld_char_end.set(qn("w:fldCharType"), "end")
    end_run._r.append(fld_char_end)


def add_tab_stop(p, pos_twips: int) -> None:
    p_pr = p._p.get_or_add_pPr()
    tabs = p_pr.find(qn("w:tabs"))
    if tabs is None:
        tabs = OxmlElement("w:tabs")
        p_pr.append(tabs)
    tab = OxmlElement("w:tab")
    tab.set(qn("w:val"), "left")
    tab.set(qn("w:pos"), str(pos_twips))
    tabs.append(tab)


def parse_markdown_sections(text: str):
    lines = text.splitlines()
    sections = []
    current = None
    buffer: list[str] = []

    def flush():
        nonlocal buffer, current
        if current is not None:
            current["content"] = "\n".join(buffer).strip()
            sections.append(current)
        buffer = []

    for line in lines:
        if line.startswith("# "):
            continue
        if line.startswith("## "):
            flush()
            current = {"level": 2, "title": line[3:].strip(), "content": ""}
            continue
        if line.startswith("### "):
            flush()
            current = {"level": 3, "title": line[4:].strip(), "content": ""}
            continue
        if line.startswith("#### "):
            flush()
            current = {"level": 4, "title": line[5:].strip(), "content": ""}
            continue
        buffer.append(line)
    flush()
    return sections


def split_paragraphs(content: str) -> list[str]:
    parts = re.split(r"\n\s*\n", content.strip())
    return [p.strip() for p in parts if p.strip()]


def write_cover(doc: Document) -> None:
    add_paragraph(doc, "中图分类号：TN384    密级：公开", align=WD_ALIGN_PARAGRAPH.RIGHT, font_name="黑体", font_size=12, line_spacing=1.0)
    add_paragraph(doc, "", align=WD_ALIGN_PARAGRAPH.CENTER, line_spacing=1.0)
    add_paragraph(doc, "本科毕业论文", align=WD_ALIGN_PARAGRAPH.CENTER, font_name="黑体", font_size=22, bold=True, line_spacing=1.0)
    add_paragraph(doc, "", align=WD_ALIGN_PARAGRAPH.CENTER, line_spacing=1.0)
    add_paragraph(doc, TITLE, align=WD_ALIGN_PARAGRAPH.CENTER, font_name="黑体", font_size=18, bold=True, line_spacing=1.0)
    add_paragraph(doc, SUBTITLE, align=WD_ALIGN_PARAGRAPH.CENTER, font_name="黑体", font_size=18, bold=True, line_spacing=1.0)
    add_paragraph(doc, "", align=WD_ALIGN_PARAGRAPH.CENTER, line_spacing=1.0)

    for label in ["学    院", "专    业", "班    级", "学    生", "学    号", "指导教师"]:
        p = add_paragraph(doc, "", align=WD_ALIGN_PARAGRAPH.CENTER, font_name="宋体", font_size=16, line_spacing=1.25)
        add_tab_stop(p, 2800)
        add_tab_stop(p, 6200)
        r1 = p.add_run(label)
        set_run_font(r1, "宋体", 16, False)
        r2 = p.add_run("\t：\t________________")
        set_run_font(r2, "宋体", 16, False)

    add_paragraph(doc, "", align=WD_ALIGN_PARAGRAPH.CENTER, line_spacing=1.0)
    add_paragraph(doc, "二〇二六年四月", align=WD_ALIGN_PARAGRAPH.CENTER, font_name="宋体", font_size=16)


def write_abstracts(doc: Document, sections) -> None:
    abstract = next(s for s in sections if s["title"] == "摘要")
    english = next(s for s in sections if s["title"] == "Abstract")

    doc.add_section(WD_SECTION_START.NEW_PAGE)
    add_paragraph(doc, "摘    要", align=WD_ALIGN_PARAGRAPH.CENTER, font_name="黑体", font_size=15, bold=True, line_spacing=1.0, space_before=10, space_after=8)
    for para in split_paragraphs(abstract["content"]):
        p = add_paragraph(doc, para, first_line_chars=2, line_spacing=1.25, space_before=0, space_after=0)
        p.paragraph_format.first_line_indent = Pt(24)

    p = add_paragraph(doc, "", font_name="宋体", font_size=10, line_spacing=1.25, space_before=8, space_after=0)
    r1 = p.add_run("关键词：")
    set_run_font(r1, "黑体", 10, True)
    r2 = p.add_run("大数据；旅游评论；文本挖掘；景区推荐；情感分析；去哪儿")
    set_run_font(r2, "宋体", 10, bold=False)
    p.paragraph_format.first_line_indent = Pt(24)

    doc.add_section(WD_SECTION_START.NEW_PAGE)
    add_paragraph(doc, "ABSTRACT", align=WD_ALIGN_PARAGRAPH.CENTER, font_name="Times New Roman", font_size=15, bold=True, line_spacing=1.0, space_before=10, space_after=8)
    for para in split_paragraphs(english["content"]):
        p = add_paragraph(doc, para, font_name="Times New Roman", first_line_chars=2, line_spacing=1.25, space_before=0, space_after=0)
        p.paragraph_format.first_line_indent = Pt(24)

    p = add_paragraph(doc, "", font_name="Times New Roman", font_size=10, line_spacing=1.25, space_before=8, space_after=0)
    p.paragraph_format.first_line_indent = Pt(24)
    r1 = p.add_run("Key words: ")
    set_run_font(r1, "Times New Roman", 10, True)
    r2 = p.add_run("big data; tourism reviews; text mining; scenic spot recommendation; sentiment analysis; Qunar")
    set_run_font(r2, "Times New Roman", 10, bold=False)


def write_toc(doc: Document) -> None:
    doc.add_section(WD_SECTION_START.NEW_PAGE)
    add_paragraph(doc, "目    录", align=WD_ALIGN_PARAGRAPH.CENTER, font_name="黑体", font_size=15, bold=True, line_spacing=1.0)
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.LEFT
    add_field(p, r'TOC \o "1-3" \h \z \u')


def normalize_heading(title: str) -> str:
    mapping = {
        "一、绪论": "一、绪论",
        "二、提出问题：旅游评论挖掘与景区推荐的理论基础": "二、旅游评论挖掘与景区推荐的理论基础",
        "三、分析问题：研究设计与数据处理": "三、研究设计与数据处理",
        "四、分析问题：旅游评论挖掘实证分析": "四、旅游评论挖掘实证分析",
        "五、分析问题：景区推荐模型构建与结果分析": "五、景区推荐模型构建与结果分析",
        "六、解决问题：系统设计、应用价值与优化路径": "六、系统设计、应用价值与优化路径",
        "七、结论与展望": "七、结论与展望",
        "参考文献": "参考文献",
    }
    return mapping.get(title, title)


CN_NUMS = "一二三四五六七八九十"


def to_cn_index(index: int) -> str:
    if 1 <= index <= 10:
        return CN_NUMS[index - 1]
    return str(index)


def format_level3(title: str) -> str:
    m = re.match(r"^(\d+\.\d+)\s*(.*)$", title.strip())
    if m:
        return f"{m.group(1)} {m.group(2).strip()}"
    return title.strip()


def format_level4(title: str, idx: int) -> str:
    raw = re.sub(r"^\d+\.\d+\.\d+\s*", "", title).strip()
    return f"（{to_cn_index(idx)}）{raw}"


def write_body(doc: Document, sections) -> None:
    doc.add_section(WD_SECTION_START.NEW_PAGE)
    current_level2 = 0
    current_level3 = 0
    current_level4 = 0

    for sec in sections:
        title = sec["title"]
        if title in {"摘要", "Abstract", "以去哪儿为例"}:
            continue

        if sec["level"] == 2:
            if title in {"七、结论与展望", "参考文献"}:
                doc.add_section(WD_SECTION_START.NEW_PAGE)
            current_level2 += 1
            current_level3 = 0
            current_level4 = 0
            add_paragraph(
                doc,
                normalize_heading(title),
                align=WD_ALIGN_PARAGRAPH.CENTER,
                font_name="黑体",
                font_size=15,
                bold=True,
                line_spacing=1.5,
                space_before=8,
                space_after=8,
                style_name="Heading 1",
            )
            if title == "参考文献":
                for para in split_paragraphs(sec["content"]):
                    add_paragraph(doc, para, font_name="宋体", font_size=12, first_line_chars=0, line_spacing=1.25)
                continue

        elif sec["level"] == 3:
            current_level3 += 1
            current_level4 = 0
            heading = format_level3(title)
            add_paragraph(
                doc,
                heading,
                align=WD_ALIGN_PARAGRAPH.LEFT,
                font_name="黑体",
                font_size=14,
                bold=True,
                line_spacing=1.5,
                space_before=6,
                space_after=4,
                style_name="Heading 2",
            )
        elif sec["level"] == 4:
            current_level4 += 1
            heading = format_level4(title, current_level4)
            add_paragraph(
                doc,
                heading,
                align=WD_ALIGN_PARAGRAPH.LEFT,
                font_name="黑体",
                font_size=12,
                bold=True,
                line_spacing=1.5,
                space_before=4,
                space_after=2,
                style_name="Heading 3",
            )

        for para in split_paragraphs(sec["content"]):
            clean = para.replace("**", "")
            add_paragraph(doc, clean, first_line_chars=2)


def main() -> None:
    text = SOURCE.read_text(encoding="utf-8")
    sections = parse_markdown_sections(text)

    doc = Document(TEMPLATE)
    clear_document(doc)

    write_cover(doc)
    write_abstracts(doc, sections)
    write_toc(doc)
    write_body(doc, sections)

    doc.save(OUTPUT)
    print(f"已生成：{OUTPUT}")


if __name__ == "__main__":
    main()
