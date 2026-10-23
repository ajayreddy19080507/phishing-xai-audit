"""Small helpers to edit the IEEE-template docx (styles come from the user's template)."""
import copy, re
import docx
from docx.oxml import parse_xml
from docx.oxml.ns import qn, nsdecls
from docx.shared import Inches
from docx.text.paragraph import Paragraph

W_NS = 'xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"'
TAG = re.compile(r"(</?(?:i|b|sub|sup)>)")


def _mk_run(text, fmt):
    r = parse_xml(f"<w:r {W_NS}/>")
    if fmt:
        rpr = parse_xml(f"<w:rPr {W_NS}/>")
        if "b" in fmt: rpr.append(parse_xml(f"<w:b {W_NS}/>"))
        if "i" in fmt: rpr.append(parse_xml(f"<w:i {W_NS}/>"))
        if "sub" in fmt: rpr.append(parse_xml(f'<w:vertAlign {W_NS} w:val="subscript"/>'))
        if "sup" in fmt: rpr.append(parse_xml(f'<w:vertAlign {W_NS} w:val="superscript"/>'))
        r.append(rpr)
    text = re.sub(r"(?<![\w\)])-(?=\d)", "\u2212", text)   # typographic minus for negative numbers
    parts = text.split("\t")
    for k, part in enumerate(parts):
        if k > 0: r.append(parse_xml(f"<w:tab {W_NS}/>"))
        if part:
            t = parse_xml(f'<w:t {W_NS} xml:space="preserve"/>'); t.text = part; r.append(t)
    return r


def set_markup(p_el, markup, base_rpr=None):
    """Replace all runs of paragraph element by runs parsed from a tiny markup (<i>, <b>, <sub>, <sup>)."""
    for ch in list(p_el):
        if ch.tag != qn("w:pPr"): p_el.remove(ch)
    fmt = set()
    for tok in TAG.split(markup):
        if not tok: continue
        m = re.fullmatch(r"<(/?)(i|b|sub|sup)>", tok)
        if m:
            (fmt.discard if m.group(1) else fmt.add)(m.group(2)); continue
        p_el.append(_mk_run(tok, set(fmt)))
    return p_el


def clone(proto, markup=None):
    el = copy.deepcopy(proto)
    if markup is not None: set_markup(el, markup)
    return el


def picture_paragraph(doc, proto_fig, path, width_in):
    el = copy.deepcopy(proto_fig)
    for ch in list(el):
        if ch.tag != qn("w:pPr"): el.remove(ch)
    Paragraph(el, doc).add_run().add_picture(path, width=Inches(width_in))
    return el


# ----------------------------------------------------------------------------- tables
def _cell(text, w, span=1, bold=False, align="center", bottom=False, vmerge=None, top=False, keep=True):
    borders = ""
    if bottom or top:
        borders = "<w:tcBorders>" + (f'<w:top w:val="single" w:sz="4" w:space="0" w:color="000000"/>' if top else "") + \
                  (f'<w:bottom w:val="single" w:sz="4" w:space="0" w:color="000000"/>' if bottom else "") + "</w:tcBorders>"
    vm = ""
    if vmerge == "restart": vm = '<w:vMerge w:val="restart"/>'
    elif vmerge == "cont": vm = "<w:vMerge/>"
    gs = f'<w:gridSpan w:val="{span}"/>' if span > 1 else ""
    tc = parse_xml(f'<w:tc {W_NS}><w:tcPr><w:tcW w:w="{w}" w:type="dxa"/>{gs}{vm}{borders}<w:vAlign w:val="center"/></w:tcPr></w:tc>')
    p = parse_xml(f'<w:p {W_NS}><w:pPr><w:pStyle w:val="tablecopy"/>{"<w:keepNext/>" if keep else ""}<w:spacing w:before="0" w:after="0"/><w:jc w:val="{align}"/></w:pPr></w:p>')
    tc.append(p)
    # text: allow markup, force 7 pt
    for tok in TAG.split(text):
        if not tok: continue
        m = re.fullmatch(r"<(/?)(i|b|sub|sup)>", tok)
        if m: continue
    fmt_stack = set(["b"]) if bold else set()
    for tok in TAG.split(text):
        if not tok: continue
        m = re.fullmatch(r"<(/?)(i|b|sub|sup)>", tok)
        if m:
            (fmt_stack.discard if m.group(1) else fmt_stack.add)(m.group(2)); continue
        r = _mk_run(tok, set(fmt_stack))
        rpr = r.find(qn("w:rPr"))
        if rpr is None:
            rpr = parse_xml(f"<w:rPr {W_NS}/>"); r.insert(0, rpr)
        rpr.append(parse_xml(f'<w:sz {W_NS} w:val="14"/>'))
        p.append(r)
    return tc


def make_table(rows, widths, header_rows=1, first_align="left", bold_cells=(), spans=None):
    """rows: list of lists of strings. spans: {(r,c): n} column spans (the next n-1 cells must be omitted from the row).
    bold_cells: set of (r,c) to bold. Header rows are bold and the last one gets a rule underneath."""
    spans = spans or {}
    total = sum(widths)
    tbl = parse_xml(
        f'<w:tbl {W_NS}><w:tblPr><w:tblW w:w="{total}" w:type="dxa"/><w:jc w:val="center"/>'
        '<w:tblBorders><w:top w:val="single" w:sz="6" w:space="0" w:color="000000"/>'
        '<w:left w:val="single" w:sz="6" w:space="0" w:color="000000"/>'
        '<w:bottom w:val="single" w:sz="6" w:space="0" w:color="000000"/>'
        '<w:right w:val="single" w:sz="6" w:space="0" w:color="000000"/>'
        '<w:insideH w:val="single" w:sz="4" w:space="0" w:color="000000"/>'
        '<w:insideV w:val="single" w:sz="4" w:space="0" w:color="000000"/></w:tblBorders>'
        '<w:tblLayout w:type="fixed"/><w:tblCellMar><w:top w:w="15" w:type="dxa"/><w:left w:w="40" w:type="dxa"/>'
        '<w:bottom w:w="15" w:type="dxa"/><w:right w:w="40" w:type="dxa"/></w:tblCellMar>'
        '<w:tblLook w:val="04A0" w:firstRow="1" w:lastRow="0" w:firstColumn="1" w:lastColumn="0" w:noHBand="0" w:noVBand="1"/></w:tblPr>'
        "<w:tblGrid>" + "".join(f'<w:gridCol w:w="{w}"/>' for w in widths) + "</w:tblGrid></w:tbl>")
    for ri, row in enumerate(rows):
        tr = parse_xml(f"<w:tr {W_NS}><w:trPr><w:cantSplit/></w:trPr></w:tr>")
        covered = set()
        for ci, cell in enumerate(row):
            if ci in covered: continue
            span = spans.get((ri, ci), 1)
            covered.update(range(ci + 1, ci + span))
            w = sum(widths[ci:ci + span])
            hdr = ri < header_rows
            tr.append(_cell(cell, w, span=span, bold=hdr or (ri, ci) in bold_cells,
                            align=first_align if ci == 0 else "center",
                            bottom=(hdr and ri == header_rows - 1) or (hdr and span > 1 and ri < header_rows - 1), keep=(ri < len(rows) - 1)))
        tbl.append(tr)
    return tbl
