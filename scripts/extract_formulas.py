#!/usr/bin/env python3
"""
Extract ALL mathematical formulas from a .docx thesis file.

Handles:
  1. OMML math (<m:oMath>, <m:oMathPara>) — Office Math Markup Language
  2. Embedded images (word/media/*.png) — formula screenshots
  3. Plain text formulas — patterns like C_lp, C_dict, equations with =, subscripts, etc.

Also extracts the document section/chapter structure by reading heading styles.
"""

import zipfile
import xml.etree.ElementTree as ET
import re
import os
import html

# ── Paths ──────────────────────────────────────────────────────────────────
DOCX_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "thesis",
    "Final Year Paper, Similarity and AI report",
    "Final Year Project1.docx",
)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# ── XML Namespaces ─────────────────────────────────────────────────────────
NS = {
    "w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main",
    "m": "http://schemas.openxmlformats.org/officeDocument/2006/math",
    "wp": "http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing",
    "a": "http://schemas.openxmlformats.org/drawingml/2006/main",
    "pic": "http://schemas.openxmlformats.org/drawingml/2006/picture",
    "r": "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
    "wps": "http://schemas.microsoft.com/office/word/2010/wordprocessingShape",
}
for prefix, uri in NS.items():
    ET.register_namespace(prefix, uri)

# Heading styleId → level mapping (from word/styles.xml analysis)
HEADING_STYLES = {"2": 1, "3": 2, "4": 3, "25": 1}  # 25 = "INTRODUCTION" / "METHODOLOGY" style
STYLE_NAME_MAP = {}  # populated from styles.xml


# ═══════════════════════════════════════════════════════════════════════════
# 1.  Load & parse the document
# ═══════════════════════════════════════════════════════════════════════════

def load_docx_xml(path):
    """Return (root_element, zipfile_handle, relationship_map)."""
    z = zipfile.ZipFile(path)
    data = z.read("word/document.xml")
    root = ET.fromstring(data)

    # Build relationship map: rId → target (e.g. "media/image1.png")
    rels_data = z.read("word/_rels/document.xml.rels")
    rels_root = ET.fromstring(rels_data)
    rel_map = {}
    for rel in rels_root:
        rid = rel.get("Id")
        target = rel.get("Target")
        rtype = rel.get("Type")
        if rid and target:
            rel_map[rid] = {"target": target, "type": rtype}

    # Load heading names from styles.xml
    styles_data = z.read("word/styles.xml")
    styles_root = ET.fromstring(styles_data)
    for style_el in styles_root:
        sid = style_el.get("{}".format("{http://schemas.openxmlformats.org/wordprocessingml/2006/main}styleId"))
        if sid is None:
            sid = style_el.get("{http://schemas.openxmlformats.org/wordprocessingml/2006/main}styleId")
        name_el = style_el.find(".//w:name", NS)
        if sid and name_el is not None:
            STYLE_NAME_MAP[sid] = name_el.get("{http://schemas.openxmlformats.org/wordprocessingml/2006/main}val")

    return root, z, rel_map


# ═══════════════════════════════════════════════════════════════════════════
# 2.  Extract paragraph information
# ═══════════════════════════════════════════════════════════════════════════

def get_paragraph_text(para):
    """Concatenate all <w:t> text within a paragraph."""
    texts = []
    for t in para.findall(".//w:t", NS):
        if t.text:
            texts.append(t.text)
    return "".join(texts)


def get_paragraph_style(para):
    """Return the styleId (e.g. '2', '3', '17') or None."""
    pPr = para.find("w:pPr", NS)
    if pPr is not None:
        pStyle = pPr.find("w:pStyle", NS)
        if pStyle is not None:
            return pStyle.get("{http://schemas.openxmlformats.org/wordprocessingml/2006/main}val")
    return None


def para_has_image(para):
    """Check if paragraph contains an embedded image.  Returns list of image info dicts."""
    images = []
    for inline in para.findall(".//wp:inline", NS):
        docPr = inline.find(".//wp:docPr", NS)
        blip = inline.find(".//a:blip", NS)
        if blip is not None and docPr is not None:
            embed = blip.get("{http://schemas.openxmlformats.org/officeDocument/2006/relationships}embed")
            images.append({
                "name": docPr.get("name", ""),
                "id": docPr.get("id", ""),
                "embed": embed,
            })
    return images


def get_paragraphs_info(root):
    """Return list of dicts for every paragraph in document body."""
    body = root.find("w:body", NS)
    if body is None:
        raise ValueError("Could not find w:body in document.xml")
    paragraphs = body.findall("w:p", NS)

    results = []
    for para in paragraphs:
        text = get_paragraph_text(para).strip()
        style = get_paragraph_style(para)
        images = para_has_image(para)
        results.append({
            "text": text,
            "style": style,
            "images": images,
            "element": para,
        })
    return results


# ═══════════════════════════════════════════════════════════════════════════
# 3.  Extract OMML formulas
# ═══════════════════════════════════════════════════════════════════════════

def extract_omml_text(math_elem):
    """Extract readable text from an OMML math element."""
    parts = []
    for t in math_elem.findall(".//m:t", NS):
        if t.text:
            parts.append(t.text)
    return "".join(parts)


def find_omml_formulas(paragraphs_info):
    """Search each paragraph for <m:oMath> or <m:oMathPara> elements."""
    formulas = []
    for pinfo in paragraphs_info:
        para = pinfo["element"]
        math_elems = para.findall(".//m:oMath", NS)
        math_para_elems = para.findall(".//m:oMathPara", NS)
        all_math = math_elems + math_para_elems
        for m in all_math:
            expr = extract_omml_text(m)
            if expr.strip():
                formulas.append({
                    "expression": expr.strip(),
                    "paragraph_index": None,  # will be filled later
                    "format": "OMML",
                })
    return formulas


# ═══════════════════════════════════════════════════════════════════════════
# 4.  Extract image references (potential formula screenshots)
# ═══════════════════════════════════════════════════════════════════════════

def find_images(paragraphs_info, rel_map):
    """Find all embedded images and their surrounding text context."""
    images = []
    for idx, pinfo in enumerate(paragraphs_info):
        for img_info in pinfo["images"]:
            embed = img_info.get("embed", "")
            rel_entry = rel_map.get(embed, {})
            target = rel_entry.get("target", "")
            images.append({
                "file_name": img_info["name"],
                "embed_rid": embed,
                "target": target,
                "paragraph_index": idx,
                "paragraph_text": pinfo["text"][:200],
            })
    return images


# ═══════════════════════════════════════════════════════════════════════════
# 5.  Extract plain-text formulas
# ═══════════════════════════════════════════════════════════════════════════

# Patterns for mathematical expressions common in this thesis
PLAIN_TEXT_PATTERNS = [
    # Named confidence/score variables with subscripts
    (r"C_lp", "C_lp — VLM log-probability confidence"),
    (r"C_dict", "C_dict — dictionary/Levenshtein confidence"),
    (r"C_final", "C_final — combined final confidence"),
    (r"C_raw", "C_raw — raw exponentiated confidence"),
    (r"w_lp", "w_lp — weight for log-probability confidence"),
    (r"w_dict", "w_dict — weight for dictionary confidence"),
    (r"avg_lp", "avg_lp — average log-probability"),
    (r"lp_k", "lp_k — per-token log-probability"),
    (r"P\(t_k\)", "P(t_k) — per-token probability"),
    (r"D\[i\]\[j\]", "D[i][j] — Levenshtein edit distance matrix entry"),

    # Full equations (containing = and mathematical notation)
    (r"sigma\(z_i\)\s*=\s*e\^?\{?z_i\}?\s*/\s*sum", "Sigma(z_i) = e^{z_i} / sum — Softmax function"),
    (r"logprob_i\s*=\s*ln", "logprob_i = ln(...) — Log-probability definition"),
    (r"C_final\s*=\s*\(?\s*w_lp", "C_final = (w_lp * C_lp + w_dict * C_dict) / (w_lp + w_dict) — Combined confidence"),
    (r"C_raw\s*=\s*e\^?\{?avg_lp\}?", "C_raw = exp(avg_lp) — Raw confidence from exponentiation"),
    (r"avg_lp\s*=\s*\(?\s*1/m", "avg_lp = (1/m) * sum(lp_k) — Average log-probability"),
    (r"C_dict\s*=\s*max\(0\.0,\s*1\.0\s*-\s*lev", "C_dict = max(0.0, 1.0 - lev(a,b)/max(...)) — Dictionary confidence"),
    (r"D\[0\]\[j\]\s*=\s*j", "D[0][j] = j — Levenshtein base case (insertions)"),
    (r"D\[i\]\[0\]\s*=\s*[ai]", "D[i][0] = a/i — Levenshtein base case (deletions)"),
    (r"D\[i-1\]\[j\]\s*\+\s*1.*D\[i\]\[j-1\]\s*\+\s*1.*D\[i-1\]\[j-1\]", "D[i][j] = min(...) — Levenshtein recurrence relation"),
    (r"lev\(a,\s*b\)", "lev(a,b) — Levenshtein distance between strings a and b"),

    # Accuracy / metric patterns
    (r"\d+%", "Percentage value (accuracy/rate)"),
]


def find_plain_text_formulas(paragraph_text, para_idx, heading_path):
    """Find formula-like patterns in plain text."""
    found = []

    # Skip very long paragraphs that are prose, not formulas
    if len(paragraph_text) > 400:
        return found

    text_lower = paragraph_text.lower()

    # Track which patterns matched to avoid duplicates
    matched_spans = []

    for pattern, description in PLAIN_TEXT_PATTERNS:
        for match in re.finditer(pattern, paragraph_text, re.IGNORECASE):
            start, end = match.span()
            # Avoid duplicate overlapping matches
            if any(m_start <= start < m_end or m_start < end <= m_end
                   for (m_start, m_end) in matched_spans):
                # Only skip if this is a subset of a larger match
                if any(m_start <= start and m_end >= end for (m_start, m_end) in matched_spans):
                    continue
            matched_spans.append((start, end))

            expr = match.group().strip()
            # Truncate very long expressions
            if len(expr) > 120:
                expr = expr[:117] + "..."

            # Get surrounding context (up to 30 chars before/after)
            ctx_before = paragraph_text[max(0, start - 40):start].strip()
            ctx_after = paragraph_text[end:min(len(paragraph_text), end + 40)].strip()

            found.append({
                "expression": expr,
                "context_before": ctx_before,
                "context_after": ctx_after,
                "format": "Plain text",
                "paragraph_index": para_idx,
                "description": description,
            })

    return found


# ═══════════════════════════════════════════════════════════════════════════
# 6.  Build section/chapter heading hierarchy
# ═══════════════════════════════════════════════════════════════════════════

def build_heading_hierarchy(paragraphs_info):
    """
    Walk paragraphs and track the current heading stack.
    Returns a list with heading info and a function to get the current section path.
    """
    # Style "2" = Heading 1 (CHAPTER ONE, CHAPTER TWO, etc.)
    # Style "25" = sub-heading 1 (INTRODUCTION, METHODOLOGY, etc.)
    # Style "3" = Heading 2 (1.1, 2.1, 3.1, etc.)
    # Style "4" = Heading 3 (1.5.1, 3.2.1, etc.)

    heading_levels = {}  # para_index → (level, heading_text)

    heading_stack = {}  # level → heading_text for active headings

    for idx, pinfo in enumerate(paragraphs_info):
        style = pinfo["style"]
        text = pinfo["text"]

        if not text:
            continue

        # Determine heading level from style
        level = None
        if style == "2":
            level = 1  # CHAPTER ONE, CHAPTER TWO, LIST OF FIGURES, etc.
        elif style == "25":
            level = 1  # INTRODUCTION, METHODOLOGY, CONCLUSION (same level as chapter)
        elif style == "3":
            level = 2  # 1.1, 2.1, 3.1, etc.
        elif style == "4":
            level = 3  # 1.5.1, 3.2.1, etc.

        if level is not None:
            # Store this heading
            heading_stack[level] = text
            # Clear any sub-headings at lower levels
            for l in range(level + 1, 4):
                heading_stack.pop(l, None)
            heading_levels[idx] = (level, text)

    # Build a function to get current heading path for any paragraph index
    # Pre-compute the heading path for every paragraph
    heading_paths = {}
    current_heading_stack = {}

    for idx, pinfo in enumerate(paragraphs_info):
        style = pinfo["style"]
        text = pinfo["text"]

        if idx in heading_levels:
            level, heading_text = heading_levels[idx]
            current_heading_stack[level] = text
            for l in range(level + 1, 4):
                current_heading_stack.pop(l, None)

        # Build path from current heading stack
        path_parts = []
        for lvl in sorted(current_heading_stack.keys()):
            path_parts.append(current_heading_stack[lvl])
        heading_paths[idx] = " > ".join(path_parts) if path_parts else "(No heading)"

    return heading_paths, heading_levels


# ═══════════════════════════════════════════════════════════════════════════
# 7.  Known formula descriptions from document content
# ═══════════════════════════════════════════════════════════════════════════

# Manual mapping of key formulas based on document sections
KNOWN_FORMULAS = {
    "sigma(z_i) = e^{z_i} / sum": "Softmax function: converts raw logits into a probability distribution over the vocabulary",
    "logprob_i = ln": "Log probability: natural logarithm of the softmax probability, maps (0,1] to (-inf, 0]",
    "avg_lp = (1/m) * sum": "Average log-probability: mean of per-token log probabilities from the VLM",
    "C_raw = exp(avg_lp)": "Raw confidence: exponentiation of the average log-probability, equivalent to geometric mean of token probabilities",
    "C_lp": "VLM-derived confidence based on average log-probability of generated tokens",
    "C_dict": "Dictionary confidence: based on normalised Levenshtein distance between extracted text and dictionary reference",
    "C_final = (w_lp * C_lp + w_dict * C_dict) / (w_lp + w_dict)": "Combined final confidence: weighted fusion of VLM confidence and dictionary confidence, bounded in [0, 1]",
    "D[0][j] = j": "Levenshtein distance base case: j insertions needed to build string b from empty prefix",
    "D[i][0] = i": "Levenshtein distance base case: i deletions needed to reduce string a to empty prefix",
    "D[i][j]": "Levenshtein distance recurrence: minimum of deletion, insertion, or substitution cost",
    "C_dict = max(0.0, 1.0 - lev(a,b)/max(len(a), len(b), 1))": "Dictionary confidence: 1.0 for exact match, decreasing linearly with edit distance",
    "C_final": "Final combined confidence score used for auto-accept/review decisions",
    "w_lp": "Weight assigned to VLM log-probability confidence in the fusion formula",
    "w_dict": "Weight assigned to dictionary confidence in the fusion formula",
    "accuracy": "Field-level accuracy metric comparing extracted values against ground truth",
    "lev(a, b)": "Levenshtein distance: minimum number of single-character edits to transform string a into string b",
}


# ═══════════════════════════════════════════════════════════════════════════
# 8.  Main extraction function
# ═══════════════════════════════════════════════════════════════════════════

def extract_formulas(docx_path):
    """Main function to extract all formulas from the docx."""

    print(f"📄 Opening docx: {docx_path}")
    root, zf, rel_map = load_docx_xml(docx_path)

    print("📋 Extracting paragraphs...")
    paragraphs_info = get_paragraphs_info(root)
    print(f"   Found {len(paragraphs_info)} paragraphs")

    # Build heading hierarchy
    print("🏷️  Building heading hierarchy...")
    heading_paths, heading_levels = build_heading_hierarchy(paragraphs_info)
    print(f"   Found {len(heading_levels)} heading elements")

    # ── OMML formulas ──
    print("🧮 Searching for OMML formulas...")
    omml_formulas = find_omml_formulas(paragraphs_info)
    print(f"   Found {len(omml_formulas)} OMML formula(s)")

    # ── Images ──
    print("🖼️  Searching for embedded images...")
    images = find_images(paragraphs_info, rel_map)
    print(f"   Found {len(images)} embedded image(s)")
    for img in images:
        print(f"      - {img['file_name']} ({img['target']})")

    # ── Plain text formulas ──
    print("📝 Searching for plain text formulas...")
    all_plain_text = []
    for idx, pinfo in enumerate(paragraphs_info):
        text = pinfo["text"]
        if text:
            formulas = find_plain_text_formulas(text, idx, heading_paths.get(idx, ""))
            all_plain_text.extend(formulas)
    print(f"   Found {len(all_plain_text)} plain text formula reference(s)")

    # ── Merge and deduplicate ──
    print("\n🔍 Consolidating formula report...")

    # Build the final structured report
    formula_report = []

    # Track which paragraphs already contributed a key formula
    covered_paragraphs = set()

    # Helper: get context before a paragraph
    def get_context(pinfo, paragraphs_info, idx, lookback=3):
        """Get surrounding text for context."""
        before_parts = []
        for i in range(max(0, idx - lookback), idx):
            t = paragraphs_info[i]["text"]
            if t:
                before_parts.append(t)
        before = " ... ".join(before_parts[-3:]) if before_parts else ""
        after_parts = []
        for i in range(idx + 1, min(len(paragraphs_info), idx + 1 + lookback)):
            t = paragraphs_info[i]["text"]
            if t:
                after_parts.append(t)
        after = " ... ".join(after_parts[:3]) if after_parts else ""
        return before, after

    # ── Process OMML formulas ──
    for f in omml_formulas:
        formula_report.append({
            "type": "OMML Equation",
            "expression": f["expression"],
            "section": "",
            "description": "Office Math Markup Language formula",
            "context_before": "",
        })

    # ── Process images: identify which are likely formula screenshots ──
    # Based on the document analysis:
    #   rId10 → media/image1.png  → signature/drawing (DECLARATION)
    #   rId11 → media/image2.png  → DAPE Pipeline Architecture (Section 3.2)
    #   rId12 → media/image3.png  → Preprocessing data flow (Section 3.2.1)
    #   rId13 → media/image4.png  → Confidence Scoring diagram (Section 3.2.5)
    #   rId14 → media/image5.png  → HITL flowchart (Section 3.2.6)
    #   rId15 → media/image6.png  → Accuracy comparison chart (Section 4.11.4)
    #   rId16 → media/image7.png  → HITL impact chart (Section 4.11.4)
    formula_images = {
        "rId10": {  # media/image1.png
            "label": "Declaration signature / drawing mark",
            "is_formula": False,
            "section": "DECLARATION",
        },
        "rId11": {  # media/image2.png
            "label": "DAPE Pipeline Architecture overview diagram",
            "is_formula": False,
            "section": "3.2 System Design and DAPE Pipeline Architecture",
        },
        "rId12": {  # media/image3.png
            "label": "Preprocessing Module data flow diagram",
            "is_formula": False,
            "section": "3.2.1 Preprocessing Module",
        },
        "rId13": {  # media/image4.png
            "label": "Confidence Scoring and Validation Logic diagram (may contain formula references)",
            "is_formula": False,
            "section": "3.2.5 Confidence Scoring and Validation Module",
        },
        "rId14": {  # media/image5.png
            "label": "Human-in-the-Loop escalation flowchart",
            "is_formula": False,
            "section": "3.2.6 Human-in-the-Loop Module",
        },
        "rId15": {  # media/image6.png
            "label": "Bar chart: accuracy comparison across extraction approaches",
            "is_formula": False,
            "section": "4.11.4 Results and Justification of the Topic",
        },
        "rId16": {  # media/image7.png
            "label": "Bar chart: HITL correction impact on field-level accuracy",
            "is_formula": False,
            "section": "4.11.4 Results and Justification of the Topic",
        },
    }

    # ── Process plain text formulas (deduplicated, organized by section) ──
    # Group by section for cleaner report
    section_formulas = {}  # section_path → [formula entries]

    for f in all_plain_text:
        para_idx = f["paragraph_index"]
        section = heading_paths.get(para_idx, "(No heading)")
        expr = f["expression"]

        # Determine a good description
        desc = KNOWN_FORMULAS.get(expr, f["description"])
        # Try substring match for longer expressions (longest keys first)
        if desc.startswith("Equation") or desc.startswith("Percentage"):
            for key in sorted(KNOWN_FORMULAS.keys(), key=len, reverse=True):
                if key in expr or expr in key:
                    desc = KNOWN_FORMULAS[key]
                    break

        if section not in section_formulas:
            section_formulas[section] = []
        section_formulas[section].append({
            "expression": expr,
            "format": "Plain text",
            "description": desc,
            "context": f["context_before"],
        })

    # ── Also find complete equation paragraphs ──
    # Look for paragraphs that are primarily equations
    for idx, pinfo in enumerate(paragraphs_info):
        text = pinfo["text"]
        if not text:
            continue
        section = heading_paths.get(idx, "")

        # Skip TOC entries, figure captions, table entries, references
        if pinfo["style"] in ("17", "18", "19", "10", "11", "12", "27"):
            continue
        if text.startswith("Figure") or text.startswith("Table") or text.startswith("S/N"):
            continue
        if text.startswith("[") and text.endswith("]"):
            continue  # reference
        # Skip long paragraph text (not an equation)
        if len(text) > 200:
            continue

        # Detect standalone equations: short lines containing = with math notation
        if "=" in text and not text.startswith("http") and not text.startswith("The"):
            # Check it's equation-like (contains math symbols or subscript notation)
            if re.search(r"[+\-*/^{}_()\[\]∑∏∫√]", text) or re.search(r"[A-Za-z]_[a-z]", text):
                # Skip prose that happens to contain = (descriptions, definitions)
                if re.match(r'^[A-Z][a-z]', text):
                    # Keep if it has explicit math notation like braces, subscripts, or operators
                    if not re.search(r'[{}\^]', text):
                        continue
                # Skip closing bracket / closing lines
                if text.strip().startswith(")") or text.strip().startswith("]"):
                    continue
                if re.match(r'^[\s\)\]]+\s+for\s+', text):
                    continue
                # Skip long English sentences that are not formulas
                word_count = len(text.split())
                if word_count > 8 and not any(c in text for c in '{}^_{}'):
                    continue
                # Skip prose starting with "If", "This", "These", "The", "Define"
                if re.match(r'^(If|This|These|Define)\s', text):
                    continue
                ctx_before, ctx_after = get_context(pinfo, paragraphs_info, idx)
                desc = "Equation/Formula"
                # Match longest keys first to avoid substring collisions
                for key in sorted(KNOWN_FORMULAS.keys(), key=len, reverse=True):
                    if key in text:
                        desc = KNOWN_FORMULAS[key]
                        break

                if section not in section_formulas:
                    section_formulas[section] = []
                section_formulas[section].append({
                    "expression": text.strip(),
                    "format": "Plain text (equation paragraph)",
                    "description": desc,
                    "context": ctx_before[:100],
                })

    return {
        "paragraphs_info": paragraphs_info,
        "heading_levels": heading_levels,
        "heading_paths": heading_paths,
        "omml_formulas": omml_formulas,
        "images": images,
        "plain_text": all_plain_text,
        "section_formulas": section_formulas,
        "formula_images": formula_images,
    }


# ═══════════════════════════════════════════════════════════════════════════
# 9.  Pretty-print the report
# ═══════════════════════════════════════════════════════════════════════════

def print_report(result):
    """Format and print the formula extraction report."""

    sections = result["section_formulas"]
    headings = result["heading_levels"]
    heading_paths = result["heading_paths"]
    images = result["images"]
    formula_images = result["formula_images"]

    print("\n" + "=" * 90)
    print("  FORMULA EXTRACTION REPORT")
    print("  AI & Intelligent Character Recognition for Manual Form Data Processing")
    print("=" * 90)

    # ── Part A: Document Structure ──
    print("\n" + "─" * 90)
    print("  A. DOCUMENT STRUCTURE (Section Headings)")
    print("─" * 90)

    for idx in sorted(headings.keys()):
        level, text = headings[idx]
        indent = "  " * (level - 1)
        marker = {1: "📘", 2: "📗", 3: "📙"}.get(level, "📄")
        print(f"  {marker} {indent}{text}")

    # ── Part B: Formulas Found ──
    print("\n" + "─" * 90)
    print("  B. MATHEMATICAL FORMULAS")
    print("─" * 90)

    formula_counter = 0

    # Ordered sections
    section_order = sorted(sections.keys())

    # Priority: show sections with the most important formulas first
    important_sections = [
        s for s in section_order
        if any("C_final" in f["expression"] or "sigma(z_i)" in f["expression"]
               or "C_dict" in f["expression"] or "logprob" in f["expression"]
               for f in sections[s])
    ]
    other_sections = [s for s in section_order if s not in important_sections]

    # Display important formula sections first
    displayed_sections = set()

    for section in important_sections + other_sections:
        formulas = sections[section]
        if not formulas:
            continue

        # Deduplicate formulas within this section
        seen_exprs = set()
        unique_formulas = []
        for f in formulas:
            key = f["expression"].lower().strip()
            if key not in seen_exprs:
                seen_exprs.add(key)
                unique_formulas.append(f)

        if not unique_formulas:
            continue

        displayed_sections.add(section)
        print(f"\n  📍 Section: {section}")
        print(f"  {'─' * 70}")

        for f in unique_formulas:
            formula_counter += 1
            expr = f["expression"]
            desc = f["description"]
            fmt = f.get("format", "Plain text")
            ctx = f.get("context", "")

            # Clean up expression for display
            if len(expr) > 100:
                display_expr = expr[:97] + "..."
            else:
                display_expr = expr

            print(f"\n  Formula #{formula_counter}")
            print(f"     Expression : {display_expr}")
            print(f"     Format     : {fmt}")
            print(f"     Description: {desc}")
            if ctx:
                print(f"     Context    : ...{ctx}...")

    # ── Part C: Embedded Images (potential formula diagrams) ──
    print("\n" + "─" * 90)
    print("  C. EMBEDDED IMAGES")
    print("─" * 90)

    for img in images:
        rid = img["embed_rid"]
        target = img["target"]
        para_idx = img["paragraph_index"]
        section = heading_paths.get(para_idx, "")
        img_info = formula_images.get(rid, {})

        label = img_info.get("label", target)
        is_formula = img_info.get("is_formula", False)

        print(f"\n  Image: {target}  (rId: {rid})")
        print(f"     Section   : {section}")
        print(f"     Content   : {label}")
        print(f"     Contains formula? {'Yes' if is_formula else 'No (figure/diagram/chart)'}")
        print(f"     Context   : {img['paragraph_text'][:100]}")

    # ── Part D: Summary Statistics ──
    print("\n" + "─" * 90)
    print("  D. SUMMARY")
    print("─" * 90)

    total_unique = formula_counter
    total_omml = len(result["omml_formulas"])
    total_images = len(images)

    print(f"\n  Total unique formulas extracted : {total_unique}")
    print(f"  OMML equations found            : {total_omml}")
    print(f"  Embedded images                 : {total_images}")
    print(f"  Document sections/headings      : {len(headings)}")
    print(f"  Document paragraphs             : {len(result['paragraphs_info'])}")

    print("\n  Formula Format Breakdown:")
    fmt_counts = {}
    for section in sections:
        for f in sections[section]:
            fmt = f.get("format", "Plain text")
            fmt_counts[fmt] = fmt_counts.get(fmt, 0) + 1
    for fmt, count in sorted(fmt_counts.items(), key=lambda x: -x[1]):
        print(f"     {fmt}: {count}")

    print("\n" + "=" * 90)
    print("  END OF REPORT")
    print("=" * 90)


# ═══════════════════════════════════════════════════════════════════════════
# 10.  Main
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    result = extract_formulas(DOCX_PATH)
    print_report(result)

    # Also print a plain-text version of all unique formulas for easy reference
    print("\n\n" + "=" * 90)
    print("  COMPLETE LIST OF ALL UNIQUE FORMULAS (alphabetically)")
    print("=" * 90)

    all_exprs = set()
    for section, formulas in result["section_formulas"].items():
        for f in formulas:
            expr = f["expression"].strip()
            if expr:
                all_exprs.add(expr)

    for i, expr in enumerate(sorted(all_exprs), 1):
        print(f"  {i:3d}. {expr}")
