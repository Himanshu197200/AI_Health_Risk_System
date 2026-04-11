from datetime import datetime


PAGE_WIDTH = 612
PAGE_HEIGHT = 792
LEFT_MARGIN = 50
TOP_START = 750
BOTTOM_MARGIN = 50
LINE_HEIGHT = 14
FONT_SIZE_BODY = 11
FONT_SIZE_TITLE = 18
FONT_SIZE_SECTION = 13
CHARS_PER_LINE = 88


def _escape_pdf_text(text):
    return text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def _wrap_text(text, width=CHARS_PER_LINE):
    words = str(text).split()
    if not words:
        return [""]

    lines = []
    current = words[0]
    for word in words[1:]:
        candidate = f"{current} {word}"
        if len(candidate) <= width:
            current = candidate
        else:
            lines.append(current)
            current = word
    lines.append(current)
    return lines


def _collect_report_lines(predictions, agent_report, patient_data):
    lines = []
    lines.append(("title", "AI Health Risk Assessment Report"))
    lines.append(("body", f"Generated on {datetime.now().strftime('%Y-%m-%d %H:%M')}"))
    lines.append(("blank", ""))

    lines.append(("section", "Patient Snapshot"))
    for key, value in patient_data.items():
        label = key.replace("_", " ").title()
        lines.append(("body", f"{label}: {value}"))
    lines.append(("blank", ""))

    lines.append(("section", "Risk Scores"))
    for disease, result in predictions.items():
        disease_label = disease.replace("_", " ").title()
        if "error" in result:
            lines.append(("body", f"{disease_label}: unavailable"))
        else:
            lines.append(
                (
                    "body",
                    f"{disease_label}: {result['risk_score']:.2f}/100 ({result['risk_category']})",
                )
            )
    lines.append(("blank", ""))

    lines.append(("section", "AI Health Report"))
    cleaned_report = (
        str(agent_report)
        .replace("#", "")
        .replace("*", "")
        .replace("`", "")
    )
    for raw_line in cleaned_report.splitlines():
        if raw_line.strip():
            lines.append(("body", raw_line.strip()))
        else:
            lines.append(("blank", ""))

    return lines


def _paginate_lines(lines):
    pages = []
    current_page = []
    y_position = TOP_START

    for line_type, text in lines:
        if line_type == "blank":
            needed_height = LINE_HEIGHT
            wrapped_lines = [""]
        else:
            wrap_width = 70 if line_type == "title" else 82 if line_type == "section" else CHARS_PER_LINE
            wrapped_lines = _wrap_text(text, width=wrap_width)
            needed_height = max(1, len(wrapped_lines)) * LINE_HEIGHT

        if y_position - needed_height < BOTTOM_MARGIN and current_page:
            pages.append(current_page)
            current_page = []
            y_position = TOP_START

        current_page.append((line_type, wrapped_lines))
        y_position -= needed_height

    if current_page:
        pages.append(current_page)

    return pages


def _render_page_stream(page_items, page_number, total_pages):
    commands = []
    y_position = TOP_START

    for line_type, wrapped_lines in page_items:
        if line_type == "blank":
            y_position -= LINE_HEIGHT
            continue

        if line_type == "title":
            font_name = "/F2"
            font_size = FONT_SIZE_TITLE
        elif line_type == "section":
            font_name = "/F2"
            font_size = FONT_SIZE_SECTION
        else:
            font_name = "/F1"
            font_size = FONT_SIZE_BODY

        for line in wrapped_lines:
            escaped = _escape_pdf_text(line)
            commands.append(f"BT {font_name} {font_size} Tf 1 0 0 1 {LEFT_MARGIN} {y_position} Tm ({escaped}) Tj ET")
            y_position -= LINE_HEIGHT

    footer = f"Page {page_number} of {total_pages}"
    commands.append(
        f"BT /F1 9 Tf 1 0 0 1 {PAGE_WIDTH / 2 - 25:.0f} 24 Tm ({_escape_pdf_text(footer)}) Tj ET"
    )
    return "\n".join(commands).encode("latin-1", errors="replace")


def _build_pdf(page_streams):
    objects = []

    def add_object(data):
        objects.append(data)
        return len(objects)

    font_regular_id = add_object(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")
    font_bold_id = add_object(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica-Bold >>")

    page_ids = []
    content_ids = []

    for stream in page_streams:
        content_id = add_object(
            b"<< /Length "
            + str(len(stream)).encode("ascii")
            + b" >>\nstream\n"
            + stream
            + b"\nendstream"
        )
        content_ids.append(content_id)
        page_ids.append(None)

    kids_placeholders = []
    for index, content_id in enumerate(content_ids):
        page_obj = (
            f"<< /Type /Page /Parent {{PAGES_ID}} 0 R /MediaBox [0 0 {PAGE_WIDTH} {PAGE_HEIGHT}] "
            f"/Resources << /Font << /F1 {font_regular_id} 0 R /F2 {font_bold_id} 0 R >> >> "
            f"/Contents {content_id} 0 R >>"
        ).encode("latin-1")
        page_ids[index] = add_object(page_obj)
        kids_placeholders.append(f"{page_ids[index]} 0 R")

    pages_id = add_object(
        (
            f"<< /Type /Pages /Count {len(page_ids)} /Kids [{' '.join(kids_placeholders)}] >>"
        ).encode("latin-1")
    )

    for index, page_id in enumerate(page_ids):
        objects[page_id - 1] = objects[page_id - 1].replace(b"{PAGES_ID}", str(pages_id).encode("ascii"))

    catalog_id = add_object(f"<< /Type /Catalog /Pages {pages_id} 0 R >>".encode("latin-1"))

    pdf = bytearray(b"%PDF-1.4\n")
    offsets = [0]

    for object_id, data in enumerate(objects, start=1):
        offsets.append(len(pdf))
        pdf.extend(f"{object_id} 0 obj\n".encode("ascii"))
        pdf.extend(data)
        pdf.extend(b"\nendobj\n")

    xref_offset = len(pdf)
    pdf.extend(f"xref\n0 {len(objects) + 1}\n".encode("ascii"))
    pdf.extend(b"0000000000 65535 f \n")
    for offset in offsets[1:]:
        pdf.extend(f"{offset:010d} 00000 n \n".encode("ascii"))

    pdf.extend(
        (
            f"trailer\n<< /Size {len(objects) + 1} /Root {catalog_id} 0 R >>\n"
            f"startxref\n{xref_offset}\n%%EOF"
        ).encode("ascii")
    )
    return bytes(pdf)


def generate_pdf_report(predictions, agent_report, patient_data):
    lines = _collect_report_lines(predictions, agent_report, patient_data)
    pages = _paginate_lines(lines)
    streams = [
        _render_page_stream(page_items, page_number=index + 1, total_pages=len(pages))
        for index, page_items in enumerate(pages)
    ]
    return _build_pdf(streams)
