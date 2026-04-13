import hashlib
from datetime import datetime
from fpdf import FPDF

# Formatting Tokens (Ochre & Ash Theme)
COLOR_BG = (245, 242, 236)
COLOR_BORDER = (212, 201, 176)
COLOR_HEADER_BG = (44, 44, 44)
COLOR_HEADER_TEXT = (255, 255, 255)
COLOR_SECTION = (92, 74, 30) # #5C4A1E
COLOR_OCHRE = (212, 168, 67) # #D4A843
COLOR_ROW_ALT = (237, 232, 222) # #EDE8DE
COLOR_TEXT = (58, 58, 58) # #3A3A3A
COLOR_FOOTER = (136, 136, 136) # #888888

BADGE_COLORS = {
    "Low": (212, 237, 218),    # #D4EDDA
    "Medium": (255, 243, 205), # #FFF3CD
    "High": (248, 215, 218),   # #F8D7DA
}
BADGE_TEXT = {
    "Low": (21, 87, 36),       # #155724
    "Medium": (133, 100, 4),   # #856404
    "High": (114, 28, 36),     # #721C24
}

class ReportPDF(FPDF):
    def __init__(self, patient_data):
        super().__init__()
        self.patient_id = hashlib.md5(str(patient_data).encode()).hexdigest()[:8].upper()
        self.report_date = datetime.now().strftime('%d %b %Y')
        self.set_auto_page_break(auto=True, margin=15)
        
    def header(self):
        # 1. Page Background
        self.set_fill_color(*COLOR_BG)
        self.rect(0, 0, 210, 297, 'F')
        
        # 2. Page Border
        self.set_draw_color(*COLOR_BORDER)
        self.set_line_width(0.3)
        self.rect(5, 5, 200, 287)
        
        # 3. Header Banner
        self.set_fill_color(*COLOR_HEADER_BG)
        self.rect(5, 5, 200, 22, 'F')
        
        self.set_y(10)
        self.set_text_color(*COLOR_HEADER_TEXT)
        self.set_font('Helvetica', 'B', 18)
        self.cell(5) # padding inside banner
        self.cell(90, 12, 'AI Health Risk Assessment Report', border=0)
        
        self.set_font('Helvetica', '', 10)
        right_text = f"Patient ID: #{self.patient_id}   |   Date: {self.report_date}"
        self.cell(90, 12, right_text, border=0, align='R')
        self.ln(22)

    def footer(self):
        self.set_y(-20)
        self.set_text_color(*COLOR_FOOTER)
        self.set_font('Helvetica', 'I', 9)
        self.cell(0, 10, f'Page {self.page_no()}', align='C')
        
    def section_title(self, title):
        self.ln(6)
        self.set_fill_color(*COLOR_OCHRE)
        # Left border accent (approx 1.5mm width, 6mm height)
        self.rect(10, self.get_y()+2, 1.5, 6, 'F')
        
        self.set_x(13)
        self.set_font('Helvetica', 'B', 13)
        self.set_text_color(*COLOR_SECTION)
        # Add a space between characters to simulate letter-spacing
        spaced_title = " ".join(title.upper())
        self.cell(0, 10, spaced_title, border=0, ln=1)
        
        # Horizontal rule below heading
        self.set_draw_color(*COLOR_OCHRE)
        self.set_line_width(0.2)
        self.line(10, self.get_y(), 200, self.get_y())
        self.ln(5)

    def text_body(self, text):
        self.set_font('Helvetica', '', 10)
        self.set_text_color(*COLOR_TEXT)
        self.multi_cell(0, 6, text)
        self.ln(3)

def generate_pdf_report(predictions, agent_report, patient_data):
    pdf = ReportPDF(patient_data)
    pdf.add_page()
    
    # --- Patient Snapshot (2 column table) ---
    pdf.section_title("Patient Snapshot")
    
    items = [(k.replace("_", " ").title(), str(v)) for k, v in patient_data.items()]
    rows = [items[i:i + 2] for i in range(0, len(items), 2)]
    
    pdf.set_draw_color(*COLOR_BORDER)
    pdf.set_line_width(0.1)
    
    fill = False
    for row in rows:
        fill_color = COLOR_ROW_ALT if fill else (255, 255, 255)
        pdf.set_fill_color(*fill_color)
        pdf.set_text_color(*COLOR_TEXT)
        
        # Col 1
        pdf.set_x(10)
        pdf.set_font('Helvetica', 'B', 10)
        pdf.cell(50, 8, row[0][0], border=1, fill=True)
        pdf.set_font('Helvetica', '', 10)
        pdf.cell(45, 8, row[0][1], border=1, fill=True)
        
        # Col 2
        if len(row) > 1:
            pdf.set_font('Helvetica', 'B', 10)
            pdf.cell(50, 8, row[1][0], border=1, fill=True)
            pdf.set_font('Helvetica', '', 10)
            pdf.cell(45, 8, row[1][1], border=1, fill=True)
        else:
            pdf.cell(95, 8, "", border=1, fill=True)
        
        pdf.ln(8)
        fill = not fill
        
    # --- Risk Scores (Table) ---
    pdf.section_title("Risk Scores")
    
    # Table Header
    pdf.set_fill_color(*COLOR_OCHRE)
    pdf.set_text_color(255, 255, 255)
    pdf.set_font('Helvetica', 'B', 10)
    pdf.set_x(10)
    pdf.cell(60, 8, "Condition", border=1, fill=True)
    pdf.cell(40, 8, "Score", border=1, fill=True)
    pdf.cell(40, 8, "Out of 100", border=1, fill=True)
    pdf.cell(50, 8, "Risk Level", border=1, fill=True)
    pdf.ln(8)
    
    fill = False
    for disease, result in predictions.items():
        fill_color = COLOR_ROW_ALT if fill else (255, 255, 255)
        
        disease_label = disease.replace("_", " ").title()
        if "error" in result:
            score_str = "N/A"
            out_str = "N/A"
            risk_cat = "Unknown"
        else:
            score_str = f"{result['risk_score']:.2f}"
            out_str = "100.00"
            risk_cat = result['risk_category']
            
        pdf.set_x(10)
        pdf.set_fill_color(*fill_color)
        pdf.set_text_color(*COLOR_TEXT)
        pdf.set_font('Helvetica', '', 10)
        pdf.cell(60, 8, disease_label, border=1, fill=True)
        pdf.cell(40, 8, score_str, border=1, fill=True)
        pdf.cell(40, 8, out_str, border=1, fill=True)
        
        # Risk Badge
        bg_col = BADGE_COLORS.get(risk_cat, (230, 230, 230))
        t_col = BADGE_TEXT.get(risk_cat, (100, 100, 100))
        
        pdf.set_fill_color(*bg_col)
        pdf.set_text_color(*t_col)
        pdf.set_font('Helvetica', 'B', 9)
        pdf.cell(50, 8, risk_cat.upper(), border=1, fill=True, align='C')
        
        pdf.ln(8)
        fill = not fill
        
    # --- AI Health Report ---
    pdf.section_title("AI Health Report")
    cleaned_report = str(agent_report).replace("#", "").replace("*", "").replace("`", "")
    for paragraph in cleaned_report.split('\n'):
        if paragraph.strip():
            pdf.text_body(paragraph.strip())

    return bytes(pdf.output(dest='S').encode('latin-1'))
