from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.units import cm

# 建立 PDF
pdf_path = "/mnt/data/Studio5000_Math_Data_Instructions.pdf"
c = canvas.Canvas(pdf_path, pagesize=A4)
width, height = A4

# 標題
c.setFont("Helvetica-Bold", 18)
c.drawCentredString(width/2, height-2*cm, "Studio5000 常見數學 / 資料指令速查表")

c.setFont("Helvetica", 12)
y = height - 3.5*cm

# 內容 (逐條寫入)
content = [
    ("ADD (加法)", "Dest = SourceA + SourceB"),
    ("SUB (減法)", "Dest = SourceA - SourceB"),
    ("MUL (乘法)", "Dest = SourceA × SourceB"),
    ("DIV (除法)", "Dest = SourceA ÷ SourceB"),
    ("MOV (搬移/複製)", "Dest = Source"),
    ("CMP (比較)", "EQU 等於 | NEQ 不等於 | LES 小於 | GRT 大於 | LEQ 小於等於 | GEQ 大於等於"),
    ("CPT (運算公式)", "Dest = (運算式)，可支援 + - × ÷ () 三角函數"),
]

for title, desc in content:
    c.setFont("Helvetica-Bold", 14)
    c.drawString(2*cm, y, f"▶ {title}")
    y -= 0.8*cm
    c.setFont("Helvetica", 12)
    c.drawString(3*cm, y, desc)
    y -= 1.2*cm

# 製作完成
c.save()
pdf_path
