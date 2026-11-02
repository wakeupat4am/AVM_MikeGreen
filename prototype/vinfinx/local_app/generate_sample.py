"""Watermarked fictional payslip, English to test local PDF extraction and image OCR."""
from pathlib import Path
from reportlab.pdfgen import canvas
from reportlab.lib.colors import HexColor
from reportlab.lib.pagesizes import A4
import pypdfium2 as pdfium

ROOT=Path(__file__).resolve().parents[3];OUT=ROOT/'output/pdf';OUT.mkdir(parents=True,exist_ok=True)
path=OUT/'vinfinx-sample-payslip.pdf'
c=canvas.Canvas(str(path),pagesize=A4);w,h=A4
c.setTitle('VinFinx synthetic sample payslip - not a real salary document')
c.setFillColor(HexColor('#083f35'));c.rect(0,h-125,w,125,fill=1,stroke=0)
c.setFillColor(HexColor('#b5e98b'));c.setFont('Helvetica-Bold',12);c.drawString(44,h-42,'VINFINX SCENARIO LABS')
c.setFillColor(HexColor('#ffffff'));c.setFont('Helvetica-Bold',26);c.drawString(44,h-78,'Monthly salary statement')
c.setFont('Helvetica',10);c.drawString(44,h-102,'FICTIONAL EMPLOYER - SYNTHETIC DEMO DATA')
c.setFillColor(HexColor('#123e33'));c.setFont('Helvetica',12)
for y,text in [(h-166,'Employee Name: Nguyen Minh Long'),(h-191,'Employee ID: DEMO-001'),(h-216,'Period: 2026-10'),(h-241,'Currency: VND')]:c.drawString(44,y,text)
rows=[('Base Salary','42,000,000'),('Allowances','3,000,000'),('Bonus','5,000,000'),('Gross Pay','50,000,000'),('Insurance','3,000,000'),('Personal Income Tax','4,000,000'),('Net Pay','43,000,000')]
y=h-295
for i,(label,value) in enumerate(rows):
    c.setFillColor(HexColor('#eaf5eb') if i%2==0 else HexColor('#f6faf6'));c.rect(40,y-13,w-80,37,fill=1,stroke=0)
    c.setFillColor(HexColor('#123e33'));c.setFont('Helvetica-Bold' if label in ['Gross Pay','Net Pay'] else 'Helvetica',12)
    # Single readable label/value line makes both native extraction and image OCR reproducible.
    c.drawString(52,y,label+': '+value+' VND');y-=42
c.setFillColor(HexColor('#f4ecd8'));c.roundRect(40,125,w-80,90,10,fill=1,stroke=0)
c.setFillColor(HexColor('#6a5424'));c.setFont('Helvetica-Bold',11);c.drawString(54,190,'SYNTHETIC - TEST DATA ONLY')
c.setFont('Helvetica',10)
for i,line in enumerate(['No real employer, payment, bank account or identity is represented.',
                         'This document is not issued by Vietcombank or any other bank.',
                         'Upload it to VinFinx, inspect extraction, then request human review.']):c.drawString(54,169-i*16,line)
c.setFillColor(HexColor('#60796d'));c.setFont('Helvetica',9);c.drawString(44,62,'Demo reference: VFX-SAL-2026-10-v1');c.drawRightString(w-44,62,'Page 1 of 1')
c.showPage();c.save()
pdf=pdfium.PdfDocument(path);page=pdf[0];page.render(scale=2).to_pil().save(OUT/'vinfinx-sample-payslip.png')
print(path);print(OUT/'vinfinx-sample-payslip.png')
