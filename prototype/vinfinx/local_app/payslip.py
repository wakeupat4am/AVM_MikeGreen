"""Synthetic personal fixture, never a bank/employer-issued document."""
import datetime
from pathlib import Path

def generate(path,name,reference):
    from reportlab.pdfgen import canvas
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.colors import HexColor
    import pypdfium2
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    c=canvas.Canvas(str(path),pagesize=A4);w,h=A4
    c.setTitle('Synthetic pilot payslip - not bank issued')
    c.setFillColor(HexColor('#083f35'));c.rect(0,h-120,w,120,fill=1,stroke=0)
    c.setFillColor(HexColor('#c1ee91'));c.setFont('Helvetica-Bold',11);c.drawString(42,h-35,'VINFINX SCENARIO LABS - FICTIONAL EMPLOYER')
    c.setFillColor(HexColor('#ffffff'));c.setFont('Helvetica-Bold',25);c.drawString(42,h-73,'Synthetic monthly payslip')
    c.setFont('Helvetica',10);c.drawString(42,h-97,'SYNTHETIC - TEST DATA ONLY')
    c.setFillColor(HexColor('#153c32'));c.setFont('Helvetica',12)
    for y,text in [(h-165,'Employee Name: '+name),(h-190,'Employee ID: '+reference),(h-215,'Period: '+datetime.date.today().strftime('%Y-%m')),(h-240,'Currency: VND')]:c.drawString(42,y,text)
    rows=[('Base Salary',42000000),('Allowances',3000000),('Bonus',5000000),('Gross Pay',50000000),('Insurance',3000000),('Personal Income Tax',4000000),('Net Pay',43000000)]
    y=h-290
    for i,(label,value) in enumerate(rows):
        c.setFillColor(HexColor('#eaf4e9' if i%2==0 else '#f5f8f3'));c.roundRect(38,y-13,w-76,36,5,fill=1,stroke=0)
        c.setFillColor(HexColor('#153c32'));c.setFont('Helvetica-Bold' if label in ['Net Pay','Gross Pay'] else 'Helvetica',12)
        c.drawString(50,y,f'{label}: {value:,.0f} VND');y-=41
    c.setFillColor(HexColor('#fff1d4'));c.roundRect(38,105,w-76,105,10,fill=1,stroke=0)
    c.setFillColor(HexColor('#6b5427'));c.setFont('Helvetica-Bold',11);c.drawString(52,185,'SYNTHETIC - NO REAL SALARY OR PAYMENT')
    c.setFont('Helvetica',10)
    for i,line in enumerate(['No real employer, bank account or identity is represented.','Not issued by Vietcombank or any other bank.','For testing extraction and human review only.']):c.drawString(52,164-i*17,line)
    c.setFont('Helvetica',9);c.drawString(42,56,'Pilot reference: '+reference);c.drawRightString(w-42,56,'Page 1 of 1');c.showPage();c.save()
    with pypdfium2.PdfDocument(path) as pdf:
        page=pdf[0];bitmap=page.render(scale=2)
        try:bitmap.to_pil().save(path.with_suffix('.png'))
        finally:bitmap.close();page.close()

