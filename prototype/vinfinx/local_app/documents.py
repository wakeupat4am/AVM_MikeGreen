"""Local extraction with page provenance. Extraction never verifies a salary by itself."""
import datetime
import re
import subprocess
import os
import shutil
import time
import platform
from pathlib import Path
from valuation import norm

FIELDS={'base':'Base Salary','allowances':'Allowances','bonus':'Bonus','gross':'Gross Pay','insurance':'Insurance','tax':'Personal Income Tax','net':'Net Pay'}

def capabilities():
    vision=Path(os.environ.get('VINFINX_VISION_BIN',str(Path(__file__).resolve().parent.parent/'.local-demo/vision-ocr')))
    tess=os.environ.get('VINFINX_TESSERACT') or shutil.which('tesseract')
    return {'native_pdf':True,'image_ready':(platform.system()=='Darwin' and vision.is_file()) or bool(tess and Path(tess).is_file()),
            'image_engine':'Apple Vision' if platform.system()=='Darwin' and vision.is_file() else 'Tesseract (local)' if tess else 'Unavailable',
            'language':os.environ.get('VINFINX_TESS_LANG','eng'),'version':'salary-extractor/2.0','local_only':True}

def tesseract_read(path):
    import pytesseract
    from PIL import Image
    pytesseract.pytesseract.tesseract_cmd=os.environ.get('VINFINX_TESSERACT') or shutil.which('tesseract') or 'tesseract'
    started=time.monotonic();spans=[];lines=[]
    def read(image,page):
        if image.width*image.height>16000000:raise ValueError('Image exceeds16 megapixels')
        remaining=40-(time.monotonic()-started)
        if remaining<=0:raise TimeoutError('OCR deadline exceeded')
        data=pytesseract.image_to_data(image,lang=os.environ.get('VINFINX_TESS_LANG','eng'),config='--psm 3',output_type=pytesseract.Output.DICT,timeout=min(20,remaining))
        grouped={}
        for i,word in enumerate(data['text'][:6000]):
            if not word.strip():continue
            line=(data['block_num'][i],data['par_num'][i],data['line_num'][i]);grouped.setdefault(line,[]).append(word)
            x,y,w,h=[data[k][i] for k in ['left','top','width','height']]
            spans.append({'text':word,'page':page,'box':[x,y,x+w,y+h],'coordinate_system':'Image pixels, top-left','confidence':max(0,float(data['conf'][i]))/100})
        lines.extend(' '.join(words) for words in grouped.values())
    if path.suffix=='.pdf':
        import pypdfium2
        with pypdfium2.PdfDocument(path) as pdf:
            if len(pdf)>5:raise ValueError('Maximum5 pages')
            for i in range(len(pdf)):
                page=pdf[i];bitmap=page.render(scale=2);image=bitmap.to_pil()
                try:read(image,i+1)
                finally:image.close();bitmap.close();page.close()
    else:
        with Image.open(path) as image:
            if image.width*image.height>16000000:raise ValueError('Image exceeds16 megapixels')
            image.load();read(image,1)
    return '\n'.join(lines),spans

def extract(path,profile):
    path=Path(path);text='';spans=[];engine='unavailable';warnings=[]
    if path.suffix=='.pdf':
        import pdfplumber
        with pdfplumber.open(path) as pdf:
            if len(pdf.pages)>5:raise ValueError('Maximum 5 pages per document')
            for page_no,page in enumerate(pdf.pages,1):
                text+=(page.extract_text() or '')+'\n'
                spans.extend({'text':w['text'],'page':page_no,'box':[w['x0'],w['top'],w['x1'],w['bottom']],'coordinate_system':'PDF points, top-left','confidence':None} for w in page.extract_words())
        if len(text.strip())>40:engine='PDF text extraction (local, not image OCR)'
    if len(text.strip())<40:
        executable=Path(os.environ.get('VINFINX_VISION_BIN',str(Path(__file__).resolve().parent.parent/'.local-demo/vision-ocr')))
        if platform.system()=='Darwin' and executable.exists():
            import json
            result=subprocess.run([str(executable),str(path)],capture_output=True,text=True,timeout=45,check=True)
            spans=json.loads(result.stdout);text='\n'.join(v['text'] for v in spans);engine='Apple Vision OCR (local, English)'
        elif capabilities()['image_ready']:
            text,spans=tesseract_read(path);engine='Tesseract OCR (local, '+os.environ.get('VINFINX_TESS_LANG','eng')+')'
        else:warnings.append('Image OCR engine unavailable. Human review required; no inferred values.')
    values={};evidence={}
    for key,label in FIELDS.items():
        matches=re.findall(re.escape(label)+r'\s*:?\s*([\d][\d,. ]*)\s*VND',text,re.I)
        nums=[float(re.sub(r'[, .]','',m)) for m in matches]
        if len(set(nums))==1:
            values[key]=nums[0]
            evidence[key]={'label':label,'text':label+': '+matches[0]+' VND',
                           'spans':[s for s in spans if re.sub(r'\D','',s['text'])==str(int(nums[0]))]}
        elif nums:warnings.append('Conflicting occurrences: '+label)
    employee=re.search(r'Employee Name\s*:\s*([^\n]+)',text,re.I)
    employee=employee.group(1).strip() if employee else None
    period=re.search(r'Period\s*:\s*(\d{4}-\d{2})',text,re.I)
    period=period.group(1) if period else None
    checks={'all_salary_fields_present':all(k in values for k in FIELDS),
            'employee_matches':bool(employee and norm(employee)==norm(profile['name'])),
            'period_present':bool(period),'explicitly_synthetic':'SYNTHETIC' in text.upper()}
    if checks['all_salary_fields_present']:
        checks['gross_arithmetic']=abs(values['base']+values['allowances']+values['bonus']-values['gross'])<1
        checks['net_arithmetic']=abs(values['gross']-values['insurance']-values['tax']-values['net'])<1
        checks['salary_range']=0<values['net']<=values['gross']<=500000000
    if period:
        try:
            date=datetime.date.fromisoformat(period+'-01');today=datetime.date.today()
            checks['recent_period']=0<=(today.year-date.year)*12+today.month-date.month<=3
        except ValueError:checks['recent_period']=False
    valid=all(checks.values()) and not warnings
    return {'engine':engine,'fields':values,'employee':employee,'period':period,'checks':checks,'warnings':warnings,
            'valid_for_demo_review':valid,'evidence':evidence,'page_spans':spans[:600],
            'version':'salary-extractor/2.0','notice':'Unverified extraction. Approval by Authenticator required before income changes.'}
