'use strict';
// Presentation only: service state is fetched from authenticated backend readiness.
(() => {
  const BRAND='VinFinx x Vietcombank';
  document.title=BRAND+(new URLSearchParams(location.search).get('demo')==='ios'?' — iPhone demo':' — Financial health');
  for(const node of document.querySelectorAll('.wordmark,.mobile-brand,.ios-studio-brand')){
    const icon=node.querySelector('.brand-icon');
    const fragment=document.createDocumentFragment();if(icon)fragment.appendChild(icon);
    const lockup=document.createElement('span');lockup.className='partner-lockup';lockup.innerHTML='VinFinx <small>x Vietcombank</small>';fragment.appendChild(lockup);
    node.replaceChildren(fragment);node.setAttribute('aria-label',BRAND);
  }
  const welcome=document.querySelector('.login-form h2');welcome.textContent='Welcome to '+BRAND;
  const footer=document.querySelector('footer');if(footer.firstChild?.nodeType===Node.TEXT_NODE)footer.firstChild.textContent=BRAND+' · Internal pilot ';
  const chip=document.createElement('button');chip.type='button';chip.className='runtime-chip';chip.textContent='Check services';chip.setAttribute('aria-label','Check live model services');
  document.querySelector('.sidebar-bottom').prepend(chip);
  let modal;
  chip.addEventListener('click',async()=>{
    if(!modal){modal=document.createElement('dialog');modal.className='runtime-dialog';document.body.appendChild(modal);}
    modal.innerHTML='<div class="dialog-head"><h2>Connected services</h2><button type="button" class="icon" aria-label="Close service status">×</button></div><p class="muted">Checking authenticated backend…</p>';
    modal.querySelector('button').addEventListener('click',()=>modal.close());modal.showModal();
    try{
      const s=await api('/api/models');
      const rows=[['Live AVM',s.avm.ready?'Ready · CPU':'Unavailable'],['Research calibration',s.avm.methods?.find(m=>m.id==='hierarchical')?.ready?'Ready · Research':'Unavailable'],['Qwen generation',s.qwen.online?'Connected':'Offline · Evidence mode'],['Image OCR',s.ocr.image_ready?s.ocr.image_engine:'Unavailable'],['Native PDF extraction','Local · Human review required'],['Database',s.database_ready?'Ready · SQLite WAL':'Unavailable'],['Macro decisions','Separate research pipeline · Not live']];
      modal.querySelector('p').outerHTML='<div>'+rows.map(([name,value])=>`<div class="metric-pair"><span>${esc(name)}</span><strong>${esc(value)}</strong></div>`).join('')+'</div><p class="notice">Synthetic-data pilot. A connected model does not constitute a formal appraisal or bank approval.</p>';
    }catch(error){modal.querySelector('p').textContent=error.message;}
  });
  function decorate(){
    // The status view is shared by Mac Vision and Linux Tesseract deployments.
    if(S.status?.ocr)for(const row of document.querySelectorAll('#content .metric-pair')){
      if(row.querySelector('span')?.textContent==='Document extraction')row.querySelector('strong').textContent='Local PDF / '+S.status.ocr.image_engine;
    }
    const form=document.querySelector('#valuationForm');
    if(form&&!form.querySelector('[data-preview]')){
      const preview=document.createElement('button');preview.type='submit';preview.className='button';preview.dataset.preview='';preview.textContent='Preview live inference';form.querySelector('.form-footer').prepend(preview);
      api('/api/models').then(s=>{if(form.isConnected){const option=form.querySelector('option[value="hierarchical"]');if(option){option.disabled=!s.avm.methods?.find(m=>m.id==='hierarchical')?.ready;if(option.disabled&&form.elements.method.value==='hierarchical'){const placeholder=new Option('Choose an available model','');placeholder.disabled=true;form.elements.method.prepend(placeholder);form.elements.method.value='';form.elements.method.required=true;}}}}).catch(()=>{});
    }
    document.querySelectorAll('#content a[href^="/api/sample/"]').forEach(link=>{link.href=link.pathname+'?customer='+encodeURIComponent(S.cid);});
    document.querySelectorAll('#valuationResult .source-label,.property-value+.property-band').forEach(node=>{
      if(node.classList.contains('source-label'))node.textContent=node.textContent.replace('Actual local LightGBM inference','Live server LightGBM inference');
    });
    const result=S.summary?.property?.result;
    // Saved estimates retain their original inference metadata; never relabel an old record as a new run.
    for(const holder of document.querySelectorAll('#valuationResult')){
      if(result?.inference&&!holder.querySelector('.inference-trace')){
        const trace=document.createElement('p');trace.className='tiny inference-trace';trace.textContent='Saved computation: '+new Date(result.inference.computed_at*1000).toLocaleString('en-GB')+' · '+result.inference.elapsed_ms+' ms';holder.appendChild(trace);
      }
    }
  }
  new MutationObserver(decorate).observe(document.querySelector('#content'),{childList:true});
})();
