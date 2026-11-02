'use strict';
// Presentation-only shell. Existing authenticated routes, drafts and model calls stay intact.
(() => {
  const isPreview = new URLSearchParams(location.search).get('demo') === 'ios';
  if (!isPreview) {
    for (const host of [document.querySelector('.login-form'), document.querySelector('.sidebar-bottom')]) {
      const link = document.createElement('a'); link.href = '/?demo=ios'; link.className = 'ios-preview-link';
      link.textContent = 'Open iPhone demo ↗'; host.appendChild(link);
    }
    return;
  }
  document.documentElement.classList.add('ios-preview');
  document.title = 'VinFinx — iPhone interactive demo';
  const glyph = (paths, extra='') => `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true" ${extra}>${paths}</svg>`;
  const icons = {
    overview:glyph('<path d="m3 10 9-7 9 7v10a1 1 0 0 1-1 1h-5v-7H9v7H4a1 1 0 0 1-1-1Z"/>'),
    property:glyph('<path d="M4 21V7l8-4 8 4v14M2 21h20M8 8v2m8-2v2M8 13v2m8-2v2m-6 6v-4h4v4"/>'),
    health:glyph('<path d="M3 12h4l3-7 4 14 3-7h4"/>'),
    documents:glyph('<path d="M14 3H5v18h14V8Zm0 0v5h5M8 12h8m-8 4h6"/>'),
    map:glyph('<path d="m3 6 6-3 6 3 6-3v15l-6 3-6-3-6 3Zm6-3v15m6-12v15"/>'),
    clients:glyph('<circle cx="9" cy="8" r="3"/><path d="M3 21v-3a6 6 0 0 1 12 0v3m2-16a3 3 0 0 1 0 6m1 4a5 5 0 0 1 3 5"/>'),
    appointments:glyph('<rect x="3" y="5" width="18" height="16" rx="3"/><path d="M7 3v4m10-4v4M3 11h18m-13 5h3"/>'),
    reviews:glyph('<path d="m8 12 3 3 5-6M12 3l8 3v6c0 5-8 9-8 9s-8-4-8-9V6Z"/>'),
    signals:glyph('<path d="M4 20V10m5 10V4m6 16v-7m5 7V7M2 22h20"/>'),
    sources:glyph('<path d="M3 4h7l2 2 2-2h7v15h-7l-2 2-2-2H3Zm9 2v15"/>'),
    audit:glyph('<path d="M8 4h12v17H4V4h4m0-2h8v5H8Zm0 10h8m-8 4h6"/>')
  };
  const stage = document.createElement('div'); stage.className = 'ios-stage';
  stage.innerHTML = `<div class="ios-preview-toolbar"><a class="ios-studio-brand" href="/">VinFinx <span>Mobile studio</span></a><a class="ios-exit" href="/">Web view ↗</a></div>
    <div class="ios-presentation"><aside class="ios-intro"><span class="ios-overline">A little more clarity.</span><h1>Your financial<br>world. <em>In hand.</em></h1><p>A quieter space for your home, your financial health, and the next step that feels right.</p><div class="ios-intro-points"><span>01 <b>Know your position</b></span><span>02 <b>Understand your property</b></span><span>03 <b>Move forward with evidence</b></span></div><p class="ios-prototype-note">Interactive web prototype · Not a native iOS app<br>Synthetic customer data · No bank approval</p></aside>
    <section class="ios-device" aria-label="iPhone web demo" data-text="standard"><div class="ios-side-key one"></div><div class="ios-side-key two"></div><div class="ios-side-key three"></div>
      <div class="ios-status" aria-label="Illustrative iPhone status bar"><span>9:41</span><div class="ios-island" aria-hidden="true"><i></i></div><div class="ios-status-icons" aria-hidden="true">${glyph('<path d="M3 18v-3m4 3v-6m4 6V9m4 9V6"/>')}${glyph('<path d="M3 9a14 14 0 0 1 18 0M6 12a9 9 0 0 1 12 0m-9 3a4 4 0 0 1 6 0"/><circle cx="12" cy="18" r=".8"/>')}<span class="ios-battery"><i></i></span></div></div>
      <div class="ios-viewport" id="iosViewport"></div>
      <nav class="ios-tabbar" id="iosTabbar" aria-label="iPhone navigation" hidden></nav><div class="ios-home-area" aria-hidden="true"><span></span></div>
    </section>
    <aside class="ios-settings"><span class="ios-overline">Make it yours</span><h2>A comfortable read.</h2><p>Soft corners. Clear hierarchy.<br>Type that gives every number room.</p><div class="ios-text-control" role="group" aria-label="Text size"><button type="button" data-ios-text="standard" aria-pressed="true">Aa <span>Default</span></button><button type="button" data-ios-text="large" aria-pressed="false">Aa <span>Larger</span></button></div><p class="ios-settings-hint">Use Light, Dark or System inside the app.<br>All three role workspaces are connected.</p></aside></div>
    <div class="ios-caption">iPhone preview <span>·</span> Scroll inside the screen to explore <button type="button" data-ios-text-toggle aria-label="Toggle larger text">Aa</button></div>`;
  const viewport = stage.querySelector('#iosViewport');
  const appNodes = [...document.body.children].filter(node => node.tagName !== 'SCRIPT');
  appNodes.forEach(node => viewport.appendChild(node));
  document.body.appendChild(stage);
  const device = stage.querySelector('.ios-device'), tabbar = stage.querySelector('#iosTabbar');
  document.querySelector('.sidebar .wordmark').href='/?demo=ios';
  document.querySelector('.sidebar').insertAdjacentHTML('afterbegin','<button class="ios-menu-close" type="button" data-ios-close-menu aria-label="Close navigation">×</button>');
  document.querySelector('.demo-banner').innerHTML='<span>✦</span> Synthetic demo. Not a bank appraisal or credit approval.';
  const labels = {overview:'Home',property:'Property',health:'Health',documents:'Documents',map:'Explore',clients:'Clients',appointments:'Meetings',reviews:'Reviews',signals:'Signals',sources:'Sources',audit:'Audit'};
  let previousView = '';
  function sync() {
    const signedIn = !document.querySelector('#shell').hidden;
    tabbar.hidden = !signedIn;
    const buttons = [...document.querySelectorAll('#nav [data-nav]')];
    const signature = buttons.map(button => button.dataset.nav + ':' + button.classList.contains('active')).join('|');
    if (tabbar.dataset.signature !== signature) {
      tabbar.dataset.signature = signature;
      tabbar.innerHTML = buttons.map(button => {
        const view=button.dataset.nav, active=button.classList.contains('active');
        return `<button type="button" data-nav="${view}" ${active?'aria-current="page"':''} aria-label="${button.textContent.trim()}">${icons[view]||icons.overview}<span>${labels[view]||view}</span>${active?'<i></i>':''}</button>`;
      }).join('');
    }
    const view = signedIn ? document.querySelector('#nav .active')?.dataset.nav || 'overview' : 'login';
    device.dataset.view = view;
    if (view !== previousView) {viewport.scrollTop=0;previousView=view;}
    for(const label of document.querySelectorAll('.ios-device .draft-note'))label.textContent='Your drafts stay in this tab until sign-out.';
    positionModals();
  }
  function positionModals() {
    const bounds=viewport.getBoundingClientRect();
    document.querySelectorAll('dialog:modal').forEach(dialog=>{
      Object.assign(dialog.style,{position:'fixed',inset:'auto',left:(bounds.left+12)+'px',top:(bounds.top+20)+'px',width:(bounds.width-24)+'px',maxWidth:(bounds.width-24)+'px',maxHeight:Math.max(180,bounds.height-40)+'px',margin:'0',overflowY:'auto'});
    });
  }
  new MutationObserver(sync).observe(document.querySelector('#nav'),{childList:true,subtree:true,attributes:true,attributeFilter:['class']});
  new MutationObserver(sync).observe(document.querySelector('#shell'),{attributes:true,attributeFilter:['hidden']});
  new MutationObserver(()=>{
    const content=document.querySelector('#content');
    content.querySelectorAll('input[type="number"]').forEach(input=>{input.inputMode=input.step&&input.step!=='1'?'decimal':'numeric';});
    if(!window.matchMedia('(prefers-reduced-motion: reduce)').matches&&content.animate)content.animate([{opacity:.3,transform:'translateY(7px)'},{opacity:1,transform:'translateY(0)'}],{duration:220,easing:'ease-out'});
  }).observe(document.querySelector('#content'),{childList:true});
  // Content changes can be results from asynchronous requests, not only navigation.
  new MutationObserver(()=>{positionModals();}).observe(document.body,{childList:true,subtree:true,attributes:true,attributeFilter:['open']});
  window.addEventListener('resize',positionModals);
  document.addEventListener('click',event=>{
    const textButton=event.target.closest('[data-ios-text],[data-ios-text-toggle]');
    if(textButton){device.dataset.text=textButton.dataset.iosText||(device.dataset.text==='large'?'standard':'large');stage.querySelectorAll('[data-ios-text]').forEach(button=>button.setAttribute('aria-pressed',String(button.dataset.iosText===device.dataset.text)));}
    if(event.target.closest('[data-ios-close-menu]'))document.querySelector('.sidebar').classList.remove('open');
    if(event.target.closest('[data-nav]'))viewport.scrollTop=0;
  });
  document.addEventListener('keydown',event=>{if(event.key==='Escape')document.querySelector('.sidebar').classList.remove('open');});
  // Keep tap targets accessible when a mobile keyboard changes the visible viewport.
  viewport.addEventListener('focusin',event=>{if(event.target.matches('input,textarea,select'))event.target.scrollIntoView({block:'nearest',behavior:window.matchMedia('(prefers-reduced-motion: reduce)').matches?'instant':'smooth'});});
  sync();
})();
