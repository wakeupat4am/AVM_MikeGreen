// Dependency-free unit checks; real browser interaction is verified separately.
const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const path = require('node:path');
const source = name => fs.readFileSync(path.join(__dirname, name), 'utf8');
function node() { return { textContent:'', innerHTML:'', value:'', hidden:false, disabled:false, dataset:{}, style:{},
  classList:{remove(){},add(){},toggle(){}}, children:[], setAttribute(){}, insertAdjacentHTML(){}, insertBefore(){},
  addEventListener(){}, replaceChildren(){this.innerHTML='';}, close(){}, reset(){}, querySelector(){return node();} }; }
function fixture() {
  const nodes = new Map(), events = {}, data = new Map();
  const document = { documentElement:{dataset:{}}, querySelector(selector){if(!nodes.has(selector))nodes.set(selector,node());return nodes.get(selector);},
    querySelectorAll(){return [];}, getElementById(id){return this.querySelector('#'+id);},
    createElement(){return node();}, addEventListener(name,fn){events[name]=fn;} };
  const context = vm.createContext({ document, console, setTimeout, clearTimeout, FormData,
    sessionStorage:{getItem:key=>data.get(key),setItem:(key,value)=>data.set(key,value),removeItem:key=>data.delete(key)},
    window:{addEventListener(){}}, S:{user:{email:'advisor@demo',role:'advisor'},cid:'minh',view:'property',summary:{profile:{name:'Minh'}}},
    appearance:{apply(){}}, esc:value=>String(value), appendMessage(){}, NAV:{advisor:[['property','','Property value']]},
    populateWards(){}, polishOptions(){}, filterStreets(){}, navigate(){} });
  return { context, nodes, events, data, document };
}
test('Background jobs finish without forcing a navigation', async () => {
  const f=fixture();vm.runInContext(source('assets/workspace.js'),f.context);
  let finish;f.context.promise=new Promise(resolve=>finish=resolve);
  const pending=vm.runInContext("workspace.run('Estimate','valuation',()=>promise)",f.context);
  assert.match(f.document.querySelector('#activityList').innerHTML,/In progress/);
  f.context.S.view='documents';finish(42);
  assert.equal((await pending).result,42);assert.equal(f.context.S.view,'documents');
  assert.match(f.document.querySelector('#activityList').innerHTML,/Completed/);
});
test('Sign-out invalidates late results and clears session drafts', async () => {
  const f=fixture();vm.runInContext(source('assets/workspace.js'),f.context);
  f.data.set('vinfinx.drafts','private');let finish;f.context.promise=new Promise(resolve=>finish=resolve);
  const pending=vm.runInContext("workspace.run('Chat','chat',()=>promise)",f.context);
  vm.runInContext('workspace.reset()',f.context);finish('old customer response');
  assert.equal((await pending).stale,true);assert.equal(f.data.has('vinfinx.drafts'),false);
  assert.doesNotMatch(f.document.querySelector('#activityList').innerHTML,/Completed/);
});
test('Task errors become visible and release the submit control', async () => {
  const f=fixture();vm.runInContext(source('assets/workspace.js'),f.context);
  await assert.rejects(vm.runInContext("workspace.run('Estimate','valuation',async()=>{throw Error('Unavailable')})",f.context),/Unavailable/);
  assert.match(f.document.querySelector('#activityList').innerHTML,/Needs attention/);
  assert.equal(f.document.querySelector('#valuationForm [type=submit]').disabled,false);
});
test('Draft restoration only reads the signed-in account partition', () => {
  const f=fixture();f.data.set('vinfinx.drafts',JSON.stringify([
    ['other@demo|advisor|minh|property',{id:'valuationForm',fields:{area:'999'}}],
    ['advisor@demo|advisor|minh|property',{id:'valuationForm',fields:{district:'d',ward:'w',street:'s',area:'73.5'}}]
  ]));
  const form=node();form.elements={district:{},ward:{},street:{},area:{}};f.nodes.set('#valuationForm',form);
  vm.runInContext(source('assets/workspace.js'),f.context);vm.runInContext('workspace.restore()',f.context);
  assert.equal(form.elements.area.value,'73.5');assert.notEqual(form.elements.area.value,'999');
});
function themeFixture(saved, matches=false) {
  const f=fixture(), media={matches,addEventListener(name,fn){this.change=fn;}};
  f.context.window.matchMedia=()=>media;
  f.context.localStorage={getItem:()=>saved,setItem:(key,value)=>{saved=value;}};
  vm.runInContext(source('assets/theme.js'),f.context);
  return {...f,media,saved:()=>saved};
}
test('Dark preference applies before initial render',()=>{
  const f=themeFixture('dark');assert.equal(f.document.documentElement.dataset.theme,'dark');
});
test('System preference follows live operating-system changes',()=>{
  const f=themeFixture('system');assert.equal(f.document.documentElement.dataset.theme,'light');
  f.media.matches=true;f.media.change();assert.equal(f.document.documentElement.dataset.theme,'dark');
});
test('Explicit appearance overrides system and persists',()=>{
  const f=themeFixture('dark',true);
  f.events.click({target:{closest:()=>({dataset:{themeChoice:'light'}})}});
  f.media.change();assert.equal(f.document.documentElement.dataset.theme,'light');assert.equal(f.saved(),'light');
});
test('Invalid saved appearance falls back to system',()=>{
  const f=themeFixture('invalid',true);assert.equal(f.document.documentElement.dataset.theme,'dark');
});
test('Rapid navigation ignores an older page response', async()=>{
  const f=fixture(), pending=[];
  f.context.fetch=()=>new Promise(resolve=>pending.push(resolve));
  f.context.workspace={remember(){},restore(){}};f.context.painted=[];
  vm.runInContext(source('app.js'),f.context);
  vm.runInContext("S.user={role:'customer'};render=async version=>painted.push(version)",f.context);
  const older=vm.runInContext("navigate('health')",f.context);
  const newer=vm.runInContext("navigate('property')",f.context);
  const response=id=>({ok:true,json:async()=>({id})});
  pending[2](response('new'));await newer;
  pending[1](response('old'));await older;
  assert.equal(vm.runInContext('S.summary.id',f.context),'new');
  assert.equal(f.context.painted.length,1);
  assert.equal(vm.runInContext('S.view',f.context),'property');
});
test('Display labels retain acronyms and Vietnamese place capitalization',()=>{
  const f=fixture();f.context.fetch=()=>new Promise(()=>{});
  vm.runInContext(source('app.js'),f.context);
  assert.equal(vm.runInContext("place('phường quang trung, quận đống đa')",f.context),'Phường Quang Trung, Quận Đống Đa');
  assert.equal(vm.runInContext("factorLabel('log road')",f.context),'Road width (log scale)');
  assert.equal(vm.runInContext("sentence('OCR checks')",f.context),'OCR checks');
});
