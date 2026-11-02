// Shell contract tests. Layout and native interactions are additionally checked in the browser.
const test=require('node:test'),assert=require('node:assert/strict'),fs=require('node:fs'),vm=require('node:vm'),path=require('node:path');
const source=fs.readFileSync(path.join(__dirname,'assets/ios.js'),'utf8');
function fixture(search='?demo=ios',views=['overview','property','health','documents','map']){
 const nodes=new Map(),events={},observers=[];
 function node(tagName='DIV'){
  return {tagName,children:[],dataset:{},style:{},attributes:{},hidden:false,scrollTop:100,innerHTML:'',textContent:'',
   classList:{add(){},remove(){},contains(){return false;}},
   appendChild(child){this.children.push(child);},insertAdjacentHTML(where,html){this.innerHTML+=html;},
   setAttribute(key,value){this.attributes[key]=value;},getBoundingClientRect(){return {left:18,top:100,width:357,height:627};},
   querySelector:selector=>get(selector),querySelectorAll:selector=>all(selector),addEventListener(name,fn){events['viewport:'+name]=fn;}};
 }
 function get(selector){if(!nodes.has(selector))nodes.set(selector,node());return nodes.get(selector);}
 const nav=views.map((view,index)=>({...node('BUTTON'),dataset:{nav:view},textContent:view,classList:{contains:()=>index===0}}));
 const textButtons=['standard','large'].map(value=>({...node('BUTTON'),dataset:{iosText:value}}));
 const modal=node('DIALOG');
 function all(selector){if(selector==='#nav [data-nav]')return nav;if(selector==='[data-ios-text]')return textButtons;if(selector==='dialog:modal')return [modal];if(selector==='.ios-device .draft-note')return [get('.draft-note')];return [];}
 const body=node('BODY');body.children=[node('SECTION'),node('DIV'),node('DIALOG'),node('SCRIPT')];
 const document={body,documentElement:node('HTML'),title:'',querySelector:get,querySelectorAll:all,createElement:node,addEventListener(name,fn){events[name]=fn;}};
 const context=vm.createContext({document,location:{search},URLSearchParams,console,
   MutationObserver:class{constructor(fn){this.fn=fn;observers.push(this);}observe(){}},
   window:{addEventListener(){},matchMedia:()=>({matches:true})}});
 vm.runInContext(source,context);
 return {nodes,events,observers,body,document,nav,modal,textButtons,get};
}
test('Standard web view gains links but is not wrapped or redirected',()=>{
 const f=fixture('');assert.equal(f.body.children.length,4);assert.equal(f.get('.login-form').children[0].href,'/?demo=ios');assert.equal(f.document.title,'');
});
test('iPhone shell reuses existing DOM and role-authorized navigation',()=>{
 const f=fixture('?demo=ios',['reviews','signals','sources','audit']);
 const html=f.get('#iosTabbar').innerHTML;
 assert.match(html,/data-nav="reviews"/);assert.match(html,/data-nav="audit"/);assert.doesNotMatch(html,/data-nav="property"/);
 assert.equal(f.get('#iosViewport').children.length,3);assert.equal(f.body.children.at(-1).className,'ios-stage');assert.equal(f.document.title,'VinFinx — iPhone interactive demo');
});
test('Sign-out hides the phone navigation',()=>{
 const f=fixture();f.get('#shell').hidden=true;f.observers[1].fn();assert.equal(f.get('#iosTabbar').hidden,true);assert.equal(f.get('.ios-device').dataset.view,'login');
});
test('Text-size control updates size and accessible selected state',()=>{
 const f=fixture();const button={dataset:{iosText:'large'}};
 f.events.click({target:{closest:selector=>selector.includes('data-ios-text')?button:null}});
 assert.equal(f.get('.ios-device').dataset.text,'large');assert.equal(f.textButtons[1].attributes['aria-pressed'],'true');assert.equal(f.textButtons[0].attributes['aria-pressed'],'false');
});
test('Modal bounds are derived from the phone screen, not desktop center',()=>{
 const f=fixture();assert.equal(f.modal.style.left,'30px');assert.equal(f.modal.style.top,'120px');assert.equal(f.modal.style.width,'333px');assert.equal(f.modal.style.maxHeight,'587px');
});
test('Bottom navigation resets phone scroll without changing app data',()=>{
 const f=fixture();f.events.click({target:{closest:selector=>selector==='[data-nav]'?{}:null}});assert.equal(f.get('#iosViewport').scrollTop,0);
});
