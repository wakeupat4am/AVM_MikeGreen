// Authenticator → Evidence sources (docs/PIPELINE_API_PLAN.md). Reads and writes go to the loopback pipeline API
// through the SSH tunnel; every write names the reviewer and runs the same checks as the command line.
const PIPELINE_BASE = 'http://127.0.0.1:8062/vinfinx-pipeline';
const ev = {loaded: false, tab: 'sources', reviewers: [], run: null, filter: 'all', open: null, gold: null, goldAt: 0,
            corpus: null, version: 'corpus-dev', versions: null, corpusVersion: null};  // version: the one with gold labels

function esc(s) {
  return String(s ?? '').replace(/[&<>"']/g, c => ({'&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'}[c]));
}
function shortHash(h) { return h ? esc(h.replace('sha256:', '').slice(0, 12)) + '…' : '—'; }
function pdfLink(sha, page, label) {
  if (!sha) return '';
  return `<a class="opp-action" target="_blank" rel="noopener" href="${PIPELINE_BASE}/originals/${esc(sha.replace('sha256:', ''))}${page ? '#page=' + page : ''}">${esc(label || 'Open PDF')}</a>`;
}
function reviewer() { return document.getElementById('evReviewer').value; }

async function evApi(path, body) {
  const opts = body ? {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(body)} : {};
  const r = await fetch(PIPELINE_BASE + path, {...opts, signal: AbortSignal.timeout(120000)});
  const d = await r.json().catch(() => ({}));
  if (!r.ok) { const e = new Error(d.detail || 'HTTP ' + r.status); e.problems = d.problems || []; throw e; }
  return d;
}
function showProblems(e) {
  const box = document.getElementById('evProblems');
  box.innerHTML = `<b>${esc(e.message)}</b>` + (e.problems?.length ? '<ul>' + e.problems.slice(0, 40).map(p => `<li>${esc(p)}</li>`).join('') + (e.problems.length > 40 ? `<li>… ${e.problems.length - 40} more</li>` : '') + '</ul>' : '');
  box.hidden = false;
}
function clearProblems() { document.getElementById('evProblems').hidden = true; }

// ---------------------------------------------------------------- navigation

const AUTH_VIEWS = [['authenticatorView', () => rqLoad()], ['signalsView', () => msLoad()], ['evidenceView', () => evLoad()],
                    ['auditView', () => auditLoad()]];  // the Authenticator navigation, in order (pipeline.js has the loaders)
function showAuthView(i) {
  document.querySelectorAll('.view').forEach(v => v.classList.remove('active'));
  document.getElementById(AUTH_VIEWS[i][0]).classList.add('active');
  document.querySelectorAll('#nav button').forEach((x, j) => x.classList.toggle('active', i === j));
  AUTH_VIEWS[i][1]();
}
document.querySelectorAll('#nav button').forEach((b, i) => b.addEventListener('click', () => {
  document.querySelectorAll('#nav button').forEach((x, j) => x.classList.toggle('active', i === j));
  if (document.querySelector('[data-role].active')?.dataset.role === 'authenticator') showAuthView(i);
}));
function evTab(tab) {
  ev.tab = tab;
  document.querySelectorAll('.ev-tab').forEach(b => b.classList.toggle('active', b.dataset.tab === tab));
  document.querySelectorAll('.ev-pane').forEach(p => p.hidden = p.id !== 'ev-' + tab);
  clearProblems();
  if (tab === 'gold' && !ev.gold) loadGold();
  if (tab === 'corpus' && !ev.corpus) loadCorpus();
  if (tab === 'mappings' && !ev.maps) loadMappings();
}

async function evLoad(force) {
  if (ev.loaded && !force) return;
  const status = document.getElementById('evStatus');
  try {
    const h = await evApi('/health');
    ev.reviewers = h.reviewers;
    let saved = null; try { saved = localStorage.getItem('vinfinx.reviewer'); } catch (e) {}
    document.getElementById('evReviewer').innerHTML = h.reviewers.map(r => `<option ${r === saved ? 'selected' : ''}>${esc(r)}</option>`).join('');
    status.textContent = 'Pipeline API online'; status.className = 'pill';
    ev.loaded = true;
    await loadSources();
  } catch (e) {
    status.textContent = 'Pipeline API offline · start serve-pipeline.sh and the bridge'; status.className = 'pill ev-off';
  }
}
document.getElementById('evReviewer').addEventListener('change', e => { try { localStorage.setItem('vinfinx.reviewer', e.target.value); } catch (x) {} });

// ---------------------------------------------------------------- source list

async function loadSources() {
  const runs = await evApi('/source-runs');
  if (!runs.length) { document.getElementById('evRun').innerHTML = '<p class="tiny">No source-agent run yet.</p>'; return; }
  const run = runs[runs.length - 1];
  ev.run = {...await evApi('/source-runs/' + encodeURIComponent(run.run_id)), summary: run};
  renderSources();
}

function decisionOf(r) { return r.progress?.decision || null; }
function overrules(r, d) { return (d === 'accept' && r.decision === 'rejected') || (d === 'reject' && r.decision === 'accepted'); }

function renderSources() {
  const run = ev.run, s = run.summary, recs = run.records;
  const decided = recs.filter(decisionOf).length, signed = run.signed[0];
  const pct = Math.round(100 * decided / recs.length);
  document.getElementById('evRun').innerHTML = `
    <div class="card-title"><div><h2>Source list · run ${esc(run.run_id)}</h2>
      <span class="tiny">Proposed by ${esc(s.agent_version)} on ${esc(s.model)} (${esc(s.prompt_version)}) · draft hash <span class="ev-hash" title="${esc(run.list_hash)}">${shortHash(run.list_hash)}</span></span></div>
      ${signed ? `<span class="pill">Signed ${esc(signed.version)}</span>` : '<span class="pill ev-amber">Awaiting sign-off</span>'}</div>
    <div class="ev-stats">
      <div><small>Agent accepted</small><b>${s.decisions.accepted}</b></div><div><small>Unclear</small><b>${s.decisions.unclear}</b></div>
      <div><small>Agent rejected</small><b>${s.decisions.rejected}</b></div><div><small>You decided</small><b>${decided} / ${recs.length}</b></div></div>
    <div class="ev-bar"><i style="width:${pct}%"></i></div>
    ${signed ? `<p class="ev-signed">Signed by <b>${esc(signed.signed_off_by)}</b> on ${esc(new Date(signed.signed_off_at).toLocaleString())} · list hash <span class="ev-hash" title="${esc(signed.list_hash)}">${shortHash(signed.list_hash)}</span>. A signed list is read-only; ingestion uses only your decisions.</p>` : `
    <div class="ev-actions">
      <button class="secondary" onclick="bulkDecide('accepted','accept')">Accept all the agent accepted (${s.decisions.accepted})</button>
      <button class="secondary" onclick="bulkDecide('rejected','reject')">Reject all the agent rejected (${s.decisions.rejected})</button>
      <a class="opp-action" href="${PIPELINE_BASE}/source-runs/${encodeURIComponent(run.run_id)}/review.csv" download>Download review.csv</a>
      <span class="ev-sign"><input id="evVersion" value="source_list/v0.1" aria-label="Version to sign"><button class="primary" ${decided < recs.length ? 'disabled' : ''} onclick="signList()">Sign list</button></span>
    </div>`}`;
  const counts = {all: recs.length, undecided: recs.length - decided, accepted: s.decisions.accepted, unclear: s.decisions.unclear, rejected: s.decisions.rejected};
  document.getElementById('evFilters').innerHTML = Object.entries(counts).map(([k, n]) =>
    `<button class="ev-chip ${ev.filter === k ? 'active' : ''}" onclick="ev.filter='${k}';renderSources()">${k[0].toUpperCase() + k.slice(1)} · ${n}</button>`).join('');
  const shown = recs.filter(r => ev.filter === 'all' || (ev.filter === 'undecided' ? !decisionOf(r) : r.decision === ev.filter));
  document.getElementById('evRows').innerHTML = shown.map(r => sourceRow(r, !!signed)).join('') || '<p class="tiny">Nothing here.</p>';
}

function sourceRow(r, locked) {
  const d = decisionOf(r), open = ev.open === r.source_id;
  const agentPill = {accepted: '<span class="pill">accepted</span>', unclear: '<span class="risk medium">unclear</span>', rejected: '<span class="risk high">rejected</span>'}[r.decision] || esc(r.decision);
  const mine = d ? `<span class="ev-mine ${d}">${d === 'accept' ? '✓ accept' : '✕ reject'}</span>` : '<span class="tiny">undecided</span>';
  return `<div class="ev-row ${open ? 'open' : ''}">
    <div class="ev-row-head" onclick="ev.open=ev.open==='${r.source_id}'?null:'${r.source_id}';renderSources()">
      <span class="ev-id">${esc(r.source_id)}</span>
      <div class="ev-title"><b>${esc(r.title)}</b><span class="tiny">${esc(r.publisher || 'unknown publisher')} · ${esc(r.publication_date || 'no date')} · ${esc(r.language || '?')}${r.failed_criterion ? ' · failed ' + esc(r.failed_criterion) : ''}</span></div>
      ${agentPill}${mine}</div>
    ${open ? sourceDetail(r, locked) : ''}</div>`;
}

function sourceDetail(r, locked) {
  const sc = r.screening, d = decisionOf(r);
  const checks = r.checks.map(c => `<li class="ev-check ${c.outcome}"><b>${esc(c.criterion)}</b> ${esc(c.outcome)} <span class="tiny">(${esc(c.by)})</span> · ${esc(c.detail)}</li>`).join('');
  const quotes = sc ? sc.quotes.map(q => `<blockquote class="ev-quote">${esc(q.quote)}<small>verified word for word in the screened text</small></blockquote>`).join('') : '';
  const unverified = sc?.unverified_quotes?.length ? `<p class="tiny">${sc.unverified_quotes.length} quote(s) the model gave were not found in the text and are never evidence.</p>` : '';
  return `<div class="ev-detail">
    <p><b>Agent's decision:</b> ${esc(r.decision)} · ${esc(r.decision_reason)}</p>
    ${sc ? `<p><b>Why it qualifies (model):</b> ${esc(sc.qualifies_because)}<br><span class="tiny">metrics ${esc(sc.metrics.join(', ') || '—')} · targets ${esc(sc.targets.join(', ') || '—')} · geography ${esc(sc.geography)}</span></p>` : ''}
    ${quotes}${unverified}
    <ul class="ev-checks">${checks}</ul>
    <p><a class="opp-action" target="_blank" rel="noopener" href="${esc(r.url)}">Publisher page</a> ${r.original_stored ? pdfLink(r.sha256, null, 'Stored original (PDF)') : '<span class="tiny">No stored original: it cannot be accepted</span>'}</p>
    ${locked ? `<p class="tiny">Signed: ${esc(r.review?.decision || d)}${r.review?.note ? ' · ' + esc(r.review.note) : ''}</p>` : `
    <div class="ev-decide">
      <textarea id="evNote" maxlength="500" placeholder="Note (required when you overrule the agent)">${esc(r.progress?.note || '')}</textarea>
      <button class="secondary ${d === 'reject' ? 'chosen' : ''}" onclick="decide('${r.source_id}','reject')">Reject</button>
      <button class="primary ${d === 'accept' ? 'chosen' : ''}" ${r.original_stored ? '' : 'disabled'} onclick="decide('${r.source_id}','accept')">Accept</button>
      ${d ? `<button class="opp-action" onclick="decide('${r.source_id}',null)">Clear</button>` : ''}
    </div>`}</div>`;
}

async function saveSourceRows(rows) {
  clearProblems();
  try {
    await evApi(`/source-runs/${encodeURIComponent(ev.run.run_id)}/progress`, {reviewer: reviewer(), rows});
    const at = new Date().toISOString();
    rows.forEach(row => { const r = ev.run.records.find(x => x.source_id === row.source_id); r.progress = row.decision ? {decision: row.decision, note: row.note || null, reviewer: reviewer(), at} : null; });
    renderSources();
    return true;
  } catch (e) { showProblems(e); await loadSources(); return false; }
}

async function decide(id, decision) {
  const r = ev.run.records.find(x => x.source_id === id), note = document.getElementById('evNote')?.value.trim() || null;
  if (decision && overrules(r, decision) && !note) { showToast('Add a note: you are overruling the agent'); document.getElementById('evNote').focus(); return; }
  if (await saveSourceRows([{source_id: id, decision, note}])) showToast(`${id}: ${decision || 'cleared'} · saved`);
}

async function bulkDecide(agentDecision, decision) {
  const rows = ev.run.records.filter(r => r.decision === agentDecision && !decisionOf(r)).map(r => ({source_id: r.source_id, decision, note: null}));
  if (!rows.length) { showToast('Nothing left to decide there'); return; }
  if (!confirm(`${decision === 'accept' ? 'Accept' : 'Reject'} ${rows.length} sources the agent ${agentDecision}, as ${reviewer()}?`)) return;
  if (await saveSourceRows(rows)) showToast(`${rows.length} decisions saved`);
}

async function signList() {
  const version = document.getElementById('evVersion').value.trim(), recs = ev.run.records;
  const acc = recs.filter(r => decisionOf(r) === 'accept').length;
  if (!confirm(`Sign ${version} as ${reviewer()}?\n\n${acc} accepted, ${recs.length - acc} rejected.\nA signed version is never changed or overwritten.`)) return;
  clearProblems();
  try {
    const out = await evApi(`/source-runs/${encodeURIComponent(ev.run.run_id)}/sign`, {reviewer: reviewer(), version});
    showToast(`Signed ${out.version} · ${out.accepted} sources`);
    await loadSources();
  } catch (e) { showProblems(e); await loadSources(); }  // show what the server holds, not what the page assumed
}

// ---------------------------------------------------------------- gold labels

async function loadGold() {
  try { ev.gold = await evApi(`/corpus/${ev.version}/gold`); }
  catch (e) { showProblems(e); return; }
  const first = ev.gold.rows.findIndex(r => !r.progress);
  ev.goldAt = first < 0 ? 0 : first;
  renderGold();
}

function highlight(passage, quote) {
  if (!quote) return esc(passage);
  const words = quote.split(/\s+/).filter(Boolean).map(w => w.replace(/[.*+?^${}()|[\]\\]/g, '\\$&'));
  const m = words.length ? new RegExp(words.join('\\s+')).exec(passage) : null;
  if (!m) return esc(passage);
  return esc(passage.slice(0, m.index)) + '<mark>' + esc(m[0]) + '</mark>' + esc(passage.slice(m.index + m[0].length));
}

function renderGold() {
  const g = ev.gold, rows = g.rows, r = rows[ev.goldAt], done = g.status === 'reviewed';
  const decided = rows.filter(x => x.progress).length;
  document.getElementById('evGoldHead').innerHTML = `
    <div class="card-title"><div><h2>Gold labels · ${esc(g.corpus_version)}</h2>
      <span class="tiny">Drafted by ${esc(g.meta.model)} (${esc(g.meta.prompt_version)}, $${esc(g.meta.cost_usd)}) · a seeded 20% sample for you to check</span></div>
      ${done ? '<span class="pill">Reviewed</span>' : '<span class="pill ev-amber">Draft</span>'}</div>
    <div class="ev-stats"><div><small>Rows to review</small><b>${rows.length}</b></div><div><small>You labelled</small><b>${decided}</b></div>
      ${done ? `<div><small>Agreement with the model</small><b>${Math.round(g.review.agreement * 100)}%</b></div><div><small>Reviewed by</small><b>${esc(g.review.reviewer)}</b></div>` : ''}</div>
    <div class="ev-bar"><i style="width:${Math.round(100 * decided / rows.length)}%"></i></div>
    ${done ? '' : `<div class="ev-actions"><a class="opp-action" href="${PIPELINE_BASE}/corpus/${ev.version}/gold/review.csv" download>Download review.csv</a>
      <button class="primary" ${decided < rows.length ? 'disabled' : ''} onclick="submitGold()">Submit review</button></div>`}`;
  if (!r) { document.getElementById('evGoldCard').innerHTML = ''; return; }
  const mine = r.progress?.label;
  document.getElementById('evGoldCard').innerHTML = `
    <div class="card-title"><span class="tiny">Row ${ev.goldAt + 1} of ${rows.length} · ${esc(r.question_id)}</span>
      <span><button class="opp-action" onclick="goldMove(-1)">← Previous</button><button class="opp-action" onclick="goldMove(1)">Next →</button><button class="opp-action" onclick="goldNextOpen()">Next unlabelled</button></span></div>
    <h3 class="ev-question">${esc(r.question)}</h3>
    <p class="tiny">Model's label: <b>${esc(r.label)}</b>${r.label === 'supports' ? (r.quote_verified ? ' · quote verified word for word (highlighted)' : ' · quote NOT found in the passage') : ''}</p>
    <div class="ev-passage">${highlight(r.passage, r.quote_verified ? r.quote : '')}</div>
    <p class="tiny">${esc(r.title || r.source_id)} · page ${esc(r.page)} · ${esc(r.chunk_id)} ${pdfLink(r.original_sha256, r.page, 'Open the page')}</p>
    ${done ? `<p class="tiny">Your label: <b>${esc(mine || '—')}</b></p>` : `
    <div class="ev-decide">
      <input id="evGoldNote" maxlength="500" placeholder="Note (optional)" value="${esc(r.progress?.note || '')}">
      ${['supports', 'related', 'irrelevant'].map((l, i) => `<button class="${mine === l ? 'primary' : 'secondary'}" onclick="goldLabel('${l}')">${i + 1} · ${l}</button>`).join('')}
    </div>
    <p class="tiny">Supports: the passage states how the metric affects the target. Related: on topic, no such effect. Keys 1–3 label, ← → move.</p>`}`;
}

function goldMove(step) { ev.goldAt = Math.min(ev.gold.rows.length - 1, Math.max(0, ev.goldAt + step)); renderGold(); }
function goldNextOpen() {
  const rows = ev.gold.rows, n = rows.length;
  for (let k = 1; k <= n; k++) { const i = (ev.goldAt + k) % n; if (!rows[i].progress) { ev.goldAt = i; break; } }
  renderGold();
}

async function goldLabel(label) {
  const r = ev.gold.rows[ev.goldAt], note = document.getElementById('evGoldNote')?.value.trim() || null;
  clearProblems();
  try {
    await evApi(`/corpus/${ev.version}/gold/progress`, {reviewer: reviewer(), rows: [{question_id: r.question_id, chunk_id: r.chunk_id, label, note}]});
    r.progress = {label, note, reviewer: reviewer()};
    goldNextOpen();
  } catch (e) { showProblems(e); ev.gold = null; await loadGold(); }
}

async function submitGold() {
  if (!confirm(`Submit the gold review as ${reviewer()}? The reviewed set is frozen and retrieval is re-measured on it.`)) return;
  clearProblems();
  try {
    const out = await evApi(`/corpus/${ev.version}/gold/submit`, {reviewer: reviewer()});
    showToast(`Gold set frozen · agreement ${Math.round(out.review.agreement * 100)}%`);
    ev.gold = null; ev.corpus = null; ev.versions = null;
    await loadGold();
  } catch (e) { showProblems(e); ev.gold = null; await loadGold(); }
}

document.addEventListener('keydown', e => {
  if (ev.tab !== 'gold' || !ev.gold || !document.getElementById('evidenceView').classList.contains('active')) return;
  if (['INPUT', 'TEXTAREA', 'SELECT'].includes(e.target.tagName) || ev.gold.status === 'reviewed') return;
  const l = {'1': 'supports', '2': 'related', '3': 'irrelevant'}[e.key];
  if (l) goldLabel(l);
  if (e.key === 'ArrowRight') goldMove(1);
  if (e.key === 'ArrowLeft') goldMove(-1);
});

// ---------------------------------------------------------------- corpus and search

async function loadCorpus() {
  try {
    if (!ev.versions) {
      ev.versions = await evApi('/corpus');
      const frozen = ev.versions.filter(v => v.frozen);
      ev.corpusVersion = (frozen[frozen.length - 1] || ev.versions[ev.versions.length - 1]).corpus_version;
    }
    ev.corpus = await evApi(`/corpus/${ev.corpusVersion}`);
  }
  catch (e) { showProblems(e); return; }
  const c = ev.corpus, m = c.manifest, ix = c.index, mt = c.metrics;
  const chunks = m.documents.reduce((a, d) => a + d.chunks, 0);
  document.getElementById('evCorpusHead').innerHTML = `
    <div class="card-title"><div><h2>Corpus · <select aria-label="Corpus version" onchange="ev.corpusVersion=this.value;document.getElementById('evHits').innerHTML='';loadCorpus()">${ev.versions.map(v => `<option ${v.corpus_version === ev.corpusVersion ? 'selected' : ''}>${esc(v.corpus_version)}</option>`).join('')}</select></h2><span class="tiny">From ${esc(m.source_list_version)} · built ${esc(new Date(m.built_at).toLocaleString())}</span></div>
      ${m.frozen ? '<span class="pill">Frozen</span>' : '<span class="pill ev-amber">Development build · never cited</span>'}</div>
    <div class="ev-stats"><div><small>Documents</small><b>${m.documents.length}</b></div><div><small>Passages</small><b>${chunks}</b></div>
      <div><small>Embedding model</small><b class="ev-small">${esc(ix?.embedding_model || '—')}</b></div><div><small>Keyword index</small><b class="ev-small">SQLite FTS5</b></div></div>
    <p class="tiny">Corpus manifest <span class="ev-hash" title="${esc(c.manifest_hash)}">${shortHash(c.manifest_hash)}</span> · index manifest <span class="ev-hash" title="${esc(c.index_manifest_hash)}">${shortHash(c.index_manifest_hash)}</span> · ${esc(m.extractor)} · ${esc(m.chunker)}</p>
    ${mt ? `<h3>Retrieval quality <span class="tiny">(labels: ${esc(mt.labels_status)}; ${mt.questions_with_gold} of ${mt.questions} questions have a gold passage)</span></h3>
    <div class="table-wrap"><table class="client-table"><thead><tr><th>Search</th><th>Recall@5</th><th>Recall@8</th><th>MRR</th><th>Recall@8 EN / VI</th></tr></thead><tbody>
      ${['hybrid', 'vector', 'keyword'].map(k => `<tr><td><b>${k}</b></td><td>${mt[k]['recall@5']}</td><td>${mt[k]['recall@8']}</td><td>${mt[k].mrr}</td><td>${mt[k]['recall@8_by_language'].en} / ${mt[k]['recall@8_by_language'].vi}</td></tr>`).join('')}
    </tbody></table></div>` : ''}`;
  document.getElementById('evDocs').innerHTML = `<div class="table-wrap"><table class="client-table"><thead><tr><th>Source</th><th>Document</th><th>Lang · geography</th><th>Pages (OCR)</th><th>Passages</th><th></th></tr></thead><tbody>
    ${m.documents.map(d => `<tr><td>${esc(d.source_id)}</td><td><b>${esc(d.title)}</b><div class="tiny">${esc(d.publisher)} · ${esc(d.publication_date)}</div></td><td>${esc(d.language)} · ${esc(d.geography)}</td><td>${d.pages_indexed} (${d.pages_ocr})</td><td>${d.chunks}</td><td>${pdfLink(d.original_sha256, null, 'PDF')}</td></tr>`).join('')}
  </tbody></table></div>`;
}

async function evSearch(e) {
  e.preventDefault();
  const q = document.getElementById('evQuery').value.trim(), mode = document.getElementById('evMode').value;
  if (!q) return;
  const out = document.getElementById('evHits');
  out.innerHTML = '<p class="tiny">Searching…</p>';
  clearProblems();
  try {
    const d = await evApi(`/corpus/${ev.corpusVersion}/search?q=${encodeURIComponent(q)}&mode=${mode}`);
    out.innerHTML = d.hits.map((h, i) => `<div class="ev-hit"><div class="card-title"><b>${i + 1}. ${esc(h.title)}</b>${pdfLink(h.original_sha256, h.page, 'Page ' + h.page)}</div>
      <div class="ev-passage">${esc(h.text)}</div>
      <p class="tiny">${esc(h.chunk_id)} · ${esc(h.publisher)} · ${esc(h.publication_date)} · ${esc(h.language)} · keyword rank ${h.keyword_rank ?? '—'} · meaning rank ${h.vector_rank ?? '—'}</p></div>`).join('') || '<p class="tiny">No passages found.</p>';
  } catch (err) { out.innerHTML = ''; showProblems(err); }
}

// ---------------------------------------------------------------- mappings (docs/MAPPING_AGENT_PLAN.md)

const OUTCOME = {accepted: ['pill', 'accepted'], no_mapping: ['pill ev-amber', 'no mapping'],
                 rejected_by_rules: ['pill ev-off', 'rejected: rules'], rejected_by_support: ['pill ev-off', 'rejected: support check'],
                 error: ['pill ev-off', 'error']};

async function loadMappings() {
  try { ev.maps = await evApi('/mappings'); } catch (e) { showProblems(e); return; }
  const m = ev.maps, run = m.runs[0];
  const rr = m.runner;
  document.getElementById('evMapHead').innerHTML = `
    <div class="card-title"><div><h2>Mappings</h2><span class="tiny">What each macro signal means for a target, proposed by Qwen from the frozen corpus, quotes checked word for word and by a second model (support check)</span></div></div>
    <p class="${rr.source === 'registry' ? 'ev-signed' : 'ev-problems'}" style="margin:0 0 14px">The runner uses <b>${esc(rr.mapping_id)}</b> (${esc(rr.source)}, ${esc(rr.status)}): ${esc(rr.reason)}.${rr.source === 'provisional' ? ' The provisional mapping is hand-built and synthetic; it stays until a registry set covers every pair the policy weights.' : ''}</p>
    ${m.sets.length ? `<div class="table-wrap"><table class="client-table"><thead><tr><th>Set</th><th>Corpus</th><th>Entries</th><th>Status</th><th></th></tr></thead><tbody>
      ${m.sets.map(s => `<tr><td><b>${esc(s.mapping_id)}</b><div class="tiny">${esc(new Date(s.created_at).toLocaleString())}</div></td><td>${esc(s.corpus_version)}</td><td>${s.entries}</td>
        <td>${esc(s.status)}${s.reviewed_by ? ' by ' + esc(s.reviewed_by) : ''}</td>
        <td>${s.status === 'unvalidated' ? `<button class="secondary" onclick="reviewMapping(${s.version},'reject')">Reject</button> <button class="primary" onclick="reviewMapping(${s.version},'validate')">Validate</button>` : ''}</td></tr>`).join('')}
    </tbody></table></div>` : '<p class="tiny">No mapping set yet: no proposal has been accepted.</p>'}
    ${run ? `<p class="tiny">Latest run ${esc(run.run_id)} on ${esc(run.corpus_version)}: ${Object.entries(run.counts).filter(([, n]) => n).map(([k, n]) => n + ' ' + k.replaceAll('_', ' ')).join(', ')} · support check $${esc(run.support_cost_usd)}</p>` : ''}`;
  document.getElementById('evMapPairs').innerHTML = run ? '<p class="tiny">Loading the run…</p>' : '<p class="tiny">The mapping agent has not run yet.</p>';
  if (run) renderMapRun(await evApi('/mappings/runs/' + encodeURIComponent(run.run_id)));
}

function renderMapRun(r) {
  document.getElementById('evMapPairs').innerHTML = `<div class="card-title"><h2>Pairs in run ${esc(r.run_id)}</h2><span class="tiny">${esc(r.agent)} · ${esc(r.prompt_version)} · support ${esc(r.support_model)}</span></div>` +
    r.pairs.map(p => {
      const [cls, label] = OUTCOME[p.outcome] || ['pill', p.outcome];
      const prop = p.proposal, sup = p.support;
      const quotes = (p.verified_quotes || []).map(q => {
        const v = sup ? sup.quotes.find(x => x.chunk_id === q.chunk_id) : null;
        return `<blockquote class="ev-quote">${esc(q.quote)}<small>${esc(q.publisher)}, ${esc(q.publication_date)} · page ${q.page} ${pdfLink(q.document_hash, q.page, 'Open the page')}${v ? ` · support check: <b>${esc(v.verdict)}</b>, ${esc(v.rationale)}` : ''}</small></blockquote>`;
      }).join('');
      const why = p.outcome === 'no_mapping' ? `<p class="tiny">Agent: ${esc(p.reason)}</p>`
        : p.outcome === 'rejected_by_rules' ? `<p class="tiny">Rules: ${esc((p.rules.findings || []).map(f => f.code + ' ' + (f.detail || '')).join('; '))}</p>`
        : p.outcome === 'error' ? `<p class="tiny">${esc(p.error)}</p>` : '';
      return `<div class="ev-row"><div class="ev-row-head" style="cursor:default"><span class="ev-id">${esc(p.metric)}</span>
          <div class="ev-title"><b>${esc(p.metric.replaceAll('_', ' '))} → ${esc(p.target.replaceAll('_', ' '))}</b>
          <span class="tiny">${prop ? `direction ${esc(prop.direction)} · lag ${prop.lag_months.join('–')} months · size ${esc(prop.size_range)} · ` : ''}${p.retrieval.chosen.length} passages retrieved · ${p.seconds} s</span></div>
          <span class="${cls}">${label}</span></div><div class="ev-detail">${why}${quotes}</div></div>`;
    }).join('');
}

async function reviewMapping(version, action) {
  let note = null;
  if (action === 'reject') { note = prompt('Why reject this mapping set? (required)'); if (!note) return; }
  else if (!confirm('Validate this mapping set? Validated mappings allow the policy\'s full set of actions.')) return;
  clearProblems();
  try { await evApi(`/mappings/sets/${version}/review`, {reviewer: reviewer(), action, note}); showToast('Mapping set ' + action + 'd'); ev.maps = null; await loadMappings(); }
  catch (e) { showProblems(e); }
}
