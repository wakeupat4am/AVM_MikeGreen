// Authenticator → Market signals, Review queue, Audit log, and the Customer view's property card
// (docs/RUNNER_PLAN.md). Uses esc, shortHash, evApi, showProblems and PIPELINE_BASE from evidence.js.
const pl = {doc: null, runs: [], open: null, poll: null, queueOpen: null};
const STAGE_NAMES = {document: 'Bulletin', ocr: 'OCR', extraction: 'Evidence', avm: 'AVM band', policy: 'Policy',
                     bundle: 'Signed bundle', decision: 'Qwen', verifier: 'Verifier'};
const VERDICT = {PASS: ['pill', 'Passed'], ROUTE_TO_HUMAN: ['pill ev-amber', 'Routed to a person'], BLOCK: ['pill ev-off', 'Blocked']};

async function plReviewer() {
  if (!ev.reviewers.length) ev.reviewers = (await evApi('/health')).reviewers;
  return document.getElementById('evReviewer').value || ev.reviewers[0];
}
function verdictPill(r) {
  if (r.status === 'running') return '<span class="pill ev-amber">Running…</span>';
  if (r.status === 'failed') return '<span class="pill ev-off">Stopped</span>';
  const [cls, label] = VERDICT[r.verdict] || ['pill', r.verdict || '—'];
  return `<span class="${cls}">${label}</span>`;
}
function stage(r, name) { return r.stages.find(s => s.name === name) || {}; }
function when(iso) { return iso ? new Date(iso).toLocaleString() : '—'; }
function metricLabel(id) { const p = id.split('.'); return `${p[2].replaceAll('_', ' ')} <span class="tiny">${esc(p[1])}</span>`; }
function plStatus(id, ok, text) { const s = document.getElementById(id); s.textContent = text; s.className = ok ? 'pill' : 'pill ev-off'; }

// ---------------------------------------------------------------- Market signals

async function msLoad() {
  try {
    const inp = await evApi('/pipeline/inputs');
    plStatus('msStatus', true, 'Pipeline API online · set ' + inp.set);
    document.getElementById('msDoc').innerHTML = inp.documents.map(d =>
      `<option value="${esc(d.document_id)}">${esc(d.variant.replaceAll('_', ' '))} · ${esc(d.language)} · ${esc(d.document_id)}</option>`).join('');
    const clean = inp.documents.find(d => d.variant === 'clean' && d.language === 'en');
    if (clean) document.getElementById('msDoc').value = clean.document_id;
    document.getElementById('msCustomer').innerHTML = inp.customers.map(c =>
      `<option value="${esc(c.customer_id)}">${esc(c.display_name)} · ${esc(c.district)}${c.synthetic ? ' · synthetic' : ''}</option>`).join('');
    await msRefresh();
  } catch (e) { plStatus('msStatus', false, 'Pipeline API offline'); }
}

async function msRefresh() {
  pl.runs = await evApi('/pipeline/runs');
  document.getElementById('msList').innerHTML = pl.runs.map(r => `
    <div class="queue-item ${pl.open === r.run_id ? 'active' : ''}" onclick="msOpen('${esc(r.run_id)}')">
      <div class="queue-top">${verdictPill(r)}<span class="tiny">${esc(when(r.started_at))}</span></div>
      <h3>${esc((r.variant || '').replaceAll('_', ' '))}</h3><p>${esc(r.document_id || '')}${r.trusted != null ? ` · ${r.trusted} trusted, ${r.untrusted} untrusted` : ''}</p></div>`).join('')
    || '<p class="tiny" style="padding:10px">No runs yet.</p>';
  if (!pl.open && pl.runs.length) msOpen(pl.runs[0].run_id);
}

async function msRun(e) {
  e.preventDefault();
  const btn = document.getElementById('msRunBtn');
  btn.disabled = true;
  try {
    const out = await evApi('/pipeline/runs', {reviewer: await plReviewer(), document_id: document.getElementById('msDoc').value,
                                               customer_id: document.getElementById('msCustomer').value});
    showToast('Run started');
    pl.open = out.run_id;
    await msRefresh();
    msOpen(out.run_id);
  } catch (err) { showToast(err.message); btn.disabled = false; }
}

async function msOpen(runId) {
  if (pl.open !== runId) pl.page = 1;
  pl.open = runId;
  clearTimeout(pl.poll);
  document.querySelectorAll('#msList .queue-item').forEach(x => x.classList.toggle('active', x.getAttribute('onclick').includes(runId)));
  let r;
  try { r = await evApi('/pipeline/runs/' + encodeURIComponent(runId)); } catch (e) { return; }
  if (pl.open !== runId) return;
  document.getElementById('msDetail').innerHTML = runDetail(r);
  if (r.status === 'running') pl.poll = setTimeout(() => msOpen(runId), 1500);
  else { document.getElementById('msRunBtn').disabled = false; const i = pl.runs.findIndex(x => x.run_id === runId); if (i >= 0 && pl.runs[i].status === 'running') msRefresh(); }
}

function stageStrip(r) {
  return '<div class="pl-stages">' + Object.keys(STAGE_NAMES).map(n => {
    const s = stage(r, n), st = s.status || 'waiting';
    return `<div class="pl-stage ${st}"><i>${st === 'ok' ? '✓' : st === 'error' ? '✕' : st === 'running' ? '…' : '·'}</i><b>${STAGE_NAMES[n]}</b><span>${s.seconds != null ? s.seconds + ' s' : ''}</span></div>`;
  }).join('') + '</div>';
}

function pageViewer(r, page) {
  const doc = stage(r, 'document'), ex = stage(r, 'extraction');
  if (!doc.pages) return '';
  const pages = Object.keys(doc.pages).map(p => +p.match(/\d+/)[0]);
  const boxes = (ex.items || []).map(item => {
    const f = item.provenance.fields || {}, v = f.value;
    if (!v || !v.bbox || v.page !== page) return '';
    const bb = Object.values(f).filter(x => x.bbox && x.page === page && Math.abs(x.bbox.y0 - v.bbox.y0) < 0.02).map(x => x.bbox);
    const x0 = Math.min(...bb.map(b => b.x0)), y0 = Math.min(...bb.map(b => b.y0)), x1 = Math.max(...bb.map(b => b.x1)), y1 = Math.max(...bb.map(b => b.y1));
    const tip = `${item.metric_id} = ${item.value ?? '—'} · ${item.trust}${item.trust_reasons.length ? ': ' + item.trust_reasons.join(', ') : ''}`;
    return `<i class="pl-box ${item.trust}" style="left:${x0 * 100}%;top:${y0 * 100}%;width:${(x1 - x0) * 100}%;height:${(y1 - y0) * 100}%" title="${esc(tip)}"></i>`;
  }).join('');
  return `<div class="pl-pages">${pages.map(p => `<button class="ev-chip ${p === page ? 'active' : ''}" onclick="msPage(${p})">Page ${p}</button>`).join('')}
      <span class="tiny"><i class="pl-key trusted"></i> trusted <i class="pl-key untrusted"></i> untrusted (hover for the reason)</span></div>
    <div class="pl-page"><img alt="Bulletin page ${page} (synthetic test data)" src="${PIPELINE_BASE}/pipeline/runs/${encodeURIComponent(r.run_id)}/pages/${page}">${boxes}</div>`;
}
function msPage(p) { pl.page = p; msOpen(pl.open); }

function evidenceTable(items) {
  return `<div class="table-wrap"><table class="client-table pl-table"><thead><tr><th>Metric</th><th>Period</th><th>Value</th><th>Baseline</th><th>OCR conf.</th><th>Trust</th></tr></thead><tbody>
    ${items.map(i => { const c = i.provenance.fields?.value?.confidence;
      return `<tr><td>${metricLabel(i.metric_id)}</td><td>${esc(i.observation_period.label)}</td><td><b>${esc(i.value ?? '—')}</b> ${esc(i.unit)}</td>
      <td>${i.baseline ? esc(i.baseline.value) + ' <span class="tiny">' + esc(i.baseline.period_label) + '</span>' : '—'}</td><td>${c != null ? Math.round(c * 100) + '%' : '—'}</td>
      <td>${i.trust === 'trusted' ? '<span class="status good">trusted</span>' : '<span class="status watch">untrusted</span><div class="tiny">' + esc(i.trust_reasons.join(', ')) + '</div>'}</td></tr>`; }).join('')}
  </tbody></table></div>`;
}

function decisionBlock(r) {
  const d = stage(r, 'decision'), v = stage(r, 'verifier'), rec = d.recommendation;
  if (!d.status) return '';
  if (d.status === 'error') return '';
  const findings = v.result ? v.result.findings.map(f => `<li><b>${esc(f.code)}</b> ${esc(f.detail || '')}</li>`).join('') : '';
  return `<h3>Recommendation <span class="tiny">Qwen3.5-9B · ${esc(d.prompt_version)} · ${d.prompt_tokens ?? '?'} + ${d.completion_tokens ?? '?'} tokens · ${d.model_seconds ? d.model_seconds.toFixed(1) + ' s' : ''}</span></h3>
    ${rec ? `<div class="recommend"><small>${esc(rec.recommended_action.replaceAll('_', ' '))} · magnitude ${esc(rec.magnitude)} · ${esc((rec.reason_codes || []).join(', ') || 'no reason codes')}</small><p>${esc(rec.explanation)}</p>
      <p class="tiny">Cites: ${(rec.evidence_ids || []).map(esc).join(', ') || '—'}</p></div>` : `<pre class="ev-passage">${esc(d.raw)}</pre>`}
    ${v.result ? `<p>Verifier ${esc(v.result.verifier_version)}: ${verdictPill(r)}</p>${findings ? '<ul class="ev-checks">' + findings + '</ul>' : ''}` : ''}`;
}

function runDetail(r) {
  const doc = stage(r, 'document'), ocr = stage(r, 'ocr'), ex = stage(r, 'extraction'), avm = stage(r, 'avm'), pol = stage(r, 'policy');
  const err = r.stages.find(s => s.status === 'error');
  const page = pl.page || 1;
  return `<div class="card-title"><div><div class="eyebrow">Run ${esc(r.run_id)}</div><h2>${esc((doc.variant || '').replaceAll('_', ' '))} bulletin</h2>
      <span class="tiny">${esc(doc.publisher || '')} · SYNTHETIC TEST DATA · ${esc(doc.document_id || '')} · for ${esc(r.customer.display_name)}</span></div>${verdictPill(r)}</div>
    ${stageStrip(r)}
    ${err ? `<div class="ev-problems"><b>Stopped at ${esc(STAGE_NAMES[err.name])}:</b> ${esc(err.error)}</div>` : ''}
    ${ocr.status === 'ok' ? `<p class="tiny">OCR: ${esc(ocr.engine)} ${esc(ocr.engine_version)} (${esc(ocr.model)}) · ${ocr.words} words · mean confidence ${Math.round(ocr.mean_confidence * 100)}% · language ${esc(ocr.language)}</p>` : ''}
    ${pageViewer(r, page)}
    ${ex.items ? `<h3>Evidence read from the bulletin <span class="tiny">${ex.trusted} trusted, ${ex.untrusted} untrusted${ex.failures?.length ? ', ' + ex.failures.length + ' rows not read' : ''}</span></h3>${evidenceTable(ex.items)}` : ''}
    ${avm.items ? `<h3>AVM band for the property <span class="tiny">${esc(avm.property.ward_name)}, ${esc(avm.property.district_name)} · ${esc(avm.property.total_area)} m² · flag ${esc(avm.property.flag)} · model output, asking prices</span></h3>${evidenceTable(avm.items)}` : ''}
    ${pol.result ? `<h3>Policy <span class="tiny">mapping ${esc(r.versions.mapping_id)} (${esc(r.versions.mapping_status)}, provisional) · ${esc(r.versions.policy)} (${esc(r.versions.policy_status)})</span></h3>
      ${pol.result.targets.map(t => `<p><b>${esc(t.target.replaceAll('_', ' '))}</b>: ${esc(t.outcome)} · allowed ${esc(t.allowed_actions.join(', '))} · cap ${esc(t.magnitude_cap)}${t.must_abstain ? ' · must abstain' : ''}<br><span class="tiny">${t.signals.map(s => `${esc(s.evidence_id)} ${s.effect ? esc(s.effect) : ''}${s.excluded_reason ? ' (excluded: ' + esc(s.excluded_reason) + ')' : ''}`).join(' · ')}</span></p>`).join('')}` : ''}
    ${decisionBlock(r)}
    ${r.review ? `<p class="ev-signed">Reviewed: <b>${esc(r.review.action.replaceAll('_', ' '))}</b> by ${esc(r.review.reviewer)} on ${esc(when(r.review.at))}${r.review.note ? ' · ' + esc(r.review.note) : ''}</p>` : ''}`;
}

// ---------------------------------------------------------------- Review queue

async function rqLoad() {
  let all;
  try { all = await evApi('/pipeline/runs'); plStatus('rqStatus', true, 'Pipeline API online'); }
  catch (e) { plStatus('rqStatus', false, 'Pipeline API offline'); return; }
  const routed = all.filter(r => r.verdict === 'ROUTE_TO_HUMAN' || r.verdict === 'BLOCK');
  const open = routed.filter(r => r.verdict === 'ROUTE_TO_HUMAN' && !r.review);
  document.getElementById('rqCount').textContent = open.length + ' open';
  document.getElementById('rqList').innerHTML = routed.map(r => `
    <div class="queue-item ${pl.queueOpen === r.run_id ? 'active' : ''}" onclick="rqOpen('${esc(r.run_id)}')">
      <div class="queue-top">${r.review ? `<span class="risk medium">${esc(r.review.action.replaceAll('_', ' '))}</span>` : r.verdict === 'BLOCK' ? '<span class="risk high">Blocked</span>' : '<span class="risk medium">Open</span>'}<span class="tiny">${esc(when(r.started_at))}</span></div>
      <h3>${esc((r.action || 'no recommendation').replaceAll('_', ' '))}</h3><p>${esc((r.variant || '').replaceAll('_', ' '))} bulletin · ${esc(r.customer_id)}</p></div>`).join('')
    || '<p class="tiny" style="padding:10px">Nothing routed yet. Runs the verifier sends to a person appear here.</p>';
  const first = open[0] || routed[0];
  if (!pl.queueOpen && first) rqOpen(first.run_id);
}

async function rqOpen(runId) {
  pl.queueOpen = runId;
  document.querySelectorAll('#rqList .queue-item').forEach(x => x.classList.toggle('active', x.getAttribute('onclick').includes(runId)));
  const r = await evApi('/pipeline/runs/' + encodeURIComponent(runId));
  const rec = stage(r, 'decision').recommendation, items = Object.fromEntries((stage(r, 'bundle').bundle?.items || []).map(i => [i.evidence_id, i]));
  const cited = rec ? [...new Set(rec.evidence_ids)].map(id => items[id]).filter(Boolean) : [];
  const canAct = r.verdict === 'ROUTE_TO_HUMAN' && !r.review;
  document.getElementById('rqDetail').innerHTML = `
    <div class="review-head"><div><div class="eyebrow">Run ${esc(r.run_id)}</div><h2>${esc(rec ? rec.recommended_action.replaceAll('_', ' ') : 'No valid recommendation')}</h2>
      <span class="tiny">For ${esc(r.customer.display_name)} · ${esc(stage(r, 'document').variant)} bulletin (synthetic) · mapping ${esc(r.versions.mapping_status)}</span></div>${verdictPill(r)}</div>
    ${decisionBlock(r)}
    ${cited.length ? `<h3>Cited evidence</h3>${evidenceTable(cited)}` : ''}
    <p><button class="opp-action" onclick="showAuthView(1);msOpen('${esc(r.run_id)}')">Open the full run in Market signals →</button></p>
    ${r.review ? `<p class="ev-signed">Reviewed: <b>${esc(r.review.action.replaceAll('_', ' '))}</b> by ${esc(r.review.reviewer)} on ${esc(when(r.review.at))}${r.review.note ? ' · ' + esc(r.review.note) : ''}</p>`
      : canAct ? `<div class="ev-decide"><textarea id="rqNote" maxlength="500" placeholder="Note (required to reject or request evidence)"></textarea></div>
      <div class="actions"><button class="danger-lite" onclick="rqAct('request_evidence')">Request evidence</button><button class="secondary" onclick="rqAct('reject')">Reject</button><button class="primary" onclick="rqAct('approve')">Approve and release</button></div>`
      : '<p class="tiny">A blocked recommendation cannot be released.</p>'}`;
}

async function rqAct(action) {
  const note = document.getElementById('rqNote').value.trim() || null;
  if (action !== 'approve' && !note) { showToast('Add a note first'); document.getElementById('rqNote').focus(); return; }
  if (action === 'approve' && !confirm('Approve this recommendation and release it to the customer?')) return;
  try {
    await evApi(`/pipeline/runs/${encodeURIComponent(pl.queueOpen)}/review`, {reviewer: await plReviewer(), action, note});
    showToast('Review saved'); await rqLoad(); await rqOpen(pl.queueOpen);
  } catch (e) { showToast(e.message); }
}

// ---------------------------------------------------------------- Audit log

async function auditLoad() {
  let a;
  try { a = await evApi('/pipeline/audit'); } catch (e) { document.getElementById('auditRuns').innerHTML = '<p class="tiny">Pipeline API offline.</p>'; return; }
  const h = (label, v) => v ? `<div><span class="tiny">${label}</span> <span class="ev-hash" title="${esc(v)}">${shortHash(v)}</span></div>` : '';
  document.getElementById('auditRuns').innerHTML = `<table class="client-table pl-table"><thead><tr><th>Run</th><th>Outcome</th><th>Versions</th><th>Hashes</th></tr></thead><tbody>
    ${a.runs.map(r => { const v = r.versions; return `<tr><td><b>${esc(when(r.started_at))}</b><div class="tiny">${esc(r.run_id)}</div></td>
      <td>${verdictPill(r)}${r.review ? `<div class="tiny">${esc(r.review.action)} by ${esc(r.review.reviewer)}</div>` : ''}</td>
      <td class="tiny">${esc(v.runner)} · catalogue ${esc(v.catalogue)} · ${esc(v.extractor)} · ${esc(v.policy || '')} (${esc(v.policy_status || '')}) · ${esc(v.mapping_id || '')} (${esc(v.mapping_status || '')}) · ${esc(v.model || '')} ${esc(v.precision || '')} · ${esc(v.prompt)} · ${esc(v.verifier)} · commit ${esc((v.code_commit || '').slice(0, 10))}</td>
      <td>${h('bulletin', r.pdf_sha256)}${h('bundle', v.bundle_hash)}${h('policy', v.policy_hash)}${h('catalogue', v.catalogue_hash)}${h('AVM file', r.avm_file_sha256)}</td></tr>`; }).join('') || '<tr><td colspan="4" class="tiny">No runs yet.</td></tr>'}
  </tbody></table>`;
  document.getElementById('auditActions').innerHTML = `<table class="client-table pl-table"><thead><tr><th>When</th><th>Action</th><th>Reviewer</th><th>Details</th></tr></thead><tbody>
    ${a.actions.map(x => { const {at, action, reviewer, ...rest} = x; return `<tr><td>${esc(when(at))}</td><td><b>${esc(action.replaceAll('_', ' '))}</b></td><td>${esc(reviewer || '—')}</td>
      <td class="tiny">${esc(Object.entries(rest).map(([k, v]) => k + ': ' + (typeof v === 'object' ? JSON.stringify(v) : v)).join(' · ').slice(0, 300))}</td></tr>`; }).join('') || '<tr><td colspan="4" class="tiny">No actions yet.</td></tr>'}
  </tbody></table>`;
}

// ---------------------------------------------------------------- Customer view

async function customerLoad() {
  let d;
  try { d = (await evApi('/pipeline/customers/demo-minh/latest')).latest; } catch (e) { return; }
  if (!d) return;
  const b = d.band, fmt = x => '₫' + Number(x).toFixed(2) + 'B';
  document.getElementById('propValue').textContent = fmt(b.value_p50.value);
  document.getElementById('propRange').textContent = `Model band (P10–P90) ${fmt(b.value_p10.value)} – ${fmt(b.value_p90.value)}`;
  document.getElementById('propLocation').textContent = `${d.property.ward_name}, ${d.property.district_name}, Hà Nội · ${d.property.total_area} m²`;
  const rec = d.recommendation;
  document.getElementById('propRecommendation').innerHTML = `<div class="recommend"><small>Latest recommendation · ${d.verdict === 'PASS' ? 'checked by the verifier' : 'approved by ' + esc(d.review.reviewer)} · ${esc(when(d.finished_at))}</small>
    <h4>${esc(rec.recommended_action.replaceAll('_', ' '))}</h4><p>${esc(rec.explanation)}</p>
    <ul class="pl-sources">${d.sources.map(s => `<li>${esc(s.source)}${s.synthetic ? ' · <b>SYNTHETIC TEST DATA</b>' : ''} · ${esc(s.evidence_id.split('.')[2].replaceAll('_', ' '))} ${esc(s.value)} ${esc(s.unit)} (${esc(s.period)})${s.published ? ', published ' + esc(new Date(s.published).toLocaleDateString()) : ''}</li>`).join('')}
      <li>${esc(b.value_p50.source)} · band from ${esc(b.value_p50.date)} · ${esc(b.value_p50.caveats[0])}</li></ul>
    <p class="tiny">Mapping ${esc(d.mapping_status)} and policy ${esc(d.policy_status)}: provisional, for demonstration.</p></div>`;
}

window.addEventListener('load', customerLoad);
document.querySelector('[data-role=authenticator]').addEventListener('click', () => rqLoad());
