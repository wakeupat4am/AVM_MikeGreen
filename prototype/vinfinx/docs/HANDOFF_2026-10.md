# Handoff: VinFinx AI layer, October 2026

*For the AI engineer taking over the VinFinx AI layer on a new server. Written 2026-10-24 from the state of branch `chun`.
Agreed with the project owner (jimmytrn154): you **rebuild everything from scratch** on your server. No data is transferred:
no corpus, no signed lists, no runs. Your source lists, sign-offs and measurements will be your own and will differ from ours.*

## 1. What you are taking over

VinFinx is a browser prototype (`index.html`) with an AI layer that turns a macro-status bulletin into a recommendation for a
customer's borrowing capacity and property valuation range, without letting a language model's output become a fact or a
decision on its own. Every piece follows one pattern: **the model proposes, deterministic code verifies, a person approves.**

Read these, in this order, before running anything:

1. [`AI_ENGINEERING_HANDOFF.md`](../AI_ENGINEERING_HANDOFF.md): the requirements (source of truth).
2. [`PIPELINE.md`](PIPELINE.md): the design, both halves.
3. [`/AGENTS.md`](../../../AGENTS.md): the current state of every piece, the hard rules and the lessons learned. It is written
   for coding agents, and it is the most complete status page.
4. [`../../AI_ENGINEER_FINDINGS.md`](../../AI_ENGINEER_FINDINGS.md): the dated decision log, including every measurement.

The two halves:

```
OFFLINE RESEARCH (per corpus version)
  source agent (Qwen) → you sign the source list → frozen corpus + index (OpenAI embeddings)
  → mapping agent (Qwen) → rules + support check (OpenAI) → mapping registry → you validate a set

ONLINE (per decision)
  synthetic bulletin → OCR (Tesseract) → checked evidence → + Phase 3 AVM band → policy engine (applies a mapping)
  → signed bundle → Qwen recommendation → verifier → pass / route to a person / block → customer view
```

Everything is shown in the site (Authenticator: Review queue, Market signals, Evidence sources, Audit log; Customer card).

## 2. What you need

| Item | Notes |
|---|---|
| A Linux server with a GPU | ~19 GiB free GPU memory for Qwen3.5-9B in bf16 (tested on RTX 3090 24 GiB); 16+ CPU threads help OCR |
| Disk | ~30 GB for environments and the model (measured: main env 5.9 GB, OCR env 0.8 GB, Qwen weights 19 GB); ~1 GB per corpus version; a few GB of logs and runs |
| conda (or mamba) | the environments are conda envs with pip packages |
| Network | Hugging Face (model download), GitHub (Tesseract language data), OpenAlex, the World Bank Documents API, bis.org, sbv.gov.vn, nso.gov.vn, api.openai.com |
| **Your own OpenAI API key** | embeddings, the support check and gold-label drafts; a full rebuild costs under $1 |
| A contact e-mail | for OpenAlex's `mailto=` and the User-Agent (the source agent identifies itself) |
| A Mac or laptop for the site | it reaches the server only through an SSH tunnel (NetBird/LAN; never a public relay) |

## 3. Set up the server

### 3.1 Paths

Nothing in the code points at a particular server: every path comes from [`ai/paths.py`](../ai/paths.py). Create the private
file `~/.config/vinfinx/paths.env` (mode 600, never committed):

```bash
mkdir -p ~/.config/vinfinx && chmod 700 ~/.config/vinfinx
cat > ~/.config/vinfinx/paths.env <<'EOF'
VINFINX_ROOT=/fast/disk/vinfinx        # environments, model weights, generated bulletins, caches
VINFINX_DATA=/big/disk/vinfinx         # corpus, pipeline runs, logs (omit to keep them under VINFINX_ROOT)
EOF
chmod 600 ~/.config/vinfinx/paths.env
set -a; . ~/.config/vinfinx/paths.env; set +a
mkdir -p $VINFINX_ROOT/{models,cache,logs,fixtures} ${VINFINX_DATA:-$VINFINX_ROOT}/{corpus,logs}
```

Without the file, everything goes under `~/vinfinx`. The `serve-*.sh` scripts read the same file.

### 3.2 Environments

Three conda environments; exact versions are frozen in [`requirements/`](../requirements/) (taken from the working server on
2026-10-24). Install with the lock files, not with "latest":

```bash
cd <repo>/prototype/vinfinx
export HF_HOME=$VINFINX_ROOT/cache/hf PIP_CACHE_DIR=$VINFINX_ROOT/cache/pip

# main: Qwen serving, every ai/ module, the pipeline API (~40 min, mostly CUDA wheels)
conda create -y -p $VINFINX_ROOT/env -c conda-forge --override-channels python=3.12
$VINFINX_ROOT/env/bin/python -m pip install -r requirements/main.lock

# OCR: Tesseract 5.5.3 and the few packages ai/ needs there
conda create -y -p $VINFINX_ROOT/env-ocr --override-channels -c conda-forge python=3.12 "tesseract=5.5.3"
$VINFINX_ROOT/env-ocr/bin/python -m pip install -r requirements/ocr.lock

# browser tests of the site (optional): Playwright + headless Chromium
$VINFINX_ROOT/env/bin/python -m venv $VINFINX_ROOT/env-webtest
$VINFINX_ROOT/env-webtest/bin/pip install -r requirements/webtest.lock
PLAYWRIGHT_BROWSERS_PATH=$VINFINX_ROOT/cache/ms-playwright \
  $VINFINX_ROOT/env-webtest/bin/python -m playwright install chromium     # Playwright 1.64.1, Chrome 156 (2026-10-24)
```

If `torch` fails to install for your CUDA driver, install the matching `torch==2.14.1` build first, then the rest of the lock.

### 3.3 Model weights and OCR language data

```bash
$VINFINX_ROOT/env/bin/hf download Qwen/Qwen3.5-9B --revision c202236235762e1c871ad0ccb60c8ee5ba337b9a \
  --local-dir $VINFINX_ROOT/models/Qwen3.5-9B
(cd $VINFINX_ROOT/models/Qwen3.5-9B && sha256sum -c <repo>/prototype/vinfinx/requirements/Qwen3.5-9B.sha256)

D=$VINFINX_ROOT/models/tesseract/tessdata_best-e12c65a; mkdir -p $D
for l in eng vie; do curl -sSL -o $D/$l.traineddata \
  https://github.com/tesseract-ocr/tessdata_best/raw/e12c65a915945e4c28e237a9b52bc4a8f39a0cec/$l.traineddata; done
cp $VINFINX_ROOT/env-ocr/share/tessdata/osd.traineddata $D/
(cd $D && sha256sum -c <repo>/prototype/vinfinx/requirements/tessdata_best-e12c65a.sha256)
```

Every hash must say `OK`. A different model revision or language data is a different system: record it.

### 3.4 Keys and names (never committed, never printed)

```bash
# OpenAI: your own key
echo "OPENAI_API_KEY=sk-..." > ~/.config/vinfinx/openai.env && chmod 600 ~/.config/vinfinx/openai.env
# bundle signing key for evidence bundles (HMAC); create once, keep private
(umask 077; python3 -c "import secrets;print('VINFINX_BUNDLE_KEY='+secrets.token_hex(32));print('VINFINX_BUNDLE_KEY_ID=<you>-2026-10')" \
  > ~/.config/vinfinx/bundle.env)
```

Two names are still ours and must be changed to yours:

- **Contact e-mail** of the source agent: `CONTACT` in [`ai/sources/__init__.py`](../ai/sources/__init__.py) (used only for
  OpenAlex `mailto=` and the User-Agent).
- **Reviewer allowed to sign and validate in the site:** `VINFINX_REVIEWERS` (environment variable read by `serve-pipeline.sh`;
  default `jimmytrn154`). Put `VINFINX_REVIEWERS=<your name>` in `paths.env` or export it before starting the API. Who may sign
  is the project owner's decision (section 6).

### 3.5 Check

```bash
cd <repo>/prototype/vinfinx
$VINFINX_ROOT/env/bin/python -m unittest discover -s ai/tests -t .     # 197 pass, 2 skipped (no GPU, no network needed)
$VINFINX_ROOT/env-ocr/bin/python -m unittest ai.tests.test_ocr         # the Tesseract tests skipped in the main env
```

## 4. Rebuild, in order

GPU etiquette first: **pick a GPU with less than 1 GB in use when you start, never a fixed index, and stop Qwen when a step is
done** (`nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits`). Run long steps in `tmux` with logs.

| # | Step | Command (from `prototype/vinfinx`, `P=$VINFINX_ROOT/env/bin/python`) | Needs | Time / cost | Check |
|---|---|---|---|---|---|
| 1 | Start Qwen | `tmux new-session -d -s vinfinx-qwen "CUDA_VISIBLE_DEVICES=<free gpu> bash serve-qwen.sh 2>&1 \| tee -a $VINFINX_ROOT/logs/qwen-api.log"` | GPU | ~30 s load | `curl -s 127.0.0.1:8061/vinfinx-api/health` → `ready` |
| 2 | Bulletins | `$P -m ai.documents.generate --set-version docs-v0.1-draft` · `… --set-version docs-v0.1-holdout-s7 --seed 7` · `… --set-version docs-v0.1-demo-<YYYYMMDD> --as-of <YYYY-MM-DD>` | – | ~3 min each | 62 documents per set (check the count) |
| 3 | OCR + extraction check | `$VINFINX_ROOT/env-ocr/bin/python -m ai.extraction.evaluate --set docs-v0.1-draft` | CPU | 10–30 min (CPU load) | **0 wrong-and-trusted** (`docs/OCR_RESULTS.md`) |
| 4 | Adversarial set | `$P -m ai.evaluation.runner --mode scripted --split all --policy engine`, then `--mode model --split dev --policy engine` | Qwen | ~15 min | scripted 13/13; dev results vs `docs/EVALUATION_RESULTS.md`. **Never tune on `--split holdout`** |
| 5 | Source run 1 | `$P -m ai.sources.run --run-id <YYYYMMDD>-sources-1` | Qwen, network | 30–60 min; OpenAlex ~$0.05 | `corpus/runs/<id>/report.json`; IMF/other blocked PDFs listed in `manual_downloads.csv` (a person downloads them into `corpus/inbox`, then re-run) |
| 6 | Pipeline API + site | `tmux new-session -d -s vinfinx-api "bash serve-pipeline.sh 2>&1 \| tee -a $VINFINX_DATA/logs/pipeline_api.log"`; on the Mac: README "Running the demo" (set `VINFINX_SSH_HOST` to your SSH alias) | CPU | – | site shows "Pipeline API online" |
| 7 | Sign v0.1 | Authenticator → Evidence sources → Source list (or `python -m ai.sources.review --run-id … --reviewer … --version source_list/v0.1`) | a person | ~15 min | `corpus/source_lists/source_list-v0.1.json` |
| 8 | Corpus v0.1 | `$P -m ai.corpus.index build --list $VINFINX_DATA/corpus/source_lists/source_list-v0.1.json --version corpus-v0.1` | OpenAI, env-ocr | ~10 min; < $0.50 | manifests `frozen: true`; Corpus tab |
| 9 | Gold labels (optional now) | `$P -m ai.corpus.evaluate draft --version corpus-v0.1`; review in the Gold labels tab | OpenAI, a person | $0.12 + ~1 h review | `gold/metrics.json` |
| 10 | Mapping agent | `$P -m ai.mapping run --corpus corpus-v0.1`; review the set in the Mappings tab | Qwen, OpenAI | ~3 min; cents | `corpus/mappings/`; `python -m ai.mapping sets` shows what the runner uses |
| 11 | Runner | site: Market signals → Run pipeline, or `$P -m ai.runner run --document LBEO-BUL-2026-10-clean-en-s42 --set docs-v0.1-demo-<date>` (set `VINFINX_DEMO_SET` for the site) | Qwen | ~25 s per run | run record in `$VINFINX_DATA/pipeline_runs/`; Review queue, Audit log, Customer card |
| 12 | Stop Qwen | `tmux kill-session -t vinfinx-qwen` | – | – | GPU memory freed |

Expect your numbers to differ from ours where live sources are involved (step 5 onward). The document chain (steps 2–4) is
deterministic and should reproduce our results.

## 5. What is left, in priority order

| # | Task | Why | Where to start |
|---|---|---|---|
| 1 | **Replace the provisional mapping** | Every decision still uses a hand-written, synthetic mapping (`ai/policy/provisional_mapping.py`) because the corpus has no Viet Nam research on the three pairs the policy weights (refinancing rate, mortgage rate, listing inventory → borrowing capacity). Two free-source runs found almost none (findings log, 2026-10-24) | Proposed, not yet planned: **mapping entries derived from loan arithmetic** for the two rate pairs (a higher rate lowers the affordable loan for the same payment; computed and checked by code, labelled "not from research"); write a plan first. Listing inventory: a domain-owner decision (section 6) |
| 2 | Offline half live in the demo | The site shows research results but cannot start the agents | [`OFFLINE_DEMO_PLAN.md`](OFFLINE_DEMO_PLAN.md): "Run mapping agent" button, pipeline overview page |
| 3 | Demo bulletins dated for the demo day | Listing inventory is fresh for 14 days, monthly figures for ~45; stale figures are excluded | step 2 with `--as-of` the day before; set `VINFINX_DEMO_SET` |
| 4 | Rehearse from the laptop | Only parts of the site were used through the tunnel | README "Running the demo" |
| 5 | Decision coverage | On 62 bulletins: 39 blocked, 6 passed, 17 routed (`runner_prompts/0.2`). Safe, but many blocks. Next idea: `runner_prompts/0.4` (do not state AVM figures; the site shows the band) | [`RUNNER_PLAN.md`](RUNNER_PLAN.md) "As built", findings log |
| 6 | Mapping agent quality | Credit growth → house prices is refused although its evidence is in the corpus; lag and size are unchecked Qwen guesses (unused); dev-split mapping cases with the support check (M8) not run | [`MAPPING_AGENT_PLAN.md`](MAPPING_AGENT_PLAN.md) "As built" |
| 7 | Retrieval | Keyword search is weak (stopwords; fusion weighted toward vector); fix after the gold review | [`CORPUS_INDEX_PLAN.md`](CORPUS_INDEX_PLAN.md) |
| 8 | Qwen second reading of the bulletin | In the design, not built | [`PIPELINE.md`](PIPELINE.md) "OCR service" |
| 9 | Later | Holdout run of the adversarial set (once, after the system changes), cost accounting (handoff §9), the Advisor view, the chat's customer context | `AGENTS.md` "Next" |

## 6. Decisions that are not yours alone

| Decision | Owner | State |
|---|---|---|
| Who signs source lists, validates mapping sets, reviews gold labels | project owner (jimmytrn154) | ask before signing anything in their name |
| Publisher terms: World Bank and BIS pass (non-commercial use, decided 2026-10-22); **IMF** reading proposed with conditions, not confirmed; SBV unread | project owner | findings log, 2026-10-24 |
| Policy weights and thresholds, incl. whether listing inventory counts for borrowing capacity; weights for valuation-range pairs | **domain owner** | `ai/policy/policy_v0.yaml` is provisional (PIPELINE.md "Open") |
| OCR confidence threshold | **risk owner** | proposal in `OCR_CONFIDENCE_THRESHOLD.md` |
| Freshness limits in the catalogue (incl. 180 days for the AVM band) | risk owner | provisional |
| Anything under "Open" in `PIPELINE.md`; any new component | propose a plan in `docs/`, get approval, then build | the working style of this project |

## 7. Rules that are not negotiable

- Customer data, identity documents and bank records never go to OpenAI or any external API; only public research text and
  the synthetic bulletin may. The search box in the site refuses queries that look like personal data.
- Qwen and the pipeline API bind to `127.0.0.1` only; the site reaches them through SSH `-L`. Never open a port publicly.
- Never commit weights, keys, credentials, bank data or generated data (bulletins, corpus, runs, logs). The lock files and hash
  lists in `requirements/` are configuration and are committed.
- Synthetic data is labelled synthetic (the bulletin's watermark and fictional publisher; the demo customer file).
- The model never decides what a macro signal means at decision time: a versioned mapping applied by code does.
- Version and hash everything a decision depends on; never overwrite a frozen corpus, a signed list, a reviewed gold set or a
  mapping set: a change is a new version.
- Never tune prompts or the verifier on the adversarial set's `holdout` cases or on `docs-v0.1-holdout-s7`.
- Respect `robots.txt`; never get around bot protection; IMF content only by a person, non-systematically.
- Shared GPUs: pick a free one at start, stop Qwen when done.

## 8. Code map

| Area | Code | Plan |
|---|---|---|
| Paths | `ai/paths.py` | this document |
| Contracts | `ai/catalog/`, `ai/schemas/` | `SPECS_AND_SCHEMAS_PLAN.md` |
| Verifiers | `ai/verifier/` (v0.2.0) | `PIPELINE.md` |
| Adversarial set | `ai/evaluation/` (frozen v0.1.0) | `EVALUATION_RESULTS.md` |
| Policy engine | `ai/policy/` | `POLICY_ENGINE_PLAN.md` |
| Bulletins, OCR, extraction | `ai/documents/`, `ai/ocr/`, `ai/extraction/` | `MACRO_DOCUMENT_SPEC.md`, `OCR_RESULTS.md` |
| Source agent | `ai/sources/` (queries v0, v1) | `SOURCE_AGENT_PLAN.md`, `SOURCE_RUN_2_PLAN.md`, `CORPUS_SOURCE_CRITERIA.md` |
| Corpus, index, gold labels | `ai/corpus/` | `CORPUS_INDEX_PLAN.md` |
| Mapping agent, registry | `ai/mapping/` | `MAPPING_AGENT_PLAN.md` |
| AVM evidence, runner | `ai/avm/`, `ai/runner/` | `RUNNER_PLAN.md` |
| Pipeline API, site | `pipeline_api.py`, `ai/api/`, `index.html`, `evidence.js`, `pipeline.js` | `PIPELINE_API_PLAN.md`, `OFFLINE_DEMO_PLAN.md` |
| Qwen serving | `qwen_api.py`, `serve-qwen.sh` | `SERVER_SETUP_PLAN.md` |

Keep `AGENTS.md` "Current state", the tracker in `docs/README.md` and the findings log current at the end of each step: they
are how the next person picks up your work.
