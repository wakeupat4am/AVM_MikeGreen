# Qwen server setup plan

*Status, 2026-10-16: environment installed and the flow tested on the server (see [Test results](#test-results-2026-10-16)). Target: our own shared GPU server `vishc-server-1` (not `cecs-server1`) serving Qwen3.5-9B with plain `transformers`; vLLM is deferred until after the first experiments (see [Future: vLLM](#future-vllm-after-experiments)). Not yet done: the GPU agreement with other users, and the connection from the Mac.*

## Goal

Serve Qwen3.5-9B (the generation model for both the mapping agent and the decision engine) on our own server, reachable from the prototype only through an SSH tunnel, with a pinned configuration that all evaluation results are reported on. Swapping to another model, or to another inference engine, must be a config change, not a change to the callers.

Local serving on the Mac (Apple M4 Pro, 24 GB) was considered: it only fits a quantized copy, which behaves differently from the bf16 reference. It can be used for development, not for reported results.

## The server

Host `vishc-server-1`, surveyed 2026-10-16 (commands in [Server survey](#server-survey-2026-10-16)):

- **8 × NVIDIA GeForce RTX 3090, 24 GiB each** (192 GiB total), Ampere (sm_86).
- **Driver 595.91.07, driver CUDA 13.2.** No system CUDA toolkit (`nvcc` absent, no `/usr/local/cuda`). PyTorch wheels ship their own CUDA runtime; current PyPI `torch` wheels use CUDA 13 libraries, which this driver supports. Without `nvcc`, packages that only ship as source and need compiling (for example `causal-conv1d`) cannot be built.
- **Ubuntu 22.04.5 LTS**, kernel 5.15.0-198, glibc 2.35.
- **Python:** system 3.10.12 (too old for the plan's 3.11+); miniconda 25.5.1 with base Python 3.13.5. `uv` is not installed.
- **CPU and RAM:** 64 threads on 2 NUMA nodes, 660 GiB RAM.
- **PCIe topology:** GPUs pair up on shared PCIe switches: (0,1), (2,3), (4,5), (6,7). GPUs 0–3 sit on NUMA node 0 and GPUs 4–7 on node 1. A model split over two GPUs should use a **same-switch pair**.
- **Shared and busy.** Every GPU already has memory in use:

| GPU | Used | Free | Util | Held by |
|---|---|---|---|---|
| 0 | 3.5 GiB | 20.5 GiB | 79% | 26hung.nt (MESSI experiments), dang.cpm (cardiac) |
| 1 | 7.3 GiB | 16.7 GiB | 100% | 21hiep.nmh (speech diffusion, 6 GiB), dang.cpm |
| 2 | 13.3 GiB | 10.7 GiB | 76% | 26hung.nt (domainbed training) |
| 3 | 13.3 GiB | 10.7 GiB | 100% | 26hung.nt (domainbed training) |
| 4 | 13.2 GiB | 10.8 GiB | 14% | **24chuong.ta (us)**, `exchange-stage2` env, running ~6 h |
| 5 | 13.0 GiB | 11.0 GiB | 9% | **24chuong.ta (us)**, `exchange-stage2` env, running ~6 h |
| 6 | 1.8 GiB | 22.2 GiB | 0% | 26hung.nt (two small MESSI jobs) |
| 7 | 2.1 GiB | 21.9 GiB | 99% | tuan.nt2 (sam-rmsprop training), dang.cpm |

**What Qwen3.5-9B needs (from the model's `config.json` at the pinned revision):**
- Weights: 9.65 B parameters in bf16 = **19.3 GB (18.0 GiB)**, vision encoder included.
- Context cache is small. Only 8 of the 32 layers use full attention (4 KV heads, head dim 256), so the KV cache costs **32 KiB per token**: 8,192 tokens ≈ 256 MiB per sequence. The other 24 layers use linear attention (gated delta rule) with a fixed-size state per sequence. The binding constraint is the weights, not the context length.

Implications:
- **One GPU:** fits on a card with ~22 GiB free (GPU 6 today), with only 2–3 GiB left for activations and for the other user's jobs on that card. Workable, but either side can run out of memory if the other grows.
- **Two GPUs** (`device_map` split over a same-switch pair): ~9 GiB of weights per card, comfortable. With `transformers` this is a layer split: the cards take turns, so it buys memory room, not speed.
- **GPUs 4 and 5 are held by our own `exchange-stage2` job**, not another user's. They are a same-switch pair, so once that job ends they are the natural pair.
- Ampere has no native FP8. Use bf16 (reference) or 4-bit AWQ/GPTQ if a larger model is tried later.

## Model decision

**Stay with Qwen3.5-9B (bf16) as the reference model for now.**

- The pipeline puts correctness in deterministic code (policy engine, verifiers), and the hardest judgment (the mapping verifier's support check) goes to OpenAI. Qwen's job is narrow: retrieve, cite verbatim, fill a JSON schema, explain.
- Without the frozen test set, "more capable" is a guess. Switching models also resets every baseline number.
- Larger models compete for shared GPUs: 4-bit quantization or splitting across several cards.

Once the frozen test set exists, run a bake-off and keep the cheapest model that passes:

| Option | Fits on | Notes |
|---|---|---|
| Qwen3.5-9B, bf16 | 1–2 GPUs | Current choice; reference baseline |
| ~30B-class model, 4-bit (AWQ/GPTQ) | 1–2 GPUs | Likely a clear step up in reading and instruction-following |
| ~70B-class model, 4-bit | 3–4 GPUs | Only if smaller models fail and that many GPUs can be claimed |

A larger model would help most for the **mapping agent** (reading real research papers, some in Vietnamese, and quoting exactly), less for the decision engine. Check which model sizes are actually published, and supported by the installed libraries, before choosing. A bake-off multiplies the number of generation runs, so it is a likely point to move to vLLM.

## Inference engine decision (2026-10-16)

**Plain `transformers` now; vLLM later, only if the experiments show we need it.**

Plain `transformers` meets every hard requirement:

| Requirement | How `transformers` meets it |
|---|---|
| JSON-schema-constrained output (handoff §6) | A grammar library plugged into `generate()` as a logits processor. Proposed: `xgrammar`, the same engine vLLM uses, so constrained output behaves the same after a later switch |
| Exact replay (handoff Q8) | One request at a time, greedy decoding, fixed seed. This is the easiest setup to repeat exactly |
| Loopback only, customer data stays local | Unchanged |
| Swap models via config | The adapter loads `AutoModelForImageTextToText` and `AutoProcessor` from `VINFINX_QWEN_PATH` and uses the model's own chat template |
| Fits on shared 3090s | One GPU, or a `device_map` split over two GPUs with `max_memory` set from the environment |
| Qwen3.5 support | Already proven: the current adapter runs it on `cecs-server1` |

What we give up is throughput: requests run one at a time. The prototype's volume is small (a 30–50-case frozen set plus variants and counterfactual pairs), so this is acceptable until measured otherwise.

**Speed of the linear-attention layers.** In `transformers`, Qwen3.5's 24 linear-attention layers have fast kernels (`flash-linear-attention`'s gated delta rule, and `causal-conv1d`) with a slow pure-PyTorch fallback. `transformers` picks, in order: Hugging Face Hub kernels (only if requested with `use_kernels=True`), the installed package, then the PyTorch reference code, and logs a warning on fallback. Measured on 2026-10-16:
- gated delta rule: **`flash-linear-attention` 0.5.2** (Triton, installed as a wheel);
- `causal_conv1d_fn` / `causal_conv1d_update`: **PyTorch fallback**. `causal-conv1d` only ships as source and needs `nvcc`, which this server lacks.
- Hub kernels are **not used**: they download code from the Hub at run time, which is not pinned by our lock file. The `kernels` package is installed but unused.

Which path ran changes the numerics slightly, so it is recorded in the [pinned setup](#pinned-reference-setup).

## Serving architecture

```
Mac (browser, prototype UI)
   │  http://127.0.0.1:8061/vinfinx-api/...
   ▼
SSH tunnel over NetBird VPN  (start-qwen-bridge.sh: -L 8061:127.0.0.1:8061 via host alias vishc-1)
   ▼
Server, loopback only
   └─ qwen_api.py   127.0.0.1:8061   origin check, message limits, system prompt
        │                            (later: evidence bundle, policy, verifier, audit)
        └─ inference backend (in-process)
             now:   TransformersBackend  Qwen3.5-9B bf16, greedy, xgrammar-constrained JSON
             later: VLLMBackend          forwards to vLLM on 127.0.0.1:8000
```

- **One process for now.** `qwen_api.py` loads the model itself, as it does today, and keeps its lock so requests run one at a time.
- **A backend interface inside the adapter.** Callers (the UI, the test harness, the mapping agent) talk to the adapter's OpenAI-style `/v1/chat/completions` endpoint, which already exists. Inference sits behind a small `InferenceBackend` interface (messages, sampling settings and an optional JSON schema in; text out). Moving to vLLM later means adding a `VLLMBackend` and changing a config value; callers do not change.
- **JSON schema per request.** The request carries an optional JSON schema; the backend compiles it with `xgrammar` and constrains generation. The verifier still checks every output.

## Access from the Mac

**Policy (admin, 2026-10-16):** per VinSoc's security hardening request, server ports must not be exposed to the internet; the cloudflared tunnels are being removed from both servers. SSH permissions are unchanged.

So **no public relays** (Pinggy, cloudflared, ngrok or similar) for this project. The Pinggy alias `vishc-1-alt` used until now goes against the policy and is not used for VinFinx. Its tunnel does not run on `vishc-server-1` (no Pinggy process here; sessions arrived from `10.140.26.32`), so whoever runs it on that machine should stop it.

**Route: SSH over the NetBird VPN**, or directly on the LAN when on site.

| Route | Address | Status on the server (2026-10-16) |
|---|---|---|
| NetBird VPN | `vishc-server-1.netbird.cloud` / `100.87.200.188` | NetBird 0.77.1, management and signal connected |
| LAN (on site) | `10.140.26.19` | SSH on port 22 |

Proposed alias in the Mac's `~/.ssh/config` (key path is the user's own):

```
Host vishc-1
    HostName vishc-server-1.netbird.cloud
    User 24chuong.ta
    IdentityFile ~/.ssh/<your key>
    IdentitiesOnly yes
    ServerAliveInterval 30
```

- The Mac must be a NetBird peer with an access policy that lets it reach `vishc-server-1`. That is granted by whoever administers the NetBird network.
- **The Qwen port is never exposed, even on the VPN.** Port 8061 binds to the server's loopback and is reached only through SSH's `-L` forward.
- The server allows TCP forwarding (`AllowTcpForwarding yes`) and is key-only (`PasswordAuthentication no` in `/etc/ssh/sshd_config.d/00-vinsoc-hardening.conf`).
- `start-qwen-bridge.sh` opens its own connection (`-o ControlPath=none`), so the tunnel does not depend on a shared `ControlMaster` session if the alias uses one.

## What has to change from the `cecs-server1` setup

| File | `cecs-server1` assumption | Change |
|---|---|---|
| [`serve-qwen.sh`](../serve-qwen.sh) | Root `/home/ubuntu/llms-conformity-rq1`, venv at `server_experiments/.venv`, `.deps-qwen35` on `PYTHONPATH`, GPUs `0,1` | All paths, GPUs and ports from environment variables (`VINFINX_ROOT`, `VINFINX_PYTHON`, `VINFINX_QWEN_PATH`, `CUDA_VISIBLE_DEVICES`, `VINFINX_QWEN_PORT`); no `PYTHONPATH` hack |
| [`qwen_api.py`](../qwen_api.py) | `pipeline("image-text-to-text")`; model path under `multimodal_rq1/`; hardcoded 21 GiB per GPU; demo context in the system prompt | `InferenceBackend` interface with a `TransformersBackend` (`AutoModelForImageTextToText` + `AutoProcessor` + `generate()`); per-GPU memory limit from an environment variable; optional JSON schema per request via `xgrammar`; keep origin check, limits, greedy decoding, fixed seed and `enable_thinking=False`. Moving the demo context out of the system prompt is separate work |
| [`start-qwen-bridge.sh`](../start-qwen-bridge.sh) | SSH host alias `cecs-server1` | `VINFINX_SSH_HOST`, default `vishc-1` (NetBird); add `-o ControlPath=none` |
| Launch command | Runs `vinfinx_qwen_api:app`, but the file is `qwen_api.py` | Run `qwen_api:app` |

All four changes were made on 2026-10-16. In addition, `qwen_api.py` stops on both `<|im_end|>` and `<|endoftext|>` (see [Test results](#test-results-2026-10-16)), returns token `usage`, answers `503` while loading and `400` for a schema it cannot compile, and `serve-qwen.sh` sets `HF_HUB_OFFLINE=1` so the running service never contacts the Hub.

## Steps

### 1. Agree on GPUs and check the server
- Agree with the other users which GPUs we may claim (see [Open questions](#open-questions) item 1).
- Survey done 2026-10-16; see [Server survey](#server-survey-2026-10-16).

### 2. Prepare the environment
Create the dedicated environment and install the pinned packages ([Environment](#environment-installed-2026-10-16)). Done 2026-10-16.

### 3. Get the weights
Download Qwen3.5-9B at the pinned revision into `$VINFINX_ROOT/models/`. Done 2026-10-16; SHA-256 hashes in `$VINFINX_ROOT/models/Qwen3.5-9B.sha256`.

### 4. Smoke test
- Load the model on the agreed GPU(s), generate one greedy reply, and check which linear-attention path ran (fast kernels or PyTorch fallback).
- Generate the same prompt twice and confirm the outputs are identical.
- Generate once with a small JSON schema (an enum field) and confirm the output parses and stays inside the enum.

Done 2026-10-16 on GPU 7; see [Test results](#test-results-2026-10-16).

### 5. Start the adapter and check it
- On the server: `curl http://127.0.0.1:8061/vinfinx-api/health`. Done 2026-10-16.
- On the Mac: run `start-qwen-bridge.sh`, open the prototype, send one chat message. **Not yet done** (waiting for NetBird access from the Mac).
- Never bind the service to a public interface.

### 6. Measure
Record tokens per second, time per typical request (prompt length and output length as in the decision engine), and peak GPU memory. These numbers decide whether vLLM is needed ([Future: vLLM](#future-vllm-after-experiments)). First numbers in [Test results](#test-results-2026-10-16); measure again on the agreed GPUs with decision-engine-sized prompts.

### 7. Keep it running
Run the adapter in `tmux` (survives logout; a user-level `systemd` service would not, because lingering is off). See [Runbook](#runbook).

### 8. Pin the reference setup
Fill in [Pinned reference setup](#pinned-reference-setup). This is the reference configuration for all baseline and frozen-test-set results. Software is pinned; GPU indices are not, until the GPU agreement.

## Open questions

Answers from the 2026-10-16 survey. Items marked **still open** need a person, not a command.

1. **Which GPUs, and who owns GPUs 2–5?**
   - GPUs 2 and 3: **26hung.nt** (two `domainbed` training runs, ~13.3 GiB each).
   - GPUs 4 and 5: **us** (24chuong.ta), an `exchange-stage2` job, ~13 GiB each, running ~6 h at survey time. Freeing these is our own call.
   - **Recommendation:** GPUs 4,5 (same switch, our own memory) once `exchange-stage2` finishes; GPU 6 alone in the meantime.
   - **Still open:** confirm with the other users (26hung.nt, dang.cpm, tuan.nt2, 21hiep.nmh) that a long-running claim on 4,5 is acceptable, and for how long.
2. **OS and Python:** Ubuntu 22.04.5 LTS (kernel 5.15), glibc 2.35. System Python 3.10.12; miniconda 25.5.1 (base Python 3.13.5). Answered.
3. **Access:** SSH over the NetBird VPN (`vishc-server-1.netbird.cloud`), or the LAN on site; no public relays, per the admin's VinSoc policy note of 2026-10-16. See [Access from the Mac](#access-from-the-mac). **Still open:** confirm the Mac is a NetBird peer that can reach `vishc-server-1`, and test `ssh vishc-1` from it.
4. **Disk:**
   - Home (`/home/24chuong.ta`) has a **100 GiB soft / 293 GiB hard quota, already at 88 GiB**.
   - `/mnt/disk1` (NVMe, 7.0 TB, **1.3 TB free, no quota**) has a directory we own: `/mnt/disk1/backup_user/24chuong.ta`. Other users keep environments there too.
   - **Decision:** environment, weights, caches and the research corpus go under `/mnt/disk1/backup_user/24chuong.ta/vinfinx/`.
   - **Still open:** the name `backup_user` suggests a backup area. Check with the admin that it is not purged or rotated.
5. **Keep-alive:** `Linger=no` for our user, so a user-level `systemd` service would stop at logout. `KillUserProcesses` is at its default (no), and other users' jobs run for hours, so `tmux` survives logout. Answered.
6. **`cecs-server1`'s `.deps-qwen35` `transformers` version:** not answerable from here. Moot: current `transformers` releases include `qwen3_5`.

## Server survey (2026-10-16)

Read-only commands run on `vishc-server-1`; their findings are recorded above.

```bash
# OS, kernel, glibc
cat /etc/os-release; uname -r; ldd --version
# GPUs, driver, CUDA, topology
nvidia-smi
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu,driver_version --format=csv
nvidia-smi --query-gpu=index,uuid --format=csv,noheader
nvidia-smi --query-compute-apps=pid,gpu_uuid,used_memory,process_name --format=csv
ps -o pid=,user=,etime=,args= -p <pid>          # owner of each GPU process
nvidia-smi topo -m
which nvcc; ls -d /usr/local/cuda*                # no system toolkit
# Python and tooling
which -a python3; python3 --version; ls /usr/bin/python3*
which uv conda tmux git-lfs hf; conda --version; conda env list
# Disk, quota, CPU, RAM
df -h $HOME /mnt/disk1; du -sh $HOME; quota -s; nproc; free -g
ls -la /mnt/disk1/backup_user/24chuong.ta
# Access and keep-alive
echo $SSH_CONNECTION; ip -4 -brief addr
grep -iE 'AllowTcpForwarding|PasswordAuthentication' /etc/ssh/sshd_config /etc/ssh/sshd_config.d/*
loginctl show-user $USER -p Linger; grep KillUserProcesses /etc/systemd/logind.conf
ps -eo user,pid,etime,args | grep -iE 'pinggy|cloudflared|ssh .*-R'   # no tunnel process on this server
netbird status                                    # VPN connected, FQDN vishc-server-1.netbird.cloud
ss -ltn | grep -E ':(8000|8061)\b'                # both ports free
# Model metadata
curl https://huggingface.co/api/models/Qwen/Qwen3.5-9B                          # sha, architecture, files
curl "https://huggingface.co/api/models/Qwen/Qwen3.5-9B?expand[]=safetensors"   # parameter count
curl -L https://huggingface.co/Qwen/Qwen3.5-9B/resolve/<sha>/config.json        # layers, KV heads
# Library versions and Qwen3.5 support
curl https://pypi.org/pypi/<package>/json         # torch, transformers, accelerate, xgrammar,
                                                  # flash-linear-attention, causal-conv1d, kernels, vllm
curl https://raw.githubusercontent.com/huggingface/transformers/v5.20.0/src/transformers/models/qwen3_5/modeling_qwen3_5.py \
  | grep -E 'fla|causal_conv1d'                   # fast-kernel paths and fallbacks
curl https://raw.githubusercontent.com/vllm-project/vllm/v0.31.0/vllm/model_executor/models/registry.py | grep Qwen3_5
```

Hugging Face and PyPI are both reachable directly; no proxy is configured.

## Environment (installed 2026-10-16)

Approved and installed 2026-10-16. Versions are the latest releases that day, except `transformers`, where 5.19.0 was chosen over 5.20.0 (released the day before) to avoid a one-day-old release.

| Item | Value | Reason |
|---|---|---|
| Root | `VINFINX_ROOT=/mnt/disk1/backup_user/24chuong.ta/vinfinx` | No quota, 1.3 TB free; home is near its quota |
| Layout | `env/` (5.8 GB), `models/` (19 GB), `cache/`, `logs/`, `corpus/` | `corpus/` holds the documents retrieved by the source agent (originals, extracted text, chunks, index; one subfolder per frozen version). Public documents only; customer documents never go here. Nothing under `$VINFINX_ROOT` is committed to git |
| Environment | conda prefix env `$VINFINX_ROOT/env`, Python 3.12.15 from `conda-forge` | Dedicated env. `conda-forge` with `--override-channels` avoids the Anaconda default channel and its terms-of-service prompt |
| PyTorch | `torch==2.14.1+cu130`, `torchvision==0.29.1`, `triton==3.8.0` | CUDA 13.0 runtime, supported by driver CUDA 13.2. `torchvision` and `pillow` were added during the smoke test: Qwen3.5's image processor needs them |
| Model library | `transformers==5.19.0`, `accelerate==1.15.0`, `tokenizers==0.23.3`, `safetensors==0.8.0` | Includes `qwen3_5`; `accelerate` for `device_map` |
| Constrained output | `xgrammar==0.2.8` | JSON-schema logits processor; same engine as vLLM |
| Linear-attention kernels | `flash-linear-attention==0.5.2` (`fla-core==0.5.2`); `kernels==0.17.2` installed but unused | See [Inference engine decision](#inference-engine-decision-2026-10-16) |
| Adapter | `fastapi==0.143.2`, `uvicorn==0.54.0`, `pydantic==2.14.1` | Used by `qwen_api.py` |
| Images | `pillow==12.3.0` | Image processor |
| Download | `huggingface_hub==2.2.0` (`hf` CLI built in; the `[cli]` extra no longer exists) | Weights pinned by revision |
| Model | `Qwen/Qwen3.5-9B` at revision `c202236235762e1c871ad0ccb60c8ee5ba337b9a` (last modified 2026-03-02), bf16, not gated | In `$VINFINX_ROOT/models/Qwen3.5-9B` |
| Caches | `HF_HOME`, `PIP_CACHE_DIR` under `$VINFINX_ROOT/cache` | Keep them off the home quota |
| Full lock | `$VINFINX_ROOT/requirements.lock` (`pip freeze`, 94 packages) | Exact replay of the environment |

Commands run (2026-10-16):

```bash
export VINFINX_ROOT=/mnt/disk1/backup_user/24chuong.ta/vinfinx
export HF_HOME=$VINFINX_ROOT/cache/hf PIP_CACHE_DIR=$VINFINX_ROOT/cache/pip
mkdir -p $VINFINX_ROOT/{models,cache,logs,corpus}

conda create -y -p $VINFINX_ROOT/env -c conda-forge --override-channels python=3.12
P=$VINFINX_ROOT/env/bin/python
$P -m pip install "torch==2.14.1" "transformers==5.19.0" "accelerate==1.15.0" \
  "xgrammar==0.2.8" "flash-linear-attention==0.5.2" "kernels==0.17.2" \
  fastapi uvicorn pydantic "huggingface_hub[cli]"      # ~40 min, mostly CUDA wheels
$P -m pip install "torchvision==0.29.1" "pillow==12.3.0"   # needed by the image processor
$P -m pip check                                           # no broken requirements
$P -m pip freeze > $VINFINX_ROOT/requirements.lock

$VINFINX_ROOT/env/bin/hf download Qwen/Qwen3.5-9B \
  --revision c202236235762e1c871ad0ccb60c8ee5ba337b9a \
  --local-dir $VINFINX_ROOT/models/Qwen3.5-9B
(cd $VINFINX_ROOT/models/Qwen3.5-9B && sha256sum *.safetensors config.json tokenizer.json \
  chat_template.jinja > ../Qwen3.5-9B.sha256)

# smoke test (script kept outside the repo; it loads the model, repeats a greedy prompt,
# and runs one JSON-schema request)
CUDA_VISIBLE_DEVICES=7 VINFINX_QWEN_PATH=$VINFINX_ROOT/models/Qwen3.5-9B $P -I smoke_test.py

# flow test: the adapter in tmux, then curl on the server
tmux new-session -d -s vinfinx-qwen \
  "CUDA_VISIBLE_DEVICES=7 bash prototype/vinfinx/serve-qwen.sh 2>&1 | tee $VINFINX_ROOT/logs/qwen-api.log"
```

## Test results (2026-10-16)

Run on **GPU 7**, the only card with enough free memory at the time (GPUs 0 and 1 had just been taken by another user, GPU 6 had ~17 GiB free). GPU 7 was ~77% busy with dang.cpm's small jobs throughout, so speeds are under contention.

**Smoke test (model loaded directly):**

| Check | Result |
|---|---|
| Load | 67.5 s cold, 11.6 s with a warm file cache; whole model on one GPU |
| Memory | 17.5 GiB after load, 17.6–17.8 GiB peak during generation (`max_memory` 21 GiB) |
| Speed | ~18–20 tokens/s for a single request |
| Exact repeat | Identical output on a second greedy run (seed 7) |
| JSON schema | Output parses; `recommended_action` inside the enum; the prompt's bait ("you may approve the loan directly") did not produce an out-of-enum action |
| Linear-attention path | gated delta rule on `flash-linear-attention`; `causal_conv1d` on PyTorch fallback |

**Flow test (adapter on `127.0.0.1:8061`, `curl` on the server):**

| Test | Result |
|---|---|
| Health | `{"status":"ready","model":"Qwen3.5-9B"}` |
| Chat (220 prompt tokens, 188 output) | 200 OK, 9.8 s per request (~19 tokens/s). The first request after start took 59 s (one-time warm-up, likely Triton kernel compilation) |
| Same chat repeated | Identical response |
| JSON schema (235 prompt tokens, 139 output) | 200 OK, 7.4 s; parses, action inside the enum |
| No `Origin` header | 403 |
| Invalid schema (`"type": "nonsense"`) | 400 `Invalid JSON schema` |
| Request to the LAN address `10.140.26.19:8061` | Connection refused (loopback only, as required) |
| GPU memory while serving | ~18.6 GiB on GPU 7 |

The service was stopped after the test to free GPU 7, since the GPU claim is not yet agreed.

**Issues found and fixed:**
- **Stop token mismatch.** The tokenizer ends a turn with `<|im_end|>` (id 248046), but the model config only lists `<|endoftext|>` (id 248044), and the download has no `generation_config.json`. With a JSON grammar, the finished grammar allows only `<|im_end|>`, which did not stop generation, so the model emitted hidden tokens until `max_tokens` (300 tokens and 15.3 s instead of 94 tokens and 5.6 s). `qwen_api.py` now stops on both ids.
- **Missing dependencies.** `torchvision` and `pillow` were not in the original list; the image processor requires them.

**Observations for later work (not setup issues):**
- `maxLength` on a string in the JSON schema makes the grammar cut text off mid-word. Bound length with `max_tokens`, and let the verifier check length.
- The model repeats the demo context's "confidence 91%" as if it were a calibrated figure, and in the JSON test it combined `no_change` with magnitude `medium` and cited demo numbers the prompt did not supply. This is the demo context in the system prompt (known issue), and exactly the kind of output the verifier must catch.

## Runbook

```bash
# start (on the server; pick the agreed GPU)
export VINFINX_ROOT=/mnt/disk1/backup_user/24chuong.ta/vinfinx
tmux new-session -d -s vinfinx-qwen \
  "CUDA_VISIBLE_DEVICES=7 bash ~/chun/AVM_MikeGreen/prototype/vinfinx/serve-qwen.sh 2>&1 | tee -a $VINFINX_ROOT/logs/qwen-api.log"

# check (status "loading" for 15–70 s after start, then "ready")
curl -s http://127.0.0.1:8061/vinfinx-api/health
curl -s -H "Origin: http://127.0.0.1:4173" -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"Hello"}],"max_tokens":64}' \
  http://127.0.0.1:8061/vinfinx-api/v1/chat/completions
nvidia-smi --query-gpu=index,memory.used --format=csv     # ~18.6 GiB on our GPU

# logs / attach
tail -f $VINFINX_ROOT/logs/qwen-api.log
tmux attach -t vinfinx-qwen                               # detach: Ctrl-b d

# stop
tmux kill-session -t vinfinx-qwen
```

Two GPUs (once agreed): `CUDA_VISIBLE_DEVICES=4,5 VINFINX_GPU_MAX_MEMORY=11GiB`; `device_map="auto"` then splits the layers across both.

## Pinned reference setup

Software pinned 2026-10-16. GPU indices and the per-GPU memory limit are pinned once the GPU agreement is settled. Every baseline and frozen-test-set result is reported on this configuration.

| Item | Value |
|---|---|
| Server | `vishc-server-1`, 8 × RTX 3090 24 GiB, Ubuntu 22.04.5 |
| Driver / driver CUDA | 595.91.07 / 13.2 |
| GPU indices, device split | TBD (tested: GPU 7, single GPU) |
| Python | 3.12.15 (conda-forge) |
| PyTorch (+ CUDA runtime) | 2.14.1+cu130 (CUDA 13.0.96) |
| transformers / accelerate | 5.19.0 / 1.15.0 |
| xgrammar | 0.2.8 |
| Linear-attention path | gated delta rule: `flash-linear-attention` 0.5.2; `causal_conv1d`: PyTorch fallback; Hub kernels off |
| Full environment | `$VINFINX_ROOT/requirements.lock` |
| Model, revision | `Qwen/Qwen3.5-9B` @ `c202236235762e1c871ad0ccb60c8ee5ba337b9a` |
| Weight hashes (SHA-256, first 16 hex) | shard 1 `db6f444b43d318c9`, 2 `31c7d7e2dd5d2078`, 3 `7ec36ba3a4176a44`, 4 `b62b0c4cd7e44ede`; `config.json` `d0883072e01861ed`; `tokenizer.json` `5f9e4d4901a92b99`; `chat_template.jinja` `a4aee8afcf2e0711`. Full hashes in `$VINFINX_ROOT/models/Qwen3.5-9B.sha256` |
| Precision | bf16 |
| Stop tokens | `<|endoftext|>` (248044), `<|im_end|>` (248046) |
| Per-GPU memory limit | 21 GiB (tested); final value TBD |
| Max new tokens | 32–600 per request (adapter limit) |
| Sampling | Greedy (`do_sample=False`), seed 7, `enable_thinking=False` |
| Hub access at run time | Off (`HF_HUB_OFFLINE=1`) |

## Future: vLLM (after experiments)

vLLM stays the planned upgrade, adopted only when the measurements from [Steps](#steps) 6 and the first test-set runs show plain `transformers` is too slow.

**What it would add:** continuous batching (many requests at once, so test sets, mapping-agent sweeps and bake-offs run in minutes instead of hours), built-in JSON-schema output (`response_format`), predictable memory on shared GPUs (`--gpu-memory-utilization`), true tensor parallelism, and a separate OpenAI-compatible server on `127.0.0.1:8000` behind the adapter's `VLLMBackend`.

**Switch when any of these holds:**
- A full frozen-set run (with variants and counterfactual pairs) takes longer than an overnight window.
- The mapping agent's sweep over the corpus is too slow to iterate on.
- A model bake-off is scheduled (several models × the full set).

**Before switching:**
- Check that repeated runs give identical output under batching. If they do not, run evaluation with batch-invariant settings, or record outputs rather than relying on regenerating them (handoff Q8).
- Re-run the frozen set on vLLM and compare with the `transformers` baseline. The engine is part of the pinned configuration, so a switch is a new reference version.

**Facts gathered 2026-10-16:** vLLM 0.31.0 (released 2026-10-05) lists `Qwen3_5ForConditionalGeneration` in its model registry. It pins `torch==2.13.0` and `transformers>=5.10.4,<5.18`, which conflict with the pins above, so it needs its own environment (`$VINFINX_ROOT/env-vllm`). Example launch, for GPUs 4,5:

```bash
CUDA_VISIBLE_DEVICES=4,5 vllm serve "$VINFINX_QWEN_PATH" \
  --served-model-name qwen3.5-9b --host 127.0.0.1 --port 8000 \
  --dtype bfloat16 --tensor-parallel-size 2 \
  --max-model-len 8192 --gpu-memory-utilization 0.60
```

## Next

Done 2026-10-16: environment approved and installed, weights downloaded and hashed, smoke test, the reworked `serve-qwen.sh`, `qwen_api.py` and `start-qwen-bridge.sh`, flow test on the server, runbook, software pins.

Remaining:
1. Agree the GPUs with the other users; then pin the GPU indices and per-GPU memory limit.
2. Get NetBird access from the Mac, add the `vishc-1` alias, and test the prototype end to end (step 5).
3. Ask the admin whether `/mnt/disk1/backup_user` is purged or rotated.
4. Measure again on the agreed GPUs with decision-engine-sized prompts (step 6).
