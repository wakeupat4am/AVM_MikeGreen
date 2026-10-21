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
- Use any GPU with enough free memory; no formal claim needed (see [Open questions](#open-questions) item 1).
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
- On the Mac: run `start-qwen-bridge.sh`, open the prototype, send one chat message. NetBird access from the Mac works (2026-10-17); the full prototype check through the bridge is still to be run.
- Never bind the service to a public interface.

### 6. Measure
Record tokens per second, time per typical request (prompt length and output length as in the decision engine), and peak GPU memory. These numbers decide whether vLLM is needed ([Future: vLLM](#future-vllm-after-experiments)). All measurements so far are in [Generation speed log](#generation-speed-log); a 5–7× slowdown seen on 2026-10-17 is under investigation.

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
   - **Answered 2026-10-17:** no formal claim is needed; any GPU with enough free memory may be used. Pick one at start time (`nvidia-smi`; ~19 GiB free for one GPU) and record the index used with each run. All eight cards are the same model (RTX 3090), so the index does not change the reference configuration.
2. **OS and Python:** Ubuntu 22.04.5 LTS (kernel 5.15), glibc 2.35. System Python 3.10.12; miniconda 25.5.1 (base Python 3.13.5). Answered.
3. **Access:** SSH over the NetBird VPN (`vishc-server-1.netbird.cloud`), or the LAN on site; no public relays, per the admin's VinSoc policy note of 2026-10-16. See [Access from the Mac](#access-from-the-mac). **Answered 2026-10-17:** a NetBird trial from the Mac worked.
4. **Disk:**
   - Home (`/home/24chuong.ta`) has a **100 GiB soft / 293 GiB hard quota, already at 88 GiB**.
   - `/mnt/disk1` (NVMe, 7.0 TB, **1.3 TB free, no quota**) has a directory we own: `/mnt/disk1/backup_user/24chuong.ta`. Other users keep environments there too.
   - **Decision:** environment, weights, caches and the research corpus go under `/mnt/disk1/backup_user/24chuong.ta/vinfinx/`.
   - **Answered 2026-10-17:** the area is not purged; it is safe for long-term storage. Note that `/mnt/disk1` had dropped to 485 GB free (93% full) by 2026-10-17, so check free space before large downloads.
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

## Generation speed log

Every measured generation time, single requests with greedy decoding. Times are wall-clock per request; inputs are short, so output length dominates. Investigation status is in [Speed investigation](#speed-investigation).

### 2026-10-16: GPU 7, ~19 tokens/s

Smoke test (model loaded directly, no adapter):

| Request | Input tokens | Output tokens | Time | Tokens/s |
|---|---|---|---|---|
| Free text | 30 | 89 | 4.5 s | 19.7 |
| Same, repeated | 30 | 89 | 4.4 s | 20.2 |
| Free text, after the stop-token fix | 30 | 87 | 4.7 s | 18.4 |
| JSON schema, before the stop-token fix | ~60 | 300 (ran to the limit) | 15.3 s | 19.6 |
| JSON schema, after the fix | ~60 | 94 | 5.6 s | 16.9 |

Flow test (through the adapter):

| Request | Input tokens | Output tokens | Time | Tokens/s |
|---|---|---|---|---|
| First chat after start (includes warm-up) | 220 | 188 | 59.2 s | n/a |
| Chat | 220 | 188 | 9.8 s | 19.1 |
| Same chat, repeated | 220 | 188 | 9.9 s | 19.0 |
| JSON schema | 235 | 139 | 7.4 s | 18.8 |

Model load: 67.5 s cold, 11.6 s with the files already in the page cache.

### 2026-10-17 (late evening): GPUs 2 and 3, 2–4.5 tokens/s

Live schema checks (adapter on GPU 2; [`SPECS_AND_SCHEMAS_PLAN.md`](SPECS_AND_SCHEMAS_PLAN.md) acceptance check):

| Request | Input tokens | Output tokens | Time | Tokens/s |
|---|---|---|---|---|
| Recommendation, bundle X (first request, includes warm-up) | 396 | 199 | 92.6 s | 2.1 |
| Recommendation, bundle Y | 487 | 212 | 70.3 s | 3.0 |
| Mapping proposal | n/a | 400 (ran to the limit on whitespace) | not recorded | n/a |

Timing checks (adapter on GPU 2, warm):

| Request | Input tokens | Output tokens | Time | Tokens/s |
|---|---|---|---|---|
| Plain text | 243 | 155 | 57.4 s | 2.7 |
| JSON schema | 243 | 200 | 79.7 s | 2.5 |
| JSON schema, repeated | 243 | 200 | 56.9 s | 3.5 |
| Plain, repeated | 243 | 155 | 58.0 s | 2.7 |
| Plain, repeated again | 243 | 155 | 61.4 s | 2.5 |
| Plain, slightly different prompt | 246 | 143 | 34.7 s | 4.1 |
| Short ("Hi") | 209 | 32 | 11.7 s | 2.7 |

Speed test (model loaded directly on GPU 3; no adapter, no schema; after warm-up; 150 tokens per run):

| Setting | Run 1 | Run 2 |
|---|---|---|
| Default (32 threads, any core) | 51.2 s, 2.9 tokens/s | 33.4 s, 4.5 tokens/s |
| 8 threads, pinned to the GPU's NUMA node (cores 0–15, 32–47) | 52.7 s, 2.8 tokens/s | 51.2 s, 2.9 tokens/s |

Grammar cost, measured separately with Qwen's tokenizer: 0.4–0.5 s to compile a schema; about 87 ms of mask work across a whole output (median 0.005 ms per token).

### 2026-10-18: GPU 7

Live schema re-check (adapter, `device_map="auto"`), after the whitespace and key-order fixes:

| Request | Input tokens | Output tokens | Time | Tokens/s |
|---|---|---|---|---|
| Recommendation, bundle X (first request) | 396 | 189 | 11.1 s | 17.0 |
| Recommendation, bundle Y | 487 | 205 | 9.7 s | 21.1 |
| Mapping proposal | 280 | 90 | 4.3 s | 20.9 |

Speed test (model loaded directly, after warm-up, 150 tokens per run), varying only the model placement:

| Setting | Run 1 | Run 2 |
|---|---|---|
| `device_map={"": 0}` (as in the 2026-10-17 speed test) | 40.8 s, 3.7 tokens/s | 46.2 s, 3.2 tokens/s |
| `device_map="auto"` (as in the adapter) | 7.1 s, 21.2 tokens/s | 7.2 s, 20.8 tokens/s |
| `device_map="auto"` + `min_new_tokens=150` | 6.9 s, 21.8 tokens/s | 7.3 s, 20.7 tokens/s |
| `device_map="cuda:0"` | 46.9 s, 3.2 tokens/s | 46.4 s, 3.2 tokens/s |

Load average 19–28 during these runs, similar to 2026-10-17.

### 2026-10-18: GPU 2, through the adapter

First check of the evaluation runner (`device_map="auto"`):

| Request | Output tokens | Time | Tokens/s |
|---|---|---|---|
| ADV-001 recommendation | 206 | 13.3 s | 15.5 |
| ADV-005 recommendation | 256 | 12.5 s | 20.5 |
| ADV-033 mapping proposal | 113 | 5.7 s | 19.8 |

### Speed investigation

- **2026-10-18 finding: model placement.** On the same GPU and load, loading with `device_map="auto"` gives ~21 tokens/s and loading onto one explicit device (`{"": 0}` or `"cuda:0"`) gives ~3.2 tokens/s. The 2026-10-17 speed tests used explicit placement, so their slowness is explained. The cause inside `transformers`/`accelerate` is not yet understood. **Always load with `device_map="auto"`** (the adapter already does).
- **Resolved as transient:** the adapter (which uses `"auto"`) was slow on GPU 2 on 2026-10-17 (2.5–4.1 tokens/s), but ran at 15–20 tokens/s on the same card on 2026-10-18. Most likely contention at that moment; re-check only if it recurs.
- **Size (2026-10-17):** a 5–7× slowdown (about 50 ms per token on 2026-10-16, 220–400 ms per token on 2026-10-17). At the slow rate a 200-token decision takes 60–80 s; at the fast rate about 10 s.
- **Ruled out:** the JSON-schema grammar (negligible cost above); the adapter (the direct speed test is equally slow); GPU 2 specifically (GPU 3 is the same); the PyTorch CPU thread pool (limiting and pinning made no difference); per-shape kernel recompilation (identical repeats stay slow).
- **Observed alongside:** a higher server load average (22–31, against ~17 earlier), heavy jobs from other users on GPUs 0, 1, 5, 6 and 7, and large run-to-run variation (2.5–4.5 tokens/s on the same request), which fits contention better than a fixed fault.
- **Next checks:** the same speed test on GPU 4 (other CPU socket) and on GPU 7 (the 2026-10-16 card), and a short PyTorch profile of one decode step to see whether time goes to CPU-side kernel launches or to GPU execution.
- **Why it matters:** it does not affect correctness or the schema work, but it decides how long baseline runs take, and it feeds the cost comparison (handoff §9) and the vLLM decision.

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

Software pinned 2026-10-16. The GPU index is chosen per run from whatever is free and recorded with each run; the per-GPU memory limit stays at 21 GiB on a single GPU. Every baseline and frozen-test-set result is reported on this configuration.

| Item | Value |
|---|---|
| Server | `vishc-server-1`, 8 × RTX 3090 24 GiB, Ubuntu 22.04.5 |
| Driver / driver CUDA | 595.91.07 / 13.2 |
| GPU indices, device split | Any free RTX 3090, single GPU; index recorded per run (tested: GPU 7) |
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
| Per-GPU memory limit | 21 GiB |
| Max new tokens | 32–600 per request (adapter limit) |
| Sampling | Greedy (`do_sample=False`), seed 7, `enable_thinking=False` |
| Hub access at run time | Off (`HF_HUB_OFFLINE=1`) |

## OCR environment (installed 2026-10-20, CPU)

For the OCR service (`PIPELINE.md`, "OCR service"; build step 6). The AI engineer chose PaddleOCR on CPU only (no Tesseract baseline). The install and the outcome are recorded under [Installed](#installed-2026-10-20) below.

### Facts gathered (2026-10-20)

| Item | Finding |
|---|---|
| PaddleOCR | `paddleocr` 3.7.0 (current on PyPI), with `paddlex` 3.7.2 |
| Vietnamese | PaddleOCR 3.7.0 maps `vi` (and `en`) to **PP-OCRv6** (`PP-OCRv6_medium_det` + `PP-OCRv6_medium_rec`), one multilingual model pair (read from the package source, `_pipelines/ocr.py`) |
| Models | Hosted by the PaddlePaddle organisation on Hugging Face (PaddleX's default model source): detector 62 MB, recognizer 77 MB |
| CPU framework | `paddlepaddle` 3.3.1 on PyPI (195 MB wheel). Resolving `paddleocr==3.7.0 paddlepaddle==3.3.1` (dry run, nothing installed): 70 packages, **344 MB download** |
| GPU framework | Not on PyPI beyond 2.6.2. Paddle's own index (paddlepaddle.org.cn) has `paddlepaddle-gpu` 3.4.0 for CUDA 13.0: the wheel alone is **1.84 GB**, the download kept failing (SSL EOF, resumed 3 times; stopped), and its CUDA library dependencies were not resolved |
| Conflicts with the Qwen env | The resolution pulls `numpy` 2.3.5 (env has 2.5.4), `pillow` 13.0.0 (12.3.0), `huggingface_hub` 2.2.0 and `protobuf`, `opencv-contrib-python` 4.10.0.84. It must not go into `$VINFINX_ROOT/env` |
| Other engines | Tesseract is not installed (needs `apt`, no root). EasyOCR and VietOCR depend on PyTorch; not assessed yet |
| Disk | Each user may store up to **400 GB** on `/mnt/disk1` (per the AI engineer, 2026-10-20); we use 36 GB. But the disk itself is **100% full, ~8.4 GB free** (was 1.3 TB free on 2026-10-16; other users), so a large install can still fail until space is freed. Home: 100 GB soft quota, ~90 GB used. `$VINFINX_ROOT/cache/pip` holds **2.9 GB** of downloaded wheels (the PyTorch stack and the aborted GPU wheel) that can be cleared |
| PaddleX cache | Defaults to `~/.paddlex` (the home quota); must be redirected |

### Proposal

| | Choice | Why |
|---|---|---|
| Framework | **CPU** `paddlepaddle==3.3.1` from PyPI | ~0.35 GB download vs several GB for GPU, which does not fit the free disk, and the GPU wheel's host is unreliable from here. The evaluation load is small (62 documents × 3 pages). 64 cores, 660 GB RAM. Revisit GPU if CPU speed is not enough, or when disk frees up |
| Environment | New conda prefix env `$VINFINX_ROOT/env-ocr`, Python 3.12 (conda-forge), like the Qwen env | Its pins conflict with the Qwen env. Estimated ~1.5 GB installed |
| Packages | `paddlepaddle==3.3.1`, `paddleocr==3.7.0` (pulls `paddlex==3.7.2`); exact versions frozen to `$VINFINX_ROOT/requirements-ocr.lock` after install | Same pinning as the Qwen env |
| Models | `PaddlePaddle/PP-OCRv6_medium_det` and `PP-OCRv6_medium_rec` downloaded once to `$VINFINX_ROOT/models/paddleocr/` at a pinned revision, with SHA-256 hashes; loaded by `model_dir`, so runs need no network | Replay: the engine, version and model are recorded with every field |
| Caches | `PADDLE_PDX_CACHE_HOME=$VINFINX_ROOT/cache/paddlex`, `PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True`, `PIP_CACHE_DIR=$VINFINX_ROOT/cache/pip` | Keep off the home quota; no connectivity check at start |
| Disk | Clear `$VINFINX_ROOT/cache/pip` first (`pip cache purge`, frees ~2.9 GB) | Leaves room on a 9 GB margin |
| Code | The OCR adapter (`ai/ocr/`) and OCR evaluation run in `env-ocr` (it has `pydantic` and `PyYAML`); the Qwen second reading is called over HTTP on the adapter as today | No process imports both stacks |

### Commands (to run after approval)

```bash
export VINFINX_ROOT=/mnt/disk1/backup_user/24chuong.ta/vinfinx
export PIP_CACHE_DIR=$VINFINX_ROOT/cache/pip PADDLE_PDX_CACHE_HOME=$VINFINX_ROOT/cache/paddlex
$VINFINX_ROOT/env/bin/pip cache purge                                  # ~2.9 GB back
df -h /mnt/disk1                                                       # check before installing
conda create -y -p $VINFINX_ROOT/env-ocr --override-channels -c conda-forge python=3.12
O=$VINFINX_ROOT/env-ocr/bin/python
$O -m pip install "paddlepaddle==3.3.1" "paddleocr==3.7.0"
$O -m pip freeze > $VINFINX_ROOT/requirements-ocr.lock
$O -c "import paddle; paddle.utils.run_check()"                         # CPU check
# models, pinned (revision recorded here after download)
$VINFINX_ROOT/env/bin/hf download PaddlePaddle/PP-OCRv6_medium_det --local-dir $VINFINX_ROOT/models/paddleocr/PP-OCRv6_medium_det
$VINFINX_ROOT/env/bin/hf download PaddlePaddle/PP-OCRv6_medium_rec --local-dir $VINFINX_ROOT/models/paddleocr/PP-OCRv6_medium_rec
```

Then a smoke test: OCR page 1 of `docs-v0.1-draft/LBEO-BUL-2026-10-clean-en-s42` and `...-clean-vi-s42`, and report time per page and a first look at accuracy.

### Installed (2026-10-20)

- **Done:**
  - Ran `pip cache purge` (1003 directories; `/mnt/disk1` went from 8.4 to 12 GB free).
  - Created `env-ocr` (Python 3.12.15).
  - Installed `paddlepaddle==3.3.1` (CPU) and `paddleocr==3.7.0` in 3.5 min: 1.5 GB, 71 packages, frozen in `$VINFINX_ROOT/requirements-ocr.lock`.
  - `paddle.utils.run_check()` passed (CPU).
- **Key pins:** `paddlepaddle==3.3.1`, `paddleocr==3.7.0`, `paddlex==3.7.2`, `numpy==2.3.5`, `pillow==13.0.0`, `pydantic==2.14.1`, `PyYAML==6.0.2`.
- **Models:** in `$VINFINX_ROOT/models/paddleocr/` (133 MB), with SHA-256 hashes in `PADDLEOCR.sha256`.
  - `PaddlePaddle/PP-OCRv6_medium_det` revision `8e0f56fb2ef86b461d99cfc7ac5c137738985f61`.
  - `PaddlePaddle/PP-OCRv6_medium_rec` revision `e5a92bcbc5cc1b494628e458d267778f0704fd7c`.
- **Smoke test (page 1 of the clean EN and VI documents):**
  - **Fails with oneDNN on:** `NotImplementedError: ConvertPirAttribute2RuntimeAttribute not support [pir::ArrayAttribute<pir::DoubleAttribute>]` (`onednn_instruction.cc`). This is Paddle 3.3.1's CPU oneDNN path with these models.
  - **With `enable_mkldnn=False`** it ran, but had not finished two full-resolution pages after more than 10 minutes. The server's load average was ~140 on 64 cores (other users' jobs: one user ~3,160% CPU), and our process got ~6 cores.
  - **Stopped** at the AI engineer's request (2026-10-20 21:21). No accuracy result yet.
- **Smoke test rerun 2026-10-21 07:16–07:27** (load ~20 on 64 cores, 16 threads, oneDNN off; script `$VINFINX_ROOT/logs/ocr_smoke.py`, log `logs/ocr-smoke-20261021.log`). A field counts as found when the OCR line at the field's box contains the printed text exactly:

  | Page 1 | Detection | Time | Fields found |
  |---|---|---|---|
  | clean EN | full resolution | 205 s | 134/145 |
  | clean VI | full resolution | 204 s | 94/145 |
  | clean EN | max side 2000 px | 130 s | 125/145 |
  | clean VI | max side 2000 px | 126 s | 90/145 |

  - **English:**
    - 10 of the 11 misses are the Unicode minus read as an ASCII hyphen (`−0.5 pp` → `-0.5 pp`); normalization handles that.
    - 1 miss is an underscore read as a space (`price_index` → `price index`).
    - All values, baselines, periods and units were read correctly.
  - **Vietnamese: the recognizer cannot output most Vietnamese letters.**
    - The character dictionary of `PP-OCRv6_medium_rec` (18,708 characters) lacks **44 of the 67** Vietnamese lowercase letters: every letter with a dot below, a hook above or stacked marks (`ảạắằẳẵặấầẩẫậẻẽẹếềểễệỉịỏọốổỗộớờởỡợủụứừửữựỳỷỹỵ`).
    - `latin_PP-OCRv5_mobile_rec` lacks 45.
    - So "Việt Nam" is read "Vit Nam", "Mức" is read "Múc", and the title loses its marks.
    - Numbers, `đpt`, periods and ASCII metric IDs are read correctly.
  - **Speed:** about 200 s per page at full resolution, 130 s downscaled, without oneDNN. That is about 10 hours for the 186 pages of the draft set. Too slow.
- **Next options, when the server is quieter:**
  1. Detect text on a downscaled page (`text_det_limit_type="max"`, `text_det_limit_side_len` ≈ 1600–2000); recognition still reads crops at full resolution.
  2. Try a Paddle 3.x release whose oneDNN path works with PP-OCRv6.
  3. Cap our threads (`cpu_threads`) so we do not add to the contention.
  4. If CPU stays too slow, the GPU build (CUDA 13.0, ~1.84 GB wheel) now that the allowance (400 GB per user) is known, once `/mnt/disk1` has physical space.

### Tesseract added (2026-10-21)

PaddleOCR cannot read Vietnamese (above), so the AI engineer approved trying Tesseract in the same environment.

```bash
conda install -y -p $VINFINX_ROOT/env-ocr --override-channels -c conda-forge "tesseract=5.5.3"   # leptonica 1.87.0
$VINFINX_ROOT/env-ocr/bin/python -m pip install "pytesseract==0.3.13"
D=$VINFINX_ROOT/models/tesseract/tessdata_best-e12c65a          # pinned tessdata_best commit
for l in eng vie; do curl -sSL -o $D/$l.traineddata \
  https://github.com/tesseract-ocr/tessdata_best/raw/e12c65a915945e4c28e237a9b52bc4a8f39a0cec/$l.traineddata; done
cp $VINFINX_ROOT/env-ocr/share/tessdata/osd.traineddata $D/    # orientation detection, from the conda package
(cd $D && sha256sum *.traineddata > TESSDATA.sha256)
```

- **Language data:** the conda package bundles 128 languages from the small "fast" set (`vie` is 0.5 MB). We use `tessdata_best` instead (`vie` 12.4 MB, `eng` 15.4 MB).
- **Hashes:** `eng` `8280aed0…`, `vie` `b6b49293…`, `osd` `9cf5d576…`.
- **Results:** about 4–6 s per page on one core, Vietnamese read correctly. Full numbers: [`OCR_RESULTS.md`](OCR_RESULTS.md).
- **Paths:** the adapter (`ai/ocr/tesseract.py`) points `pytesseract` at `env-ocr/bin/tesseract` and at the pinned `tessdata` directory, so the env does not need activating.

### PaddleOCR removed (2026-10-21)

The AI engineer chose Tesseract and asked to remove PaddleOCR. `env-ocr` was rebuilt rather than uninstalling Paddle's 70 dependencies.

```bash
cp $VINFINX_ROOT/requirements-ocr.lock $VINFINX_ROOT/logs/requirements-ocr-paddle-20261020.lock   # kept for the record
conda env remove -y -p $VINFINX_ROOT/env-ocr
conda create -y -p $VINFINX_ROOT/env-ocr --override-channels -c conda-forge python=3.12 "tesseract=5.5.3"
$VINFINX_ROOT/env-ocr/bin/python -m pip install "pytesseract==0.3.13" "pillow==12.3.0" "pydantic==2.14.1" \
  "PyYAML==6.0.3" "numpy==2.5.4"          # the main environment's versions, so ai/ code behaves the same in both
$VINFINX_ROOT/env-ocr/bin/python -m pip freeze > $VINFINX_ROOT/requirements-ocr.lock
rm -rf $VINFINX_ROOT/models/paddleocr $VINFINX_ROOT/cache/paddlex
```

The env is now 760 MB (was 1.5 GB); the `tessdata_best` models in `$VINFINX_ROOT/models/tesseract/` are unchanged.

## Research corpus storage (2026-10-21)

`/mnt/disk1` was full (230 MB free on 2026-10-21, other users), so the research corpus goes on the second disk.

| Item | Value |
|---|---|
| Disk | `/mnt/sdb` (`/dev/sdb`, ext4, 3.5 TB, **1.2 TB free**, 65% used) |
| Layout | `/mnt/sdb/shared/` is world-writable (sticky bit) with one folder per user; `shared/chuong/` is ours and already holds ~219 GB of another project's data (`dataset`, `b`, `m01`–`m03`; not VinFinx, not touched). `/mnt/sdb/archive/` is root's per-user archive from September |
| Corpus | `/mnt/sdb/shared/chuong/vinfinx/corpus/`; `vinfinx/` is mode `700` so other users cannot read or rename it |
| Link | `$VINFINX_ROOT/corpus` is a symlink to it, so code and docs keep using `$VINFINX_ROOT/corpus` |
| Not known | Whether `/mnt/sdb/shared` is ever cleaned up by the admin (the September archive suggests it is managed). Nothing there is the only copy of anything that cannot be re-downloaded: the corpus is rebuilt from its source list and hashes |

```bash
mkdir -p /mnt/sdb/shared/chuong/vinfinx/corpus
chmod 700 /mnt/sdb/shared/chuong/vinfinx
rmdir $VINFINX_ROOT/corpus                       # was empty
ln -s /mnt/sdb/shared/chuong/vinfinx/corpus $VINFINX_ROOT/corpus
```

Environments, models, fixtures and evaluation reports stay on `/mnt/disk1`.

## Source access check (2026-10-21)

The AI engineer chose **free scholarly and institutional sources only** for the source agent: OpenAlex, the World Bank Documents & Reports API, and the publication lists of the IMF, ADB, BIS, the State Bank of Viet Nam (SBV) and the statistics office. Checked from the server with `curl` (User-Agent `VinFinx-research/0.1`):

| Source | Endpoint | Result | Use |
|---|---|---|---|
| OpenAlex | `https://api.openalex.org/works?search=…&filter=is_oa:true` | 200 JSON, no key. Gives DOI, year, type, open-access status, **licence** and an open PDF URL. "monetary policy housing Vietnam", open access: 18,214 hits | Main discovery for papers, and the **metadata route for IMF, ADB and BIS** (DOI prefix `10.5089` IMF: 117 Viet Nam works; `10.22617` ADB: 93) |
| World Bank | `https://search.worldbank.org/api/v3/wds?format=json&qterm=…` | 200 JSON, no key. Gives title, date, document type, country, language, `pdfurl`. "Vietnam housing credit": 488 | Discovery; filter by document type (many hits are project documents such as resettlement plans). PDF download works |
| IMF | `www.imf.org`, `www.elibrary.imf.org` | **403 / 202 empty**: bot protection (Akamai, WAF challenge) | Not reachable by script. Discover via OpenAlex; a person downloads chosen PDFs in a browser into an inbox, then code hashes them |
| ADB | `www.adb.org` pages | **403**: Cloudflare challenge | Discover via OpenAlex; **PDFs under `/sites/default/files/` download** (200, `application/pdf`) |
| BIS | `https://www.bis.org/doclist/wppubls.rss` | 200 RSS (working papers); `robots.txt` allows | RSS feeds plus OpenAlex. No search API found |
| SBV | `https://sbv.gov.vn/` | 200; `robots.txt` allows everything | Publication pages, mostly Vietnamese. Needs a small page reader |
| Statistics office | `https://www.nso.gov.vn/en/` (the old `gso.gov.vn` no longer resolves) | 200; `robots.txt` blocks only bulk-copier user agents and `/wp-admin/` | Publication pages (WordPress) |

- **Rules:** respect `robots.txt` and rate limits, never get around bot protection, and identify the client in the User-Agent. OpenAlex asks for a contact email (`mailto=`) for its faster pool; which address to use is the AI engineer's choice.
- **Installed tools:** the main env has `requests` 2.34.2 and `httpx` 2.13.1. There was no PDF text library in either env.
- **Contact for OpenAlex** (`mailto=`, and in the User-Agent): `jimmytrn154@gmail.com`, given by the AI engineer 2026-10-21.

### PDF library (installed 2026-10-21)

The AI engineer left the choice to me. **`pypdfium2==5.14.0`** (released 2026-10-04; PDFium 156.0.8076.0), in the main env:
- BSD-3-Clause / Apache-2.0. PyMuPDF (2.0) was not chosen: it is AGPL-3.0, a problem for a bank product.
- Text per page with character boxes (the chunker needs page and character offsets), and page rendering to images, so scanned PDFs can go to Tesseract.
- Faster than the pure-Python options (`pypdf` 6.21.1, `pdfplumber` 0.11.10).

```bash
$VINFINX_ROOT/env/bin/python -m pip install --no-cache-dir "pypdfium2==5.14.0"   # no cache: /mnt/disk1 had 136 MB free
$VINFINX_ROOT/env/bin/python -m pip check                                        # no broken requirements
$VINFINX_ROOT/env/bin/python -m pip freeze > $VINFINX_ROOT/requirements.lock     # 95 packages
```

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
1. Test the prototype end to end from the Mac through the bridge (step 5). NetBird access works (2026-10-17).
2. Measure again with decision-engine-sized prompts (step 6).
