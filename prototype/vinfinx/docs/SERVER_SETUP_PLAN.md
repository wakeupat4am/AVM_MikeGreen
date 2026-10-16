# Qwen server setup plan

*Status: draft, updated 2026-10-16 with the new server's GPU facts. Target: our own shared GPU server (not `cecs-server1`) serving Qwen3.5-9B. Nothing has been set up yet.*

## Goal

Serve Qwen3.5-9B (the generation model for both the mapping agent and the decision engine) on our own server, reachable from the prototype only through an SSH tunnel, with a pinned configuration that all evaluation results are reported on. Swapping to another model later must be a config change, not a code change.

Local serving on the Mac (Apple M4 Pro, 24 GB) was considered: it only fits a quantized copy, which behaves differently from the bf16 reference. It can be used for development, not for reported results.

## The server

From `nvidia-smi` (snapshot, 2026-10-16):

- **8 × NVIDIA GeForce RTX 3090, 24 GB each** (192 GB total), Ampere architecture.
- **Driver 595.91.07, CUDA 13.2.** PyTorch and vLLM builds for CUDA 12.x run on this driver.
- **Shared and busy.** Every GPU already has memory in use by other processes:

| GPU | Used | Free (approx.) | Utilization |
|---|---|---|---|
| 0 | 3.5 GB | 21 GB | 73% |
| 1 | 7.5 GB | 17 GB | 100% |
| 2 | 13.6 GB | 11 GB | 0% |
| 3 | 13.6 GB | 11 GB | 0% |
| 4 | 13.5 GB | 11 GB | 0% |
| 5 | 13.3 GB | 11 GB | 0% |
| 6 | 2.2 GB | 22 GB | 10% |
| 7 | 2.2 GB | 22 GB | 91% |

Implications:
- Qwen3.5-9B in bf16 needs about 19 GB for weights plus room for the context (KV) cache. No single GPU is currently free and idle.
- It fits either on **one GPU with ~22 GB free** (tight, short context) or **split across two GPUs** (tensor parallel 2, comfortable, matches the old `cecs-server1` setup).
- GPUs 2–5 hold ~13.5 GB each while idle; someone has memory reserved there. Which GPUs we may use must be **agreed with the other users** before starting.
- Ampere has no native FP8. Use bf16 (reference) or 4-bit AWQ/GPTQ if a larger model is tried later.
- The cards most likely connect over PCIe, so multi-GPU serving is slower than on data-centre cards.

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

A larger model would help most for the **mapping agent** (reading real research papers, some in Vietnamese, and quoting exactly), less for the decision engine. Check which model sizes are actually published, and supported by the installed vLLM, before choosing.

## Serving architecture

```
Mac (browser, prototype UI)
   │  http://127.0.0.1:8061/vinfinx-api/...
   ▼
SSH tunnel  (start-qwen-bridge.sh, -L 8061:127.0.0.1:8061)
   ▼
Server, loopback only
   ├─ qwen_api.py adapter  127.0.0.1:8061   origin check, message limits, system prompt
   │                                          (later: policy, verifier, audit)
   └─ vLLM                 127.0.0.1:8000   OpenAI-compatible API, Qwen3.5-9B bf16
```

- **vLLM is the primary serving engine** (no longer optional). It batches requests, which matters for running test sets, and supports JSON-schema-constrained output (`response_format` with a JSON schema), so the model cannot emit an action outside the allowed enum. This is handoff §6 "structured outputs", enforced before the verifier.
- **`qwen_api.py` becomes a thin adapter** in front of vLLM instead of loading the model itself. The UI keeps working unchanged, and the adapter is where the evidence bundle, verifier and audit log will later plug in.
- **Fallback:** if the installed vLLM does not support the Qwen3.5 architecture, keep the current `transformers` adapter (one request at a time) until it does.

## What has to change from the `cecs-server1` setup

| File | `cecs-server1` assumption | Change |
|---|---|---|
| [`serve-qwen.sh`](../serve-qwen.sh) | Root `/home/ubuntu/llms-conformity-rq1`, venv at `server_experiments/.venv`, `.deps-qwen35` on `PYTHONPATH`, GPUs `0,1` | Start vLLM, then the adapter; all paths, GPUs, ports from environment variables |
| [`qwen_api.py`](../qwen_api.py) | Loads the model with `transformers`; model path under `multimodal_rq1/`; hardcoded 21 GiB per GPU | Forward requests to vLLM's `/v1/chat/completions`; keep origin check and limits; no GPU code |
| [`start-qwen-bridge.sh`](../start-qwen-bridge.sh) | SSH host alias `cecs-server1` | Host alias from an environment variable (for example `VINFINX_SSH_HOST`) |
| Launch command | Runs `vinfinx_qwen_api:app`, but the file is `qwen_api.py` | Run `qwen_api:app` |

The `.deps-qwen35` folder suggests `cecs-server1` needed a special `transformers` build for Qwen3.5. If that server is still accessible, record which version it is; it shows how new the libraries must be.

## Steps

### 1. Agree on GPUs and check the server
- Agree with the other users which GPUs we may claim (two GPUs for tensor parallel 2, or one GPU with ~22 GB free).
- Record OS, Python version and disk space for the weights (~20 GB for 9B).
- See who holds GPU memory:
  ```bash
  nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv
  nvidia-smi --query-compute-apps=pid,gpu_uuid,used_memory --format=csv
  ```

### 2. Prepare the environment
- A dedicated Python 3.11+ environment (for example with `uv` or `venv`), not shared with other users' projects.
- Install vLLM (pulls a matching PyTorch), FastAPI, uvicorn and an HTTP client for the adapter.
- Confirm the installed vLLM supports the Qwen3.5 architecture.

### 3. Get the weights
- Download Qwen3.5-9B from Hugging Face into a path we own.
- Record the exact revision or commit hash for the version log (handoff Q8).

### 4. Start vLLM (loopback only)
Two-GPU example; the GPU indices are placeholders until agreed:

```bash
CUDA_VISIBLE_DEVICES=6,7 vllm serve "$VINFINX_QWEN_PATH" \
  --served-model-name qwen3.5-9b \
  --host 127.0.0.1 --port 8000 \
  --dtype bfloat16 \
  --tensor-parallel-size 2 \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.80
```

- `--gpu-memory-utilization` is a fraction of each GPU's total memory. On shared GPUs it must fit inside what is actually free, or vLLM refuses to start; lower it, or lower `--max-model-len`, if needed.
- Single-GPU variant: one GPU with ~22 GB free, `--tensor-parallel-size 1`, a shorter `--max-model-len` (for example 4096). Tight; expect to tune.
- Pin the sampling settings used for evaluation (greedy, fixed seed) in the adapter, as the current adapter does.

### 5. Start the adapter and check it
- Run the adapter on `127.0.0.1:8061`, forwarding to `127.0.0.1:8000`.
- On the server: `curl http://127.0.0.1:8000/health` and `curl http://127.0.0.1:8061/vinfinx-api/health`.
- On the Mac: run `start-qwen-bridge.sh`, open the prototype, send one chat message.
- Never bind either service to a public interface.

### 6. Keep it running
- Run both processes under something that survives logout (`tmux`, or a user-level `systemd` service).
- Add a short runbook to this file: how to start, stop and check the service, and which GPUs it uses.

### 7. Pin the reference setup
Record GPU indices, driver, CUDA, PyTorch, vLLM and `transformers` versions, model revision, precision (bf16), `--max-model-len`, tensor-parallel size and sampling settings. This is the reference configuration for all baseline and frozen-test-set results.

## Open questions

1. **Which GPUs** may we claim, and for how long? Who owns the memory held on GPUs 2–5?
2. **OS and Python version** on the server.
3. **Access:** direct SSH from the Mac, or through a VPN? What SSH host alias to use?
4. **Disk:** where to store the weights and environments.

## Next

Once the questions above are answered: write the portable `serve-qwen.sh`, the vLLM-backed `qwen_api.py` adapter and the configurable `start-qwen-bridge.sh`, then add the runbook and pinned versions here.
