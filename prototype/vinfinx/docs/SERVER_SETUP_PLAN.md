# Qwen server setup plan

*Status: draft, 2026-10-14. Target: a personal GPU server replacing `cecs-server1` for serving Qwen3.5-9B. Nothing has been set up yet.*

## Goal

Serve Qwen3.5-9B (the generation model for both the mapping agent and the decision engine) on our own server, reachable from the prototype only through an SSH tunnel, with a pinned configuration that all evaluation results are reported on.

Local serving on the Mac (Apple M4 Pro, 24 GB) was considered: it only fits a quantized copy, which behaves differently from the bf16 reference. It can be used for development, not for reported results.

## What is tied to `cecs-server1` today

| File | Assumption |
|---|---|
| [`serve-qwen.sh`](../serve-qwen.sh) | Root `/home/ubuntu/llms-conformity-rq1`, venv at `server_experiments/.venv`, extra library folder `multimodal_rq1/.deps-qwen35` on `PYTHONPATH`, GPUs `0,1` |
| [`qwen_api.py`](../qwen_api.py) | Model at `.../multimodal_rq1/models/Qwen3.5-9B`; hardcoded 21 GiB limit per GPU (implies two ~24 GB GPUs) |
| [`start-qwen-bridge.sh`](../start-qwen-bridge.sh) | SSH host alias `cecs-server1`, port 8061 |
| Launch command | `serve-qwen.sh` runs `vinfinx_qwen_api:app`, but the file is named `qwen_api.py` |

The `.deps-qwen35` folder suggests `cecs-server1` needed a special `transformers` build for Qwen3.5. Find out which version it is before setting up the new server; it is the most likely thing to break.

## Steps

### 1. Prepare the server
- Linux with an NVIDIA driver and CUDA.
- A dedicated Python 3.11+ environment.
- Install PyTorch, a `transformers` version that supports Qwen3.5 (match `cecs-server1`), FastAPI and uvicorn.

### 2. Get the weights
- Download Qwen3.5-9B from Hugging Face, or copy it from `cecs-server1`.
- Record the exact revision or commit hash for the version log (handoff Q8).

### 3. Make the scripts portable
- Read model path, GPU list, per-GPU memory limit and port from environment variables (the memory limit is currently hardcoded in `qwen_api.py`).
- Fix the module-name mismatch between `serve-qwen.sh` and `qwen_api.py`.
- Make the SSH host alias in `start-qwen-bridge.sh` configurable and point it at the new server.

### 4. Bring it up and check it
- Start the service bound to `127.0.0.1` on the server. Reach it through the SSH tunnel only; never open the port publicly.
- Check `GET /vinfinx-api/health`, then send one chat request from the prototype UI.

### 5. Pin the reference setup
Record GPU model, driver, CUDA, PyTorch and `transformers` versions, model revision and precision (bf16). This is the reference configuration for all baseline and frozen-test-set results.

## Optional upgrade: vLLM

The current adapter uses the `transformers` pipeline behind a global lock, so it serves one request at a time and cannot force output into a JSON schema. A serving engine such as vLLM would:
- batch requests, which matters when running test sets of dozens or hundreds of cases;
- support JSON-schema-constrained output, so the model cannot emit an action outside the allowed enum (handoff §6 "structured outputs", enforced at generation time before the verifier).

Check that the vLLM version in use supports the Qwen3.5 architecture. If not, start with the existing `transformers` adapter and switch later.

## Open questions about the server

1. **GPU:** model, memory and count. bf16 Qwen3.5-9B needs about 19 GB for weights plus room for context. One 24 GB card is tight; one 40–80 GB card is comfortable; two 24 GB cards match the current setup.
2. **OS and CUDA version.**
3. **Access:** direct SSH from the Mac, or through a VPN as with `cecs-server1`?
4. **Other GPU users:** is anything else running on the GPUs?

## Next

Once the questions above are answered: write the concrete setup commands and the portable versions of the three scripts.
