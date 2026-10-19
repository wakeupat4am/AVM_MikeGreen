# Frozen case sets

A frozen set is never edited. Changes go into a new file and version, with a new row here. `ai/tests/test_cases.py` fails if a frozen file no longer matches its recorded hashes.

| File | Version | Frozen by | Frozen at | Cases (holdout) | Content hash (`CaseSet.set_hash`, reported by the runner) | File SHA-256 |
|---|---|---|---|---|---|---|
| `adversarial_v0.yaml` | 0.1.0 | jimmytrn154 | 2026-10-18T22:55:15+07:00 | 42 (11) | `sha256:0d436a31345b0c598b6a99b4311309e1520b98182d0ff3e2722e900fbbe3926e` | `sha256:06b6b828e3213de108df46248f7265f7f621c47ba2262193b3a95c01d966ac29` |
