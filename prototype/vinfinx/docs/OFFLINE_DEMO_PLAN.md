# Offline research half in the live demo

*Status: **noted 2026-10-24, not started** (user: note it in the progress and a plan; build later). Not a proposal for
approval yet: the choices below are to be confirmed before building.*

## The gap

The site shows every offline step's results and lets a person do the human steps live (sign the source list, review gold
labels, validate a mapping set), but the offline agents can only be started from the server's command line. The online half
runs live (Market signals → Run pipeline, ~25 s). To show the full pipeline in the demo:

| Offline step | Shown in the site | Started from the site |
|---|---|---|
| Source agent | Evidence sources → Source list | No: a run takes 10+ minutes (candidates, downloads, Qwen screening) |
| Source-list sign-off | Yes (live) | Yes |
| Corpus and index | Corpus tab (versions, hashes, metrics, search) | No |
| Gold-label review | Gold labels tab | Yes |
| Mapping agent and support check | Mappings tab (outcomes, quotes, verdicts) | **No: to add** |
| Mapping-set validation | Yes (live) | Yes |

## Proposed additions (to confirm)

1. **"Run mapping agent" button** in the Mappings tab: starts a run on the newest frozen corpus as a background job (one at a
   time, like pipeline runs), shows each pair as it finishes, needs Qwen running and the OpenAI key (support check). About
   3 minutes.
2. **Pipeline overview** in the Authenticator view: one strip for each half (papers → source list → corpus → mappings;
   bulletin → evidence → policy → decision → verifier → review), each stage with its latest result, version, status and a
   link to its tab, so an audience can follow the whole story on one page.
3. **Source agent stays pre-run** (too long for a live demo), as agreed for step 5: run it before the demo and sign the new list
   live as `source_list/v0.2`.
