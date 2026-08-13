# Chat Templates: auto-detect + manual switch

Some GGUF models embed a broken or missing `tokenizer.chat_template`, which makes
them reply with gibberish or not respond at all. This project ships:

1. `llama-run.ps1` - a wrapper around `llama-server`/`llama-cli`/`llama-completion`
   that **auto-detects** the correct template by model filename, and also accepts an
   **explicit switch** to load any template the user wants.
2. `templates/auto-templates.json` - the filename-pattern -> template map.
3. `templates/*.jinja` - the fetched templates (from HuggingFace).

## Quick start

```powershell
# auto-detect (no flags needed)
powershell -File llama-run.ps1 -Tool llama-server -Model E:\OLLAMA-Models\GGUF\laguna-xs2-Q4_K_M.gguf -Args "-ngl 99 -c 4096"

# load ANY template explicitly (file)
powershell -File llama-run.ps1 -Tool llama-server -Model <model> -Template path\to\my.jinja -Args "-ngl 99"

# load a builtin llama.cpp template by name
powershell -File llama-run.ps1 -Tool llama-server -Model <model> -Template gemma -Args "-ngl 99"
```

`-Template <path-or-name>` always wins over auto-detect and over the model's own
embedded template. Any `--chat-template(-file)` already on the command line also wins.

## Why

llama.cpp normally uses the template embedded in the GGUF metadata
(`tokenizer.chat_template`). A model cloned/finetuned from another family often:

- embeds `{% include 'chat_template.jinja' %}` (an unresolvable jinja include) ->
  jinja fails, output is garbage,
- embeds nothing -> falls back to a generic template -> wrong format,
- looks like another model but needs its own template.

## How auto-detect works

1. The wrapper reads the GGUF filename.
2. It checks `templates/auto-templates.json` patterns in order.
3. First match injects `--chat-template-file <file>` (or `--chat-template <builtin>`).
4. No match -> the model's embedded template is used unchanged.

## Current map (templates/auto-templates.json)

| Filename pattern | Applied template | Source (HF) |
|------------------|------------------|-------------|
| `laguna-xs2`, `Laguna-XS.2` | `templates/laguna_xs2.jinja` | poolside/Laguna-XS-2.1 |
| `laguna-s-2.1`, `laguna_s` | `templates/laguna_s21.jinja` | poolside/Laguna-S-2.1 |
| `gemma-4-12B`, `Gemma-4-12B` | builtin `gemma` | google/gemma-4-12B-it |

## Adding a new template

1. Fetch the template from the model's HF repo:
   `tokenizer_config.json` -> `chat_template` field
   (see `bench_results/hf_template.py` / `build_template_manifest.py`).
2. Save it as `templates/<name>.jinja`.
3. Add a `{ "match": "<filename-substring>", "template": "templates/<name>.jinja" }`
   (or `"builtin": "<name>"`) entry to `templates/auto-templates.json`.
4. Keep the pattern narrow enough not to catch unrelated clones.

## Audited models (2026-08-13)

Full audit: `bench_results/templates.json` (per-model status + fetched template).

- laguna family (XS.2 / S-2.1): embedded template was a broken `{% include %}` ->
  fixed via HF templates.
- gemma-4-12B (IQ4_XS, OBLITERATED): no embedded template -> use builtin `gemma`.
- GIGABATEMAN-7B.Q2_K, Samastam-2.5B: no template on HF either (finetunes); needs
  base-model template - add pattern when a template is sourced.
- Ternary/Bonsai, Qwen3.5-35B-A3B, L3.2-8X3B-MOE: HF 404/gated; embedded qwen-style
  template present and usable.
- qwen family / GLM-4.7 / DeepSeek-Coder-V2 / ornith / VibeThinker / qwable:
  embedded templates OK, no override needed.
- mmproj (vision projector) files need no chat template.

## Direct llama.cpp flags (no wrapper)

- `--chat-template-file <file>` - load a jinja template from a file.
- `--chat-template <name>` - use a builtin template (list: llama3, gemma, chatml, ...).
- `--jinja / --no-jinja` - toggle the jinja engine.

Docs: `tools/server/README.md` (server), `--help` on any tool.
