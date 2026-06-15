# remotelocalllm

DISCLAIMER: Completely created with claude code

A small CLI client that chats with your **remote Ollama server** and feeds it **context from the internet and your
local machine** — search results, fetched web pages, piped command output, and
local files. Inference runs on the GPU box; all fetching happens locally on the
laptop, then gets handed to the model as context (RAG-style).

## Setup

```bash
cd remotelocalllm
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Point the client at your desktop's tailnet (MagicDNS) name:

```bash
export OLLAMA_HOST=http://<your-desktop-name>:11434
export OLLAMA_MODEL=qwen3:14b    # optional; this is the default
export OLLAMA_CTX=16384          # optional; context window in tokens (default 16384)
```

> The desktop must be awake and serving (Part 1 setup). Quick check:
> `curl $OLLAMA_HOST/api/version`

## Usage

```bash
# One-shot. The model decides when to search/fetch (agentic tool calling).
python chat.py "what changed in the latest Tailscale release?"

# Interactive REPL (multi-turn, keeps history).
python chat.py

# Pipe context in from stdin.
cat error.log | python chat.py "explain this stack trace"
git diff | python chat.py "review this change"

# Pull in specific pages and/or local files as context.
python chat.py --url https://docs.example.com/api --file notes.md "draft a client"

# Search-first RAG: always DuckDuckGo the prompt and fetch the top pages
# before answering. Works even with models that can't tool-call.
python chat.py --search "qwen3 14b vram usage on a 3080 Ti"

# Plain chat, no internet at all.
python chat.py --no-tools "rewrite this paragraph: ..."
```

### How retrieval works

| Mode | Flag | Behavior |
| --- | --- | --- |
| Tool calling | *(default)* | Model emits `web_search` / `fetch_url` calls when it needs them; the client runs them and loops results back, up to 6 rounds. No wasted searches on questions it can answer directly. |
| Search-first | `--search` | The model first **plans** the search (decides whether one is needed and writes 1–2 *optimized* queries via structured JSON), then the client searches, auto-fetches the top pages, and answers. Predictable, and works even on models that can't tool-call. |
| None | `--no-tools` | Plain chat. Local context flags (`--file`, `--url`, stdin) still apply. |

Tool/search activity is printed to **stderr** (`[tool] ...`, `[search] ...`,
`[fetch] ...`) so you can see what it's doing; the model's answer streams to
**stdout**, so you can still pipe the answer somewhere clean.

## Flags

| Flag | Purpose |
| --- | --- |
| `--host URL` | Ollama endpoint (default `$OLLAMA_HOST` or `http://localhost:11434`). |
| `--model NAME` | Model to use (default `$OLLAMA_MODEL` or `qwen3:14b`). |
| `--ctx N` | Context window in tokens (default `$OLLAMA_CTX` or 16384; `qwen3:14b` supports up to 40960). |
| `--search` | Search-first RAG pipeline. |
| `--no-tools` | Disable internet tools. |
| `--url URL` | Fetch a page as context (repeatable). |
| `--file PATH` | Include a local file as context (repeatable). |
| `--no-compact` | Disable auto-summarization of old REPL history (on by default). |

## Notes

- A 14B model is solid for single-shot "search → read → answer" but gets less
  reliable over long autonomous tool chains — that's why the loop is capped.
- Fetched pages are stripped to text and truncated (~8k chars each) to protect
  the model's context window and your 12 GB of VRAM.
- No API keys needed — search uses DuckDuckGo via the `ddgs` package.
- **Context window:** Ollama defaults to a tiny 4096-token window regardless of
  what the model supports, which silently truncates long RAG payloads. This
  client overrides it to 16384 by default (`--ctx` to change). This fits fully
  on a 12 GB card **only with a quantized KV cache** on the server
  (`OLLAMA_FLASH_ATTENTION=1`, `OLLAMA_KV_CACHE_TYPE=q8_0`); at the default f16
  cache, drop to `--ctx 8192`. Beyond ~20K the KV cache spills into system RAM
  and slows generation down.
- **REPL compaction:** in interactive mode, once the conversation passes ~70% of
  the window, older turns are automatically summarized into a compact note (recent
  turns kept verbatim) so long sessions don't overflow. Disable with `--no-compact`.
- **Freshness:** answer quality is limited by what search surfaces. Very recent
  pages (e.g. a patch posted yesterday) may not be indexed yet, and the model
  may fall back to stale training data — prefer `--search`, or pass the primary
  source directly with `--url` when you know it.
- **Pasted URLs:** any `http(s)` URL in your prompt is fetched automatically and
  added as context (same as `--url`), so the model reads the page instead of
  guessing from the link text. For GitHub repos the README usually comes through;
  a very large repo page may truncate it (paste the raw README URL if so).
