#!/usr/bin/env python3
"""Internet-augmented CLI chat client for a remote Ollama server.

The model runs on the remote GPU box (over your tailnet); this script runs on
your laptop and does the actual internet fetching locally, feeding the results
back to the model as context -- "RAG-ish".

Two retrieval modes:
  * default      -- agentic tool calling: the model decides when to call
                    web_search / fetch_url, and this client executes them.
  * --search     -- search-first: always DuckDuckGo the prompt, fetch the top
                    pages, prepend them as context, then a single plain chat.

Extra context can be injected from piped stdin, --file, and --url.

Examples:
    python chat.py "what changed in the latest tailscale release?"
    python chat.py                          # interactive REPL
    cat error.log | python chat.py "explain this stack trace"
    python chat.py --url https://docs.foo/api --file notes.md "draft a client"
    python chat.py --search "qwen3 14b vram usage"
    python chat.py --ctx 16384 --search "long question needing several pages"
    python chat.py --no-tools "just chat, no internet"

Config via env: OLLAMA_HOST (default http://localhost:11434), OLLAMA_MODEL,
OLLAMA_CTX (context window in tokens).
"""

import argparse
import json
import os
import re
import sys
from datetime import date

import requests
from bs4 import BeautifulSoup

try:  # the package was renamed from duckduckgo_search to ddgs
    from ddgs import DDGS
except ImportError:  # pragma: no cover - fallback for older installs
    from duckduckgo_search import DDGS

DEFAULT_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
DEFAULT_MODEL = os.environ.get("OLLAMA_MODEL", "qwen3:14b")
DEFAULT_CTX = int(os.environ.get("OLLAMA_CTX", "16384"))  # fits a 12GB card with q8_0 KV cache; max 40960
TODAY = date.today().isoformat()  # injected into prompts so the model knows "now"

MAX_TOOL_ROUNDS = 6        # cap on agentic search->read->answer loops
FETCH_CHAR_LIMIT = 8000    # per-page text cap (protect the context window)
CONTEXT_CHAR_LIMIT = 12000  # per stdin/file context-block cap
SEARCH_RESULTS = 6         # results per web_search
SEARCH_FETCH_TOP = 3       # pages to auto-fetch in --search mode
HTTP_TIMEOUT = 300
COMPACT_TRIGGER = 0.70     # compact REPL history once it passes this fraction of num_ctx
KEEP_RECENT_MSGS = 4       # most-recent messages always kept verbatim when compacting

SYSTEM_PROMPT = (
    "You are a helpful assistant running locally with live internet access via tools. "
    "Your built-in knowledge has a training cutoff and is STALE for anything current. "
    "You MUST call web_search before answering any question about: latest/current/newest "
    "versions or releases, dates, prices, news, recent events, or any fact that changes "
    "over time. Do NOT answer such questions from memory -- your memory is likely wrong. "
    "After searching, if the snippets are not enough, call fetch_url on the most promising "
    "result to read the page. Prefer primary/official sources over secondhand aggregators, "
    "and prefer the result with the most recent date. Note the publication date of what you "
    "read and tell the user how fresh it is. Always cite the URLs you relied on. "
    "Only skip the tools for timeless questions (math, definitions, code, reasoning). "
    "Be concise."
)


def _model_extra_system(model):
    """Per-model system-prompt additions. The abliterated qwen3 build tends to
    reason in Chinese, so force English for it."""
    if "abliterated" in (model or "").lower():
        return (" Always reason and respond in English only. Do not output any "
                "Chinese characters anywhere, including in your reasoning.")
    return ""


# --------------------------------------------------------------------------- #
# Tools (executed locally on this machine)
# --------------------------------------------------------------------------- #

def _ddg_search(query, max_results=SEARCH_RESULTS, timelimit=None):
    """timelimit: None, or 'd'/'w'/'m'/'y' to restrict to the last day/week/month/year."""
    with DDGS() as ddgs:
        results = list(ddgs.text(query, max_results=max_results, timelimit=timelimit))
    # Drop tracking/redirect hrefs (e.g. "/clev?...") that aren't real URLs.
    return [r for r in results if str(r.get("href", "")).startswith("http")]


def _format_results(results):
    if not results:
        return "No results found."
    lines = []
    for r in results:
        lines.append(
            f"- {r.get('title', '').strip()}\n"
            f"  {r.get('href', '')}\n"
            f"  {r.get('body', '').strip()}"
        )
    return "\n".join(lines)


def web_search(query):
    """Tool: search the web, return titles/URLs/snippets as text."""
    return _format_results(_ddg_search(query))


def fetch_url(url):
    """Tool: fetch a page and return its readable text (boilerplate stripped)."""
    resp = requests.get(
        url,
        timeout=15,
        headers={"User-Agent": "Mozilla/5.0 (compatible; remotelocalllm/1.0)"},
    )
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")
    for tag in soup(
        ["script", "style", "nav", "header", "footer", "aside",
         "form", "button", "svg", "noscript", "menu"]
    ):
        tag.decompose()
    # Prefer the main content region so we don't spend the budget on site chrome.
    root = soup.find("main") or soup.find("article") or soup.body or soup
    text = " ".join(root.get_text(separator=" ").split())
    if len(text) > FETCH_CHAR_LIMIT:
        text = text[:FETCH_CHAR_LIMIT] + " [truncated]"
    return text or "Page contained no readable text."


TOOL_FUNCTIONS = {"web_search": web_search, "fetch_url": fetch_url}

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the web. Returns a list of titles, URLs, and snippets.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "The search query."},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "fetch_url",
            "description": "Fetch a web page and return its readable text content.",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "The full URL to fetch."},
                },
                "required": ["url"],
            },
        },
    },
]


def run_tool(call):
    spec = call.get("function", {})
    name = spec.get("name", "")
    raw_args = spec.get("arguments") or {}
    if isinstance(raw_args, str):
        try:
            raw_args = json.loads(raw_args)
        except json.JSONDecodeError:
            raw_args = {}
    fn = TOOL_FUNCTIONS.get(name)
    if fn is None:
        return f"Unknown tool: {name}"
    preview = ", ".join(f"{k}={v!r}" for k, v in raw_args.items())
    print(f"  [tool] {name}({preview})", file=sys.stderr)
    try:
        return fn(**raw_args)
    except Exception as exc:  # surface failures back to the model, don't crash
        return f"Tool error: {exc}"


# --------------------------------------------------------------------------- #
# Ollama chat (streaming) + agentic tool loop
# --------------------------------------------------------------------------- #

# CJK + fullwidth/kana ranges, stripped from output for models that leak Chinese
# (the abliterated qwen3 falls into Chinese under degenerate/repetitive output).
CJK_RE = re.compile(r'[　-〿぀-ヿ㐀-䶿一-鿿豈-﫿＀-￯]')


def stream_chat(host, model, messages, tools=None, num_ctx=DEFAULT_CTX):
    """One streaming chat call. Prints content as it arrives; returns
    (full_content, tool_calls)."""
    payload = {
        "model": model,
        "messages": messages,
        "stream": True,
        "options": {"num_ctx": num_ctx},
    }
    if tools:
        payload["tools"] = tools
    resp = requests.post(
        f"{host}/api/chat", json=payload, stream=True, timeout=HTTP_TIMEOUT
    )
    resp.raise_for_status()

    strip_cjk = "abliterated" in (model or "").lower()
    parts, tool_calls = [], []
    for line in resp.iter_lines():
        if not line:
            continue
        chunk = json.loads(line)
        msg = chunk.get("message") or {}
        piece = msg.get("content")
        if piece:
            if strip_cjk:
                piece = CJK_RE.sub("", piece)
            if piece:
                sys.stdout.write(piece)
                sys.stdout.flush()
                parts.append(piece)
        if msg.get("tool_calls"):
            tool_calls.extend(msg["tool_calls"])
        if chunk.get("done"):
            break
    return "".join(parts), tool_calls


def run_turn(host, model, messages, use_tools, num_ctx=DEFAULT_CTX):
    """Drive one user turn to completion, looping over any tool calls."""
    tools = TOOLS if use_tools else None
    for _ in range(MAX_TOOL_ROUNDS):
        content, tool_calls = stream_chat(host, model, messages, tools, num_ctx)
        assistant_msg = {"role": "assistant", "content": content}
        if tool_calls:
            assistant_msg["tool_calls"] = tool_calls
        messages.append(assistant_msg)
        if content:
            print()  # terminate the streamed line

        if not tool_calls:
            return
        for call in tool_calls:
            messages.append({"role": "tool", "content": run_tool(call)})

    print("\n(stopped: too many tool rounds without a final answer)", file=sys.stderr)


# --------------------------------------------------------------------------- #
# History compaction (summarize old turns to stay within the context window)
# --------------------------------------------------------------------------- #

SUMMARIZER_SYSTEM = (
    "You compress a chat transcript into a concise summary that preserves every "
    "fact, decision, name, number, file path, code snippet, and unresolved question "
    "needed to continue the conversation seamlessly. Be terse but complete. "
    "Output only the summary."
)


def _est_tokens(messages):
    """Rough token estimate (~4 chars/token) over message content."""
    return sum(len(str(m.get("content") or "")) for m in messages) // 4


def _summarize(host, model, transcript, num_ctx):
    # Size this call's window to fit the whole transcript (capped at the model's
    # 40960 max), since the text being compacted can exceed the chat's working
    # window. think=False: a summary needs no reasoning trace, and it keeps the
    # output in `content` instead of the `thinking` field.
    summ_ctx = max(num_ctx, min(40960, len(transcript) // 4 + 1024))
    payload = {
        "model": model,
        "stream": False,
        "think": False,
        "options": {"num_ctx": summ_ctx, "temperature": 0.2},
        "messages": [
            {"role": "system", "content": SUMMARIZER_SYSTEM + _model_extra_system(model)},
            {"role": "user", "content": transcript},
        ],
    }
    resp = requests.post(f"{host}/api/chat", json=payload, timeout=HTTP_TIMEOUT)
    resp.raise_for_status()
    return resp.json()["message"]["content"].strip()


def compact_history(host, model, messages, num_ctx):
    """If the running history is large, summarize the older turns into one compact
    note folded into the oldest kept turn, and keep recent turns verbatim. The
    cutoff is aligned to a user-message boundary so assistant/tool pairs are never
    split. Returns a (possibly new) message list."""
    if _est_tokens(messages) < COMPACT_TRIGGER * num_ctx or len(messages) <= KEEP_RECENT_MSGS + 1:
        return messages

    system = messages[0]
    cutoff = len(messages) - KEEP_RECENT_MSGS
    while cutoff > 1 and messages[cutoff].get("role") != "user":
        cutoff -= 1
    if cutoff <= 1:
        return messages  # no safe boundary to compact at yet

    older, tail = messages[1:cutoff], list(messages[cutoff:])
    lines = []
    for m in older:
        content = (m.get("content") or "").strip()
        if m.get("tool_calls"):
            names = ", ".join(c.get("function", {}).get("name", "") for c in m["tool_calls"])
            content = f"(called tools: {names}) {content}".strip()
        if content:
            lines.append(f"{m.get('role')}: {content}")

    print(f"  [compact] summarizing {len(older)} older messages (~{_est_tokens(older)} tok)",
          file=sys.stderr)
    try:
        summary = _summarize(host, model, "\n".join(lines), num_ctx)
    except requests.RequestException as exc:
        print(f"  [compact] failed ({exc}); keeping full history", file=sys.stderr)
        return messages

    tail[0] = {
        **tail[0],
        "content": (
            f"[Summary of earlier conversation:]\n{summary}\n\n"
            f"[Current message:]\n{tail[0].get('content') or ''}"
        ),
    }
    return [system, *tail]


# --------------------------------------------------------------------------- #
# Context assembly (stdin / --file / --url / --search)
# --------------------------------------------------------------------------- #

def _truncate(text, limit=CONTEXT_CHAR_LIMIT):
    text = text.strip()
    if len(text) > limit:
        return text[:limit] + " [truncated]"
    return text


def _context_budget_chars(num_ctx):
    """Roughly how many chars of injected context fit, reserving ~45% of the
    window for the system prompt, the question, and the model's answer
    (~4 chars per token)."""
    return int(num_ctx * 4 * 0.55)


def format_context_block(label, text):
    return f"--- context: {label} ---\n{text}\n--- end context ---"


def gather_static_context(args):
    """Context blocks from piped stdin, --file, and --url (no search)."""
    blocks = []
    if not sys.stdin.isatty():
        data = sys.stdin.read().strip()
        if data:
            blocks.append(format_context_block("stdin", _truncate(data)))
    for path in args.file or []:
        try:
            with open(path, encoding="utf-8", errors="replace") as fh:
                blocks.append(format_context_block(path, _truncate(fh.read())))
        except OSError as exc:
            print(f"warning: could not read {path}: {exc}", file=sys.stderr)
    for url in args.url or []:
        print(f"  [fetch] {url}", file=sys.stderr)
        try:
            blocks.append(format_context_block(url, fetch_url(url)))
        except Exception as exc:
            print(f"warning: could not fetch {url}: {exc}", file=sys.stderr)
    return blocks


PLANNER_SYSTEM = (
    "You turn a user's question into a web-search plan. Decide whether answering "
    "it well requires a CURRENT web search -- it does for anything time-sensitive, "
    "recent, version/price/news-related, or that you are not confident about. "
    "It does NOT for timeless questions (math, definitions, code, reasoning). "
    "If a search is needed, write 1-2 focused search queries using concise keywords "
    "(and operators like site: or quotes when they help). Do not hard-code a year "
    "in a query unless the user specified one. Respond with JSON only."
)

# Ollama structured-output schema -> forces valid, parseable JSON from the model.
PLAN_SCHEMA = {
    "type": "object",
    "properties": {
        "needs_search": {"type": "boolean"},
        "queries": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["needs_search", "queries"],
}


def plan_search(host, model, question, num_ctx):
    """Ask the model whether to search and, if so, for optimized queries.
    Returns (needs_search, queries). Fails open to a raw-question search."""
    payload = {
        "model": model,
        "stream": False,
        "think": False,
        "format": PLAN_SCHEMA,
        "options": {"num_ctx": num_ctx, "temperature": 0.2},
        "messages": [
            {"role": "system", "content": f"Today is {TODAY}. " + PLANNER_SYSTEM + _model_extra_system(model)},
            {"role": "user", "content": question},
        ],
    }
    try:
        resp = requests.post(f"{host}/api/chat", json=payload, timeout=HTTP_TIMEOUT)
        resp.raise_for_status()
        plan = json.loads(resp.json()["message"]["content"])
        needs = bool(plan.get("needs_search", True))
        queries = [q for q in (plan.get("queries") or []) if q.strip()][:3]
        return needs, (queries or [question])
    except (requests.RequestException, json.JSONDecodeError, KeyError, TypeError) as exc:
        print(f"  [plan] planner failed ({exc}); searching with the raw question", file=sys.stderr)
        return True, [question]


URL_RE = re.compile(r'https?://\S+')


def extract_urls(text):
    """Pull http(s) URLs out of free text, trimming trailing punctuation."""
    urls = []
    for raw in URL_RE.findall(text or ""):
        u = raw.rstrip('.,;:!?\'")]>')
        if u and u not in urls:
            urls.append(u)
    return urls


def build_user_message(prompt_text, static_blocks, search, host, model, num_ctx=DEFAULT_CTX):
    """Assemble the user message: optional search results + context + prompt,
    trimmed so the whole payload fits the model's context window. In --search
    mode the model first plans whether to search and writes optimized queries."""
    context_parts = []
    if search and prompt_text:
        needs_search, queries = plan_search(host, model, prompt_text, num_ctx)
        if not needs_search:
            print("  [plan] model judged no web search needed", file=sys.stderr)
        else:
            print(f"  [plan] optimized queries: {queries}", file=sys.stderr)
            results, seen = [], set()
            for q in queries:
                for r in _ddg_search(q):
                    href = r.get("href")
                    if href and href not in seen:
                        seen.add(href)
                        results.append(r)
            context_parts.append(format_context_block("web search results", _format_results(results)))
            for r in results[:SEARCH_FETCH_TOP]:
                url = r.get("href")
                if not url:
                    continue
                print(f"  [fetch] {url}", file=sys.stderr)
                try:
                    context_parts.append(format_context_block(url, fetch_url(url)))
                except Exception as exc:
                    print(f"warning: could not fetch {url}: {exc}", file=sys.stderr)
    # Auto-fetch any URLs the user pasted into the prompt, so the model reads the
    # actual page instead of guessing from the link text.
    if prompt_text:
        already = "\n".join(static_blocks)
        for url in extract_urls(prompt_text):
            if url in already:
                continue  # already provided via --url
            print(f"  [fetch] {url} (from prompt)", file=sys.stderr)
            try:
                context_parts.append(format_context_block(url, fetch_url(url)))
            except Exception as exc:
                print(f"warning: could not fetch {url}: {exc}", file=sys.stderr)

    context_parts.extend(static_blocks)
    context_text = "\n\n".join(context_parts)

    budget = _context_budget_chars(num_ctx)
    if len(context_text) > budget:
        context_text = context_text[:budget] + "\n[context truncated to fit the model's context window]"
        print(f"  [note] trimmed injected context to ~{budget} chars to fit num_ctx={num_ctx}",
              file=sys.stderr)

    final_prompt = prompt_text or "Please review the provided context above and summarize it."
    return f"{context_text}\n\n{final_prompt}" if context_text else final_prompt


# --------------------------------------------------------------------------- #
# Entry points
# --------------------------------------------------------------------------- #

def repl(host, model, messages, static_blocks, use_tools, search, num_ctx=DEFAULT_CTX, compact=True):
    print(f"Connected to {host} ({model}, ctx={num_ctx}). Ctrl-D or 'exit' to quit.")
    first = True
    while True:
        try:
            user = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not user or user.lower() in {"exit", "quit"}:
            break
        content = build_user_message(user, static_blocks if first else [], search, host, model, num_ctx)
        first = False
        messages.append({"role": "user", "content": content})
        if compact:
            messages = compact_history(host, model, messages, num_ctx)
        print()
        try:
            run_turn(host, model, messages, use_tools, num_ctx)
        except requests.RequestException as exc:
            print(f"\nrequest failed: {exc}", file=sys.stderr)


def main():
    parser = argparse.ArgumentParser(
        description="Chat with a remote Ollama model that can pull in internet + local context.",
    )
    parser.add_argument("prompt", nargs="*", help="One-shot prompt; omit for interactive REPL.")
    parser.add_argument("--host", default=DEFAULT_HOST, help=f"Ollama URL (default: {DEFAULT_HOST}).")
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"Model name (default: {DEFAULT_MODEL}).")
    parser.add_argument("--ctx", type=int, default=DEFAULT_CTX,
                        help=f"Context window in tokens (default: {DEFAULT_CTX}; qwen3:14b supports up to 40960).")
    parser.add_argument("--search", action="store_true",
                        help="Search-first RAG: DuckDuckGo the prompt and prepend results before answering.")
    parser.add_argument("--no-tools", action="store_true",
                        help="Plain chat with no internet tools (context flags still apply).")
    parser.add_argument("--no-compact", dest="compact", action="store_false",
                        help="Disable automatic summarization of old REPL history.")
    parser.add_argument("--url", action="append", metavar="URL",
                        help="Fetch this URL and include its text as context (repeatable).")
    parser.add_argument("--file", action="append", metavar="PATH",
                        help="Include this local file as context (repeatable).")
    args = parser.parse_args()

    # In --search mode we feed results in directly, so the agentic tools are off.
    use_tools = not args.no_tools and not args.search

    static_blocks = gather_static_context(args)
    messages = [{"role": "system",
                 "content": f"Today is {TODAY}. " + SYSTEM_PROMPT + _model_extra_system(args.model)}]
    prompt_text = " ".join(args.prompt).strip()

    interactive = not prompt_text and sys.stdin.isatty()

    try:
        if interactive:
            repl(args.host, args.model, messages, static_blocks, use_tools, args.search, args.ctx, args.compact)
        else:
            if not prompt_text and not static_blocks:
                parser.error("no prompt provided (and nothing piped on stdin)")
            content = build_user_message(prompt_text, static_blocks, args.search, args.host, args.model, args.ctx)
            messages.append({"role": "user", "content": content})
            run_turn(args.host, args.model, messages, use_tools, args.ctx)
    except requests.RequestException as exc:
        print(f"request failed: {exc}", file=sys.stderr)
        print(f"(is Ollama reachable at {args.host}? check your tailnet and that the desktop is awake)",
              file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
