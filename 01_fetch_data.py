#!/usr/bin/env python3
"""
Step 1: Fetch all CALL4 case data from the API and cache locally.
Fetches case list, case details, and updates for each case.
"""

import json
import time
import urllib.request
import urllib.error
from pathlib import Path
from datetime import datetime, timezone

API_ENDPOINT = "https://www.call4.jp/flight_api/mcp/sse"
CACHE_DIR = Path("cache")
CACHE_DIR.mkdir(exist_ok=True)

_req_id = 0
_cookie = ""


def call_tool(name: str, arguments: dict) -> dict:
    global _req_id, _cookie

    # Initialize session if needed (simple approach: always init)
    _req_id += 1
    payload = {
        "jsonrpc": "2.0",
        "id": _req_id,
        "method": "tools/call",
        "params": {"name": name, "arguments": arguments},
    }
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        API_ENDPOINT,
        data=body,
        headers={
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
            **({"Cookie": _cookie} if _cookie else {}),
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        # capture cookies
        set_cookie = resp.headers.get("Set-Cookie", "")
        if set_cookie:
            parts = [c.split(";")[0].strip() for c in set_cookie.split(",") if "=" in c.split(";")[0]]
            _cookie = "; ".join(parts)
        raw = resp.read().decode("utf-8").strip()

    if raw.startswith("data:"):
        raw = raw[5:].strip()
    elif raw.startswith("event:"):
        lines = raw.split("\n")
        for line in lines:
            if line.startswith("data:"):
                raw = line[5:].strip()
                break

    obj = json.loads(raw)
    if "error" in obj:
        raise RuntimeError(f"API error: {obj['error']}")
    result = obj.get("result", {})
    if result.get("isError"):
        content = result.get("content", [])
        msg = " ".join(c.get("text", "") for c in content if c.get("type") == "text")
        raise RuntimeError(f"Tool error: {msg}")
    content = result.get("content", [])
    for c in content:
        if c.get("type") == "text":
            return json.loads(c["text"])
    return {}


def fetch_all_cases():
    """Fetch all case summaries."""
    cache_path = CACHE_DIR / "cases_list.json"
    if cache_path.exists():
        print(f"[cache] Loading case list from {cache_path}")
        return json.loads(cache_path.read_text(encoding="utf-8"))

    print("Fetching case list from API...")
    data = call_tool("search_cases", {"query": "", "search_mode": "or"})
    data["fetched_at"] = datetime.now(timezone.utc).isoformat()
    cache_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"  -> {data.get('total', 0)} cases found, cached to {cache_path}")
    return data


def fetch_case_details(case_id: str) -> dict:
    """Fetch detailed info for a single case, with caching."""
    cache_path = CACHE_DIR / f"case_{case_id}.json"
    if cache_path.exists():
        return json.loads(cache_path.read_text(encoding="utf-8"))

    try:
        data = call_tool("get_case_details", {"id": case_id})
        data["_fetched_at"] = datetime.now(timezone.utc).isoformat()
        cache_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        return data
    except Exception as e:
        print(f"  [WARN] Failed to fetch details for {case_id}: {e}")
        return {"id": case_id, "_error": str(e)}


def fetch_updates(case_id: str) -> list:
    """Fetch litigation progress updates for a case."""
    cache_path = CACHE_DIR / f"updates_{case_id}.json"
    if cache_path.exists():
        return json.loads(cache_path.read_text(encoding="utf-8"))

    try:
        data = call_tool("get_updates", {"id": case_id})
        updates = data if isinstance(data, list) else data.get("updates", [])
        cache_path.write_text(json.dumps(updates, ensure_ascii=False, indent=2), encoding="utf-8")
        return updates
    except Exception as e:
        print(f"  [WARN] Failed to fetch updates for {case_id}: {e}")
        return []


def build_corpus():
    """Combine all fetched data into a single corpus file."""
    cases_list = fetch_all_cases()
    all_cases = cases_list.get("cases", [])
    print(f"\nFetching details for {len(all_cases)} cases...")

    corpus = []
    for i, case_summary in enumerate(all_cases, 1):
        case_id = case_summary["id"]
        print(f"  [{i:3d}/{len(all_cases)}] {case_id}: {case_summary['title'][:40]}...", end="", flush=True)

        details = fetch_case_details(case_id)
        updates = fetch_updates(case_id)

        # Build combined text for embedding
        text_parts = []
        title = details.get("title") or case_summary.get("title") or ""
        description = details.get("description") or case_summary.get("description") or ""
        claims = details.get("claims", "") or ""  # 原告の主張
        background = details.get("background", "") or ""
        issues = details.get("issues", "") or details.get("争点", "") or ""

        # Updates text
        update_texts = []
        for u in updates[:5]:  # top 5 updates
            if isinstance(u, dict):
                t = u.get("body") or u.get("content") or u.get("text") or ""
                if t:
                    update_texts.append(t[:500])

        text_parts = [p for p in [title, description, background, claims, issues] if p.strip()]
        if update_texts:
            text_parts.extend(update_texts[:3])

        combined_text = "\n".join(text_parts)

        record = {
            "id": case_id,
            "title": title,
            "description": description,
            "tags": details.get("tags") or case_summary.get("tags") or [],
            "case_status": details.get("case_status") or case_summary.get("case_status") or "",
            "term_start": details.get("term_start") or case_summary.get("term_start"),
            "term_end": details.get("term_end") or case_summary.get("term_end"),
            "items_category": details.get("items_category") or case_summary.get("items_category"),
            "combined_text": combined_text,
            "text_length": len(combined_text),
            "has_description": bool(description.strip()),
            "has_claims": bool(claims.strip()),
            "has_updates": len(updates) > 0,
            "update_count": len(updates),
            "raw_details": details,  # keep for reference
        }
        corpus.append(record)
        print(f" [{len(combined_text)} chars]")
        time.sleep(0.1)  # polite rate limiting

    corpus_path = CACHE_DIR / "corpus.json"
    corpus_path.write_text(json.dumps(corpus, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nCorpus saved to {corpus_path} ({len(corpus)} records)")
    return corpus


def print_summary(corpus):
    print("\n=== Corpus Summary ===")
    print(f"Total cases: {len(corpus)}")
    statuses = {}
    for c in corpus:
        s = c.get("case_status", "unknown")
        statuses[s] = statuses.get(s, 0) + 1
    print("Status distribution:", statuses)

    all_tags = {}
    for c in corpus:
        for tag in c.get("tags", []):
            all_tags[tag] = all_tags.get(tag, 0) + 1
    print("Top tags:", sorted(all_tags.items(), key=lambda x: -x[1])[:15])

    text_lengths = [c["text_length"] for c in corpus]
    print(f"Text length: min={min(text_lengths)}, median={sorted(text_lengths)[len(text_lengths)//2]}, max={max(text_lengths)}")
    print(f"Has description: {sum(1 for c in corpus if c['has_description'])}/{len(corpus)}")
    print(f"Has updates: {sum(1 for c in corpus if c['has_updates'])}/{len(corpus)}")


if __name__ == "__main__":
    corpus = build_corpus()
    print_summary(corpus)
