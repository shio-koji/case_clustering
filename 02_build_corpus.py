#!/usr/bin/env python3
"""
Step 2: Build clean corpus from cached case data.
Uses 'contents' as the primary text field (much richer than 'description').
Tags come from the cases list (human-readable).
"""

import json
import re
from pathlib import Path

CACHE_DIR = Path("cache")


def clean_html(text: str) -> str:
    """Remove HTML entities and basic markup from text."""
    if not text:
        return ""
    # Decode common HTML entities
    text = text.replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">")
    text = text.replace("&quot;", '"').replace("&#39;", "'").replace("&rdquo;", '"')
    text = text.replace("&ldquo;", '"').replace("&nbsp;", " ")
    # Remove HTML tags
    text = re.sub(r"<[^>]+>", "", text)
    # Remove URLs
    text = re.sub(r"https?://\S+", "", text)
    # Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text


def build_corpus():
    # Load the list data (has human-readable tags)
    cases_list = json.loads((CACHE_DIR / "cases_list.json").read_text(encoding="utf-8"))
    list_by_id = {c["id"]: c for c in cases_list["cases"]}

    corpus = []
    for case_id, list_data in list_by_id.items():
        # Load cached details
        detail_path = CACHE_DIR / f"case_{case_id}.json"
        if not detail_path.exists():
            print(f"[WARN] No detail cache for {case_id}")
            continue
        details = json.loads(detail_path.read_text(encoding="utf-8"))

        # Load updates
        updates_path = CACHE_DIR / f"updates_{case_id}.json"
        updates = []
        if updates_path.exists():
            raw = json.loads(updates_path.read_text(encoding="utf-8"))
            if isinstance(raw, list):
                updates = raw
            elif isinstance(raw, dict):
                updates = raw.get("updates", [])

        # --- Extract text fields ---
        title = clean_html(details.get("title") or list_data.get("title") or "")
        description = clean_html(details.get("description") or list_data.get("description") or "")
        contents = clean_html(details.get("contents") or "")

        # Updates: concatenate latest 3 updates' titles + contents
        update_texts = []
        for u in updates[:3]:
            if not isinstance(u, dict):
                continue
            ut = clean_html(u.get("title", ""))
            uc = clean_html(u.get("contents", ""))
            if ut:
                update_texts.append(ut)
            if uc:
                update_texts.append(uc[:800])  # truncate long update content

        # Combined text for embedding (title + description + contents + recent updates)
        parts = [p for p in [title, description, contents] + update_texts if p.strip()]
        combined_text = "\n".join(parts)

        # --- Metadata ---
        # Tags: human-readable from list API
        raw_tags = list_data.get("tags") or []
        if isinstance(raw_tags, str):
            raw_tags = [raw_tags]
        # Filter out 'アーカイブ' as it's a system tag not a subject tag
        subject_tags = [t for t in raw_tags if t != "アーカイブ" and t != "no donation"]

        record = {
            "id": case_id,
            "title": title,
            "description": description,
            "contents_length": len(contents),
            "tags": raw_tags,
            "subject_tags": subject_tags,
            "case_status": details.get("case_status") or list_data.get("case_status") or "",
            "items_category": details.get("items_category") or list_data.get("items_category"),
            "term_start": details.get("term_start") or list_data.get("term_start"),
            "raised_amount": details.get("raised_amount", 0),
            "supporter_count": details.get("supporter_count", 0),
            "lawyer": details.get("lawyer", ""),
            "combined_text": combined_text,
            "text_length": len(combined_text),
            "has_description": bool(description.strip()),
            "has_contents": bool(contents.strip()),
            "has_updates": len(updates) > 0,
            "update_count": len(updates),
        }
        corpus.append(record)

    # Sort by case ID
    corpus.sort(key=lambda x: x["id"])

    out_path = CACHE_DIR / "corpus_clean.json"
    out_path.write_text(json.dumps(corpus, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Corpus saved: {len(corpus)} cases -> {out_path}")
    return corpus


def print_summary(corpus):
    print(f"\n=== Corpus Summary ({len(corpus)} cases) ===")

    # Status
    statuses = {}
    for c in corpus:
        s = c["case_status"]
        statuses[s] = statuses.get(s, 0) + 1
    print("Status:", statuses)

    # Tags
    all_tags = {}
    for c in corpus:
        for tag in c["subject_tags"]:
            all_tags[tag] = all_tags.get(tag, 0) + 1
    print("Top subject tags:")
    for tag, count in sorted(all_tags.items(), key=lambda x: -x[1]):
        print(f"  {count:3d}  {tag}")

    # Text lengths
    lengths = sorted([c["text_length"] for c in corpus])
    n = len(lengths)
    print(f"\nText length: min={lengths[0]}, p25={lengths[n//4]}, median={lengths[n//2]}, p75={lengths[3*n//4]}, max={lengths[-1]}")
    print(f"Has contents: {sum(1 for c in corpus if c['has_contents'])}/{len(corpus)}")
    print(f"Has updates: {sum(1 for c in corpus if c['has_updates'])}/{len(corpus)}")
    print(f"Text < 500 chars: {sum(1 for c in corpus if c['text_length'] < 500)}")
    print(f"Text >= 1000 chars: {sum(1 for c in corpus if c['text_length'] >= 1000)}")


if __name__ == "__main__":
    corpus = build_corpus()
    print_summary(corpus)
