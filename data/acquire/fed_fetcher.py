"""
fed_fetcher.py

Downloads 50 economic documents from three sources:
  1. Federal Reserve Beige Book sections  (20 docs)
  2. FOMC Meeting Minutes HTML              (15 docs)
  3. NBER Working Paper abstracts           (15 docs)

Output: data/corpora/economic/*.txt  +  metadata.json
No API key required — all public sources.
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import List, Optional
from urllib.parse import urljoin

import requests
from requests.adapters import HTTPAdapter, Retry

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (compatible; KanTransferResearch/1.0; "
        "+https://github.com/jnariani/kan_transfer)"
    )
}

def _session() -> requests.Session:
    s = requests.Session()
    retry = Retry(total=4, backoff_factor=1.5, status_forcelist=[429, 500, 502, 503])
    s.mount("https://", HTTPAdapter(max_retries=retry))
    s.headers.update(HEADERS)
    return s


# ── 1. Beige Book ─────────────────────────────────────────────────────────────

BEIGE_BOOK_URLS = [
    # Format: (year, month-abbr) → URL
    # Using 2022–2025 releases; each URL is one regional chapter (~800–1500 words)
    "https://www.federalreserve.gov/monetarypolicy/beigebook202301.htm",
    "https://www.federalreserve.gov/monetarypolicy/beigebook202303.htm",
    "https://www.federalreserve.gov/monetarypolicy/beigebook202305.htm",
    "https://www.federalreserve.gov/monetarypolicy/beigebook202307.htm",
    "https://www.federalreserve.gov/monetarypolicy/beigebook202309.htm",
    "https://www.federalreserve.gov/monetarypolicy/beigebook202311.htm",
    "https://www.federalreserve.gov/monetarypolicy/beigebook202401.htm",
    "https://www.federalreserve.gov/monetarypolicy/beigebook202403.htm",
    "https://www.federalreserve.gov/monetarypolicy/beigebook202405.htm",
    "https://www.federalreserve.gov/monetarypolicy/beigebook202407.htm",
    "https://www.federalreserve.gov/monetarypolicy/beigebook202409.htm",
    "https://www.federalreserve.gov/monetarypolicy/beigebook202411.htm",
    "https://www.federalreserve.gov/monetarypolicy/beigebook202501.htm",
    "https://www.federalreserve.gov/monetarypolicy/beigebook202503.htm",
    "https://www.federalreserve.gov/monetarypolicy/beigebook202302.htm",  # alt date format fallback
    "https://www.federalreserve.gov/monetarypolicy/beigebook202402.htm",
    "https://www.federalreserve.gov/monetarypolicy/beigebook202502.htm",
    "https://www.federalreserve.gov/monetarypolicy/beigebook202304.htm",
    "https://www.federalreserve.gov/monetarypolicy/beigebook202404.htm",
    "https://www.federalreserve.gov/monetarypolicy/beigebook202406.htm",
]

def _strip_html(html: str) -> str:
    """Very lightweight HTML → plaintext. No BeautifulSoup dependency."""
    text = re.sub(r"<script[^>]*>.*?</script>", " ", html, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<style[^>]*>.*?</style>",  " ", text,  flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"&nbsp;", " ", text)
    text = re.sub(r"&amp;",  "&", text)
    text = re.sub(r"&lt;",   "<", text)
    text = re.sub(r"&gt;",   ">", text)
    text = re.sub(r"&#\d+;", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def fetch_beige_book(out_dir: Path, session: requests.Session, target: int = 20) -> List[dict]:
    meta = []
    saved = 0
    for url in BEIGE_BOOK_URLS:
        if saved >= target:
            break
        slug = url.split("/")[-1].replace(".htm", "")
        out_path = out_dir / f"beige_{slug}.txt"
        if out_path.exists():
            print(f"  [BB] {slug} already on disk, skipping.")
            meta.append({"source": "beige_book", "url": url, "file": out_path.name})
            saved += 1
            continue
        try:
            resp = session.get(url, timeout=30)
            if resp.status_code == 404:
                print(f"  [BB] 404 for {url}, skipping.")
                continue
            resp.raise_for_status()
            text = _strip_html(resp.text)
            if len(text) < 500:
                continue
            # Keep first 8000 chars — one beige book section
            text = text[:8000]
            out_path.write_text(text, encoding="utf-8")
            meta.append({"source": "beige_book", "url": url, "file": out_path.name, "chars": len(text)})
            saved += 1
            print(f"  [BB] {slug} saved ({len(text)} chars).")
            time.sleep(1.5)
        except Exception as exc:
            print(f"  [BB] Failed {url}: {exc}")
    return meta


# ── 2. FOMC Minutes ────────────────────────────────────────────────────────────

FOMC_DATES = [
    "20230201", "20230322", "20230503", "20230614", "20230726",
    "20230920", "20231101", "20231213",
    "20240131", "20240320", "20240501", "20240612", "20240731",
    "20240918", "20241107",
]

FOMC_BASE = "https://www.federalreserve.gov/monetarypolicy/fomcminutes{date}.htm"


def fetch_fomc_minutes(out_dir: Path, session: requests.Session, target: int = 15) -> List[dict]:
    meta = []
    saved = 0
    for date in FOMC_DATES:
        if saved >= target:
            break
        url = FOMC_BASE.format(date=date)
        out_path = out_dir / f"fomc_{date}.txt"
        if out_path.exists():
            print(f"  [FOMC] {date} already on disk.")
            meta.append({"source": "fomc", "date": date, "file": out_path.name})
            saved += 1
            continue
        try:
            resp = session.get(url, timeout=30)
            if resp.status_code == 404:
                print(f"  [FOMC] 404 for {date}, skipping.")
                continue
            resp.raise_for_status()
            text = _strip_html(resp.text)
            if len(text) < 1000:
                continue
            # FOMC minutes are long (5k–15k words) — keep first 12000 chars
            text = text[:12000]
            out_path.write_text(text, encoding="utf-8")
            meta.append({"source": "fomc", "date": date, "file": out_path.name, "chars": len(text)})
            saved += 1
            print(f"  [FOMC] {date} saved ({len(text)} chars).")
            time.sleep(2.0)
        except Exception as exc:
            print(f"  [FOMC] Failed {date}: {exc}")
    return meta


# ── 3. NBER Working Paper Abstracts ───────────────────────────────────────────

# Fixed list of NBER papers on monetary policy / macroeconomic causality.
# Abstracts are short but dense and public.
NBER_PAPER_IDS = [
    "w31363", "w31211", "w30991", "w30888", "w30744", "w30682", "w30612",
    "w30431", "w30325", "w30201", "w30099", "w29950", "w29801", "w29700",
    "w29612",
]

NBER_BASE = "https://www.nber.org/papers/{pid}"


def fetch_nber_abstracts(out_dir: Path, session: requests.Session, target: int = 15) -> List[dict]:
    meta = []
    saved = 0
    for pid in NBER_PAPER_IDS:
        if saved >= target:
            break
        out_path = out_dir / f"nber_{pid}.txt"
        if out_path.exists():
            print(f"  [NBER] {pid} already on disk.")
            meta.append({"source": "nber", "paper_id": pid, "file": out_path.name})
            saved += 1
            continue
        try:
            url = NBER_BASE.format(pid=pid)
            resp = session.get(url, timeout=30)
            resp.raise_for_status()
            text = _strip_html(resp.text)
            # Extract just the abstract region (heuristic: first 3000 chars of body)
            # NBER pages have abstract text near the top after navigation
            # Find the abstract section
            abstract_match = re.search(
                r"Abstract\s*[:\-]?\s*(.{200,3000}?)(?:\n\n|\r\n\r\n|JEL)", text,
                re.DOTALL | re.IGNORECASE,
            )
            if abstract_match:
                abstract_text = abstract_match.group(1).strip()
            else:
                # Fall back to first 2000 chars after "Abstract"
                idx = text.lower().find("abstract")
                abstract_text = text[max(0, idx):idx + 2000] if idx >= 0 else text[:2000]

            if len(abstract_text) < 100:
                continue

            out_path.write_text(abstract_text, encoding="utf-8")
            meta.append({"source": "nber", "paper_id": pid, "file": out_path.name, "chars": len(abstract_text)})
            saved += 1
            print(f"  [NBER] {pid} saved ({len(abstract_text)} chars).")
            time.sleep(2.5)
        except Exception as exc:
            print(f"  [NBER] Failed {pid}: {exc}")
    return meta


# ── Main ───────────────────────────────────────────────────────────────────────

def fetch_economic_corpus(out_dir: Path, target_total: int = 50) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    session = _session()
    all_meta = []

    print("\n[EconFetch] === Beige Book ===")
    all_meta.extend(fetch_beige_book(out_dir, session, target=20))

    print("\n[EconFetch] === FOMC Minutes ===")
    all_meta.extend(fetch_fomc_minutes(out_dir, session, target=15))

    print("\n[EconFetch] === NBER Abstracts ===")
    all_meta.extend(fetch_nber_abstracts(out_dir, session, target=15))

    meta_path = out_dir / "metadata.json"
    meta_path.write_text(json.dumps(all_meta, indent=2), encoding="utf-8")

    count = len(list(out_dir.glob("*.txt")))
    print(f"\n[EconFetch] Done. {count} documents saved to {out_dir}")


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from config import ECON_DIR, ECON_TARGET_COUNT

    fetch_economic_corpus(ECON_DIR, target_total=ECON_TARGET_COUNT)
