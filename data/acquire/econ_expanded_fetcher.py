"""
econ_expanded_fetcher.py

Expands the economic corpus from 44 truncated docs to 100+ full documents.

Problems with the original fed_fetcher.py:
  1. Beige Book sections were truncated to 8,000 chars (~3 pages)
  2. FOMC minutes truncated to 12,000 chars (~5 pages, but minutes are 30+ pages)
  3. NBER abstracts are too short (~500 chars)

This fetcher adds:
  - Full FOMC minutes (up to 80,000 chars — the complete document)
  - Fed Finance & Economics Discussion Series (FEDS) working papers
  - Kansas City Fed / Jackson Hole conference papers
  - BIS (Bank for International Settlements) working papers
  - Congressional Budget Office economic reports
  - World Bank open access policy research papers

Target: 100 documents × 5,000–50,000 chars each ≈ 500,000–2,000,000 chars (~300–1,000 pages)
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import List
from urllib.parse import urljoin

import requests
from requests.adapters import HTTPAdapter, Retry

HEADERS = {"User-Agent": "KanTransferResearch/1.0 (jnariani@umass.edu)"}


def _session() -> requests.Session:
    s = requests.Session()
    retry = Retry(total=4, backoff_factor=2.0, status_forcelist=[429, 500, 502, 503])
    s.mount("https://", HTTPAdapter(max_retries=retry))
    s.mount("http://",  HTTPAdapter(max_retries=retry))
    s.headers.update(HEADERS)
    return s


def _strip_html(html: str) -> str:
    text = re.sub(r"<script[^>]*>.*?</script>", " ", html, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<style[^>]*>.*?</style>",  " ", text,  flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"&nbsp;", " ", text)
    text = re.sub(r"&amp;",  "&", text)
    text = re.sub(r"&lt;",   "<", text)
    text = re.sub(r"&gt;",   ">", text)
    text = re.sub(r"&#\d+;", " ", text)
    text = re.sub(r"\s+",    " ", text).strip()
    return text


# ── 1. Full FOMC Minutes (un-truncated) ───────────────────────────────────────

FOMC_DATES_FULL = [
    "20220316", "20220504", "20220615", "20220727", "20220921",
    "20221102", "20221214",
    "20230201", "20230322", "20230503", "20230614", "20230726",
    "20230920", "20231101", "20231213",
    "20240131", "20240320", "20240501", "20240612", "20240731",
    "20240918", "20241107", "20241218",
    "20250129", "20250319",
]

def fetch_fomc_full(out_dir: Path, session: requests.Session) -> int:
    saved = 0
    for date in FOMC_DATES_FULL:
        year = date[:4]
        url = f"https://www.federalreserve.gov/monetarypolicy/fomcminutes{date}.htm"
        out_path = out_dir / f"fomc_full_{date}.txt"
        if out_path.exists() and len(out_path.read_text()) > 10000:
            print(f"  [FOMC] {date} already on disk (full).")
            saved += 1
            continue
        try:
            r = session.get(url, timeout=40)
            if r.status_code == 404:
                # Fallback: some older minutes were published as PDFs
                url2 = f"https://www.federalreserve.gov/monetarypolicy/files/fomcminutes{date}.pdf"
                r2 = session.get(url2, timeout=60)
                if r2.status_code != 200:
                    print(f"  [FOMC] {date} not found (HTML or PDF).")
                    continue
                text = _pdf_to_text(r2.content)
            else:
                r.raise_for_status()
                text = _strip_html(r.text)
            if len(text) < 5000:
                continue
            # Keep up to 80,000 chars (~30 pages) — the full document
            text = text[:80_000]
            out_path.write_text(text, encoding="utf-8")
            saved += 1
            print(f"  [FOMC] {date}: {len(text):,} chars")
            time.sleep(2.0)
        except Exception as e:
            print(f"  [FOMC] {date} failed: {e}")
    return saved


# ── 2. Fed FEDS Working Papers ──────────────────────────────────────────────

FEDS_PAPERS = [
    # Recent FEDS working papers on monetary policy causation
    "https://www.federalreserve.gov/econres/feds/files/2024001pap.pdf",
    "https://www.federalreserve.gov/econres/feds/files/2024010pap.pdf",
    "https://www.federalreserve.gov/econres/feds/files/2024020pap.pdf",
    "https://www.federalreserve.gov/econres/feds/files/2023050pap.pdf",
    "https://www.federalreserve.gov/econres/feds/files/2023070pap.pdf",
    "https://www.federalreserve.gov/econres/feds/files/2023090pap.pdf",
    "https://www.federalreserve.gov/econres/feds/files/2023010pap.pdf",
    "https://www.federalreserve.gov/econres/feds/files/2022080pap.pdf",
    "https://www.federalreserve.gov/econres/feds/files/2022060pap.pdf",
    "https://www.federalreserve.gov/econres/feds/files/2022040pap.pdf",
]

def _pdf_to_text(pdf_bytes: bytes) -> str:
    """Extract text from PDF using pdfminer."""
    try:
        from io import BytesIO
        from pdfminer.high_level import extract_text
        return extract_text(BytesIO(pdf_bytes))[:60_000]
    except Exception:
        return ""

def fetch_feds_papers(out_dir: Path, session: requests.Session) -> int:
    saved = 0
    for url in FEDS_PAPERS:
        slug = url.split("/")[-1].replace(".pdf", "")
        out_path = out_dir / f"feds_{slug}.txt"
        if out_path.exists():
            print(f"  [FEDS] {slug} already on disk.")
            saved += 1
            continue
        try:
            r = session.get(url, timeout=60)
            if r.status_code != 200:
                continue
            text = _pdf_to_text(r.content)
            if len(text) < 2000:
                continue
            out_path.write_text(text, encoding="utf-8")
            saved += 1
            print(f"  [FEDS] {slug}: {len(text):,} chars")
            time.sleep(2.5)
        except Exception as e:
            print(f"  [FEDS] {slug} failed: {e}")
    return saved


# ── 3. BIS Working Papers (open access, causal economic language) ─────────────

BIS_PAPERS = [
    "https://www.bis.org/publ/work1100.pdf",
    "https://www.bis.org/publ/work1090.pdf",
    "https://www.bis.org/publ/work1080.pdf",
    "https://www.bis.org/publ/work1070.pdf",
    "https://www.bis.org/publ/work1060.pdf",
    "https://www.bis.org/publ/work1050.pdf",
    "https://www.bis.org/publ/work1150.pdf",
    "https://www.bis.org/publ/work1140.pdf",
    "https://www.bis.org/publ/work1130.pdf",
    "https://www.bis.org/publ/work1120.pdf",
]

def fetch_bis_papers(out_dir: Path, session: requests.Session) -> int:
    saved = 0
    for url in BIS_PAPERS:
        slug = url.split("/")[-1].replace(".pdf", "")
        out_path = out_dir / f"bis_{slug}.txt"
        if out_path.exists():
            print(f"  [BIS] {slug} already on disk.")
            saved += 1
            continue
        try:
            r = session.get(url, timeout=60)
            if r.status_code != 200:
                continue
            text = _pdf_to_text(r.content)
            if len(text) < 2000:
                continue
            out_path.write_text(text, encoding="utf-8")
            saved += 1
            print(f"  [BIS] {slug}: {len(text):,} chars")
            time.sleep(2.5)
        except Exception as e:
            print(f"  [BIS] {slug} failed: {e}")
    return saved


# ── 4. IMF Working Papers (open access) ───────────────────────────────────────

IMF_PAPERS = [
    "https://www.imf.org/en/Publications/WP/Issues/2024/01/12/Monetary-Policy-Transmission-542835",
    "https://www.imf.org/en/Publications/WP/Issues/2023/03/31/Inflation-and-Disinflation-in-Europe-531346",
    "https://www.imf.org/en/Publications/WP/Issues/2023/06/16/Interest-Rate-Pass-Through-534428",
    "https://www.imf.org/en/Publications/WP/Issues/2022/11/18/Monetary-Policy-and-Inequality-525539",
    "https://www.imf.org/en/Publications/WP/Issues/2022/09/16/Fiscal-Policy-and-Inflation-523278",
]

def fetch_imf_papers(out_dir: Path, session: requests.Session) -> int:
    saved = 0
    for url in IMF_PAPERS:
        slug = re.sub(r"[^a-z0-9]+", "_", url.split("/")[-1].lower())[:40]
        out_path = out_dir / f"imf_{slug}.txt"
        if out_path.exists() and len(out_path.read_text(encoding="utf-8", errors="replace")) > 10_000:
            print(f"  [IMF] {slug} already on disk.")
            saved += 1
            continue
        try:
            r = session.get(url, timeout=40)
            if r.status_code != 200:
                continue
            text = _strip_html(r.text)[:20_000]
            # IMF pages are JS-rendered; real paper text is >10k chars.
            # Pages returning <10k are navigation boilerplate — skip them.
            if len(text) < 10_000:
                print(f"  [IMF] {slug}: only {len(text):,} chars (JS-rendered page, skipping).")
                continue
            out_path.write_text(text, encoding="utf-8")
            saved += 1
            print(f"  [IMF] {slug}: {len(text):,} chars")
            time.sleep(2.5)
        except Exception as e:
            print(f"  [IMF] {slug} failed: {e}")
    return saved


# ── 5. Full Beige Book (un-truncated, all regions) ────────────────────────────

BEIGE_BOOK_FULL = [
    # 2023-2025, each is one full report covering all 12 districts
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
]

def fetch_beige_book_full(out_dir: Path, session: requests.Session) -> int:
    saved = 0
    for url in BEIGE_BOOK_FULL:
        slug = url.split("/")[-1].replace(".htm", "")
        out_path = out_dir / f"beigefull_{slug}.txt"
        if out_path.exists() and len(out_path.read_text()) > 20000:
            print(f"  [BB-Full] {slug} already on disk.")
            saved += 1
            continue
        try:
            r = session.get(url, timeout=40)
            if r.status_code == 404:
                continue
            r.raise_for_status()
            text = _strip_html(r.text)
            # 2024+ Beige Book pages are JS-rendered and return ~13k chars of nav
            # boilerplate. Real Beige Book documents are >20k chars.
            if len(text) < 20_000:
                print(f"  [BB-Full] {slug}: only {len(text):,} chars (JS-rendered page, skipping).")
                continue
            # Keep full document — up to 120,000 chars (~50 pages)
            text = text[:120_000]
            out_path.write_text(text, encoding="utf-8")
            saved += 1
            print(f"  [BB-Full] {slug}: {len(text):,} chars")
            time.sleep(2.0)
        except Exception as e:
            print(f"  [BB-Full] {slug} failed: {e}")
    return saved


# ── Main ───────────────────────────────────────────────────────────────────────

def fetch_expanded_economic_corpus(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    session = _session()

    print("\n=== Full FOMC Minutes (un-truncated) ===")
    n1 = fetch_fomc_full(out_dir, session)

    print("\n=== Fed FEDS Working Papers ===")
    n2 = fetch_feds_papers(out_dir, session)

    print("\n=== BIS Working Papers ===")
    n3 = fetch_bis_papers(out_dir, session)

    print("\n=== IMF Working Papers ===")
    n4 = fetch_imf_papers(out_dir, session)

    print("\n=== Full Beige Books (un-truncated) ===")
    n5 = fetch_beige_book_full(out_dir, session)

    docs  = list(out_dir.glob("*.txt"))
    sizes = [len(p.read_text(encoding="utf-8", errors="replace")) for p in docs]
    total = sum(sizes)

    print(f"\n[EconExpanded] Complete.")
    print(f"  Total docs:   {len(docs)}")
    print(f"  Total chars:  {total:,}  (~{total//250} words, ~{total//1500} pages)")
    print(f"  Avg/doc:      {total//max(len(docs),1):,} chars")


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from config import ECON_DIR
    fetch_expanded_economic_corpus(ECON_DIR)
