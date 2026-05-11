"""
pubmed_fulltext_fetcher.py

Gets FULL PAPER TEXT from PubMed Central (PMC) for clinical trials.
The original pubmed_fetcher.py only grabbed 250-word abstracts.
The proposal specifies "2-5 pages each" — we need full papers.

Strategy:
  1. For each PMID we already have, find the corresponding PMC ID
  2. Fetch the full XML from PMC and extract structured text:
     title + abstract + methods + results + conclusions
  3. Fall back to abstract-extended for papers not in PMC
  4. Also search PMC directly for additional clinical trials

Output: overwrites data/corpora/medical/*.txt with full-text versions
        and adds new papers to reach ~100 documents total.
"""

from __future__ import annotations

import json
import re
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import List, Optional, Tuple

import requests

BASE_EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
HEADERS     = {"User-Agent": "KanTransferResearch/1.0 (jnariani@umass.edu)"}


def _session(ncbi_key: str = "") -> requests.Session:
    s = requests.Session()
    s.headers.update(HEADERS)
    return s


def _ncbi_params(extra: dict, ncbi_key: str = "") -> dict:
    p = {"tool": "kan_transfer", "email": "jnariani@umass.edu", **extra}
    if ncbi_key:
        p["api_key"] = ncbi_key
    return p


# ── 1. Find PMC IDs for our existing PMIDs ─────────────────────────────────────

def get_pmc_ids(pmids: List[str], session: requests.Session, ncbi_key: str = "") -> dict:
    """Map PMID → PMCID for papers available in PMC full text."""
    pmc_map = {}
    # Process in batches of 20
    for i in range(0, len(pmids), 20):
        batch = pmids[i:i+20]
        params = _ncbi_params({
            "dbfrom": "pubmed", "db": "pmc",
            "id": ",".join(batch), "retmode": "json",
        }, ncbi_key)
        try:
            r = session.get(f"{BASE_EUTILS}/elink.fcgi", params=params, timeout=30)
            r.raise_for_status()
            data = r.json()
            link_sets = data.get("linksets", [])
            for ls in link_sets:
                pmid = ls.get("ids", [None])[0]
                links = ls.get("linksetdbs", [])
                for ldb in links:
                    if ldb.get("dbto") == "pmc":
                        pmcids = ldb.get("links", [])
                        if pmcids and pmid:
                            pmc_map[str(pmid)] = str(pmcids[0])
        except Exception as e:
            print(f"  [PMC link] Error for batch {i}: {e}")
        time.sleep(0.4 if ncbi_key else 1.2)
    return pmc_map


# ── 2. Fetch full PMC article XML ───────────────────────────────────────────────

def fetch_pmc_fulltext(pmcid: str, session: requests.Session, ncbi_key: str = "") -> str:
    """Fetch full article XML from PMC and extract structured text."""
    params = _ncbi_params({
        "db": "pmc", "id": pmcid,
        "rettype": "xml", "retmode": "xml",
    }, ncbi_key)
    r = session.get(f"{BASE_EUTILS}/efetch.fcgi", params=params, timeout=60)
    r.raise_for_status()
    return _parse_pmc_xml(r.content, pmcid)


def _parse_pmc_xml(xml_bytes: bytes, pmcid: str) -> str:
    """Extract title + abstract + body sections from PMC XML."""
    try:
        root = ET.fromstring(xml_bytes)
    except ET.ParseError:
        return ""

    parts = []

    # Title
    for el in root.iter("article-title"):
        t = "".join(el.itertext()).strip()
        if t:
            parts.append(f"TITLE: {t}\n")
            break

    # Abstract
    for el in root.iter("abstract"):
        text = " ".join(el.itertext()).strip()
        if text and len(text) > 50:
            parts.append(f"\nABSTRACT:\n{text}\n")
            break

    # Body sections: methods, results, discussion, conclusion
    TARGET_SECTIONS = {
        "methods", "materials and methods", "results", "discussion",
        "conclusions", "conclusion", "findings", "outcomes",
    }
    for sec in root.iter("sec"):
        title_el = sec.find("title")
        sec_title = "".join(title_el.itertext()).strip().lower() if title_el is not None else ""

        if not sec_title or not any(k in sec_title for k in TARGET_SECTIONS):
            continue

        text_parts = []
        for p_el in sec.iter("p"):
            t = " ".join(p_el.itertext()).strip()
            if t:
                text_parts.append(t)

        if text_parts:
            section_text = " ".join(text_parts)
            if len(section_text) > 100:
                parts.append(f"\n{sec_title.upper()}:\n{section_text}\n")

    full_text = "\n".join(parts)
    return full_text


# ── 3. Search PMC directly for more clinical trials ────────────────────────────

PMC_QUERIES = [
    # 20 per query, 5 queries = up to 100 additional papers
    (
        "cardiovascular_ext",
        '("randomized controlled trial"[pt]) AND ("statin" OR "hypertension" OR "heart failure" OR "atrial fibrillation" OR "coronary artery") AND (("causes"[tiab] OR "reduces"[tiab] OR "leads to"[tiab] OR "associated with"[tiab])) AND free full text[filter]',
        20,
    ),
    (
        "diabetes_ext",
        '("randomized controlled trial"[pt]) AND ("insulin resistance" OR "type 2 diabetes" OR "glucose tolerance" OR "GLP-1" OR "SGLT2") AND (("causes"[tiab] OR "reduces"[tiab] OR "improves"[tiab])) AND free full text[filter]',
        20,
    ),
    (
        "oncology_ext",
        '("randomized controlled trial"[pt]) AND ("chemotherapy" OR "immunotherapy" OR "breast cancer" OR "lung cancer" OR "colorectal") AND (("causes"[tiab] OR "inhibits"[tiab] OR "reduces"[tiab])) AND free full text[filter]',
        15,
    ),
    (
        "neurology_ext",
        '("randomized controlled trial"[pt]) AND ("Alzheimer" OR "Parkinson" OR "multiple sclerosis" OR "antidepressant" OR "cognitive decline") AND (("causes"[tiab] OR "improves"[tiab] OR "reduces"[tiab])) AND free full text[filter]',
        15,
    ),
    (
        "infectious_disease",
        '("randomized controlled trial"[pt]) AND ("vaccine" OR "antibiotic" OR "antiviral" OR "COVID-19" OR "infection") AND (("causes"[tiab] OR "reduces"[tiab] OR "leads to"[tiab])) AND free full text[filter]',
        15,
    ),
]


def search_pmc_pmids(query: str, retmax: int, session: requests.Session, ncbi_key: str = "") -> List[str]:
    """Search PubMed (not PMC directly) and return PMIDs — we then link to PMC."""
    params = _ncbi_params({
        "db": "pubmed", "term": query, "retmax": retmax + 20,
        "retmode": "json", "usehistory": "n",
    }, ncbi_key)
    r = session.get(f"{BASE_EUTILS}/esearch.fcgi", params=params, timeout=30)
    r.raise_for_status()
    return r.json().get("esearchresult", {}).get("idlist", [])


def fetch_abstract_extended(pmid: str, session: requests.Session, ncbi_key: str = "") -> str:
    """Fallback: fetch structured abstract (Medline format) when PMC full text unavailable."""
    params = _ncbi_params({
        "db": "pubmed", "id": pmid,
        "rettype": "medline", "retmode": "text",
    }, ncbi_key)
    r = session.get(f"{BASE_EUTILS}/efetch.fcgi", params=params, timeout=30)
    r.raise_for_status()
    return r.text[:6000]  # Medline format gives structured text


# ── 4. Main fetch function ──────────────────────────────────────────────────────

def fetch_medical_fulltext_corpus(
    out_dir: Path,
    ncbi_key: str = "",
    target_total: int = 100,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    session = _session(ncbi_key)
    rate_delay = 0.35 if ncbi_key else 1.1

    # Load existing PMIDs from abstract files
    existing_pmids = set()
    for f in out_dir.glob("pmid_*.txt"):
        pmid = f.stem.replace("pmid_", "")
        existing_pmids.add(pmid)
    print(f"[PMC] {len(existing_pmids)} existing PMID files found.")

    # Step 1: Upgrade existing abstracts to full text
    print(f"\n[PMC] === Step 1: Upgrading existing abstracts to full text ===")
    pmc_map = get_pmc_ids(list(existing_pmids), session, ncbi_key)
    print(f"[PMC] {len(pmc_map)}/{len(existing_pmids)} existing PMIDs have PMC full text.")

    upgraded = 0
    for pmid, pmcid in pmc_map.items():
        out_path = out_dir / f"pmid_{pmid}.txt"
        # Check if already upgraded (>2000 chars means full text, not just abstract)
        if out_path.exists() and len(out_path.read_text()) > 3000:
            print(f"  [Skip] pmid_{pmid} already has full text ({len(out_path.read_text())} chars)")
            upgraded += 1
            continue
        try:
            full_text = fetch_pmc_fulltext(pmcid, session, ncbi_key)
            if len(full_text) > 1000:
                out_path.write_text(full_text, encoding="utf-8")
                print(f"  [OK] pmid_{pmid} (PMC{pmcid}): {len(full_text):,} chars")
                upgraded += 1
            else:
                print(f"  [Sparse] PMC{pmcid}: only {len(full_text)} chars, keeping abstract")
            time.sleep(rate_delay)
        except Exception as e:
            print(f"  [Error] PMC{pmcid}: {e}")
            time.sleep(rate_delay * 2)

    print(f"[PMC] Upgraded {upgraded}/{len(pmc_map)} files to full text.")

    # Step 2: Search for additional papers to reach target_total
    current_count = len(list(out_dir.glob("pmid_*.txt")))
    needed = target_total - current_count
    print(f"\n[PMC] === Step 2: Fetching {needed} additional papers ===")

    for subdomain, query, retmax in PMC_QUERIES:
        if current_count >= target_total:
            break

        print(f"\n[PMC] Subdomain: {subdomain} (need {target_total - current_count} more)")
        pmids = search_pmc_pmids(query, retmax + 10, session, ncbi_key)
        new_pmids = [p for p in pmids if p not in existing_pmids]
        print(f"[PMC]   {len(pmids)} found, {len(new_pmids)} new.")

        if not new_pmids:
            continue

        # Get PMC IDs for the new PMIDs
        new_pmc_map = get_pmc_ids(new_pmids[:retmax], session, ncbi_key)
        print(f"[PMC]   {len(new_pmc_map)} have PMC full text.")

        for pmid, pmcid in list(new_pmc_map.items())[:retmax]:
            if current_count >= target_total:
                break
            out_path = out_dir / f"pmid_{pmid}.txt"
            if out_path.exists():
                existing_pmids.add(pmid)
                current_count += 1
                continue
            try:
                full_text = fetch_pmc_fulltext(pmcid, session, ncbi_key)
                if len(full_text) > 1000:
                    out_path.write_text(full_text, encoding="utf-8")
                    existing_pmids.add(pmid)
                    current_count += 1
                    print(f"  [New] pmid_{pmid}: {len(full_text):,} chars")
                time.sleep(rate_delay)
            except Exception as e:
                print(f"  [Error] pmid_{pmid}: {e}")
                time.sleep(rate_delay * 2)

        # For those without PMC full text, get extended Medline records
        no_pmc = [p for p in new_pmids[:5] if p not in new_pmc_map]
        for pmid in no_pmc:
            if current_count >= target_total:
                break
            out_path = out_dir / f"pmid_{pmid}.txt"
            if out_path.exists():
                continue
            try:
                text = fetch_abstract_extended(pmid, session, ncbi_key)
                if len(text) > 300:
                    out_path.write_text(text, encoding="utf-8")
                    existing_pmids.add(pmid)
                    current_count += 1
                    print(f"  [Medline] pmid_{pmid}: {len(text):,} chars")
                time.sleep(rate_delay)
            except Exception as e:
                print(f"  [Error Medline] pmid_{pmid}: {e}")

    # Final stats
    docs = list(out_dir.glob("pmid_*.txt"))
    sizes = [len(p.read_text(encoding="utf-8", errors="replace")) for p in docs]
    total_chars = sum(sizes)
    print(f"\n[PMC] DONE.")
    print(f"  Documents: {len(docs)}")
    print(f"  Total chars: {total_chars:,} (~{total_chars//250} words, ~{total_chars//1500} pages)")
    print(f"  Avg chars/doc: {int(total_chars/len(docs)) if docs else 0:,}")


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from config import MED_DIR, NCBI_API_KEY

    fetch_medical_fulltext_corpus(MED_DIR, ncbi_key=NCBI_API_KEY, target_total=100)
