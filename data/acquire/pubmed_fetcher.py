"""
pubmed_fetcher.py

Downloads 50 PubMed clinical trial abstracts with explicit causal language.
Uses the NCBI E-utilities REST API (free, no auth required; NCBI_API_KEY optional).

Output: data/corpora/medical/pmid_{id}.txt  +  data/corpora/medical/metadata.json
"""

from __future__ import annotations

import json
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import List, Optional

import requests

BASE_ESEARCH = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
BASE_EFETCH  = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"

# Four subdomains × 13 abstracts each ≈ 52 total (we keep best 50)
QUERIES = [
    (
        "cardiovascular",
        '("randomized controlled trial"[pt]) AND ("statin" OR "hypertension" OR "blood pressure" '
        'OR "myocardial infarction" OR "cholesterol") AND ("causes" OR "reduces" OR "leads to" '
        'OR "associated with" OR "effect of") AND hasabstract[text] AND english[lang]',
        13,
    ),
    (
        "diabetes",
        '("randomized controlled trial"[pt]) AND ("insulin" OR "HbA1c" OR "glycemic" '
        'OR "metformin" OR "type 2 diabetes") AND ("causes" OR "reduces" OR "improves" '
        'OR "leads to") AND hasabstract[text] AND english[lang]',
        13,
    ),
    (
        "oncology",
        '("randomized controlled trial"[pt]) AND ("chemotherapy" OR "tumor" OR "cancer" '
        'OR "immunotherapy" OR "radiation") AND ("causes" OR "inhibits" OR "reduces" '
        'OR "leads to") AND hasabstract[text] AND english[lang]',
        12,
    ),
    (
        "neurology",
        '("randomized controlled trial"[pt]) AND ("SSRI" OR "depression" OR "cognitive" '
        'OR "neurological" OR "stroke") AND ("improves" OR "reduces" OR "affects" '
        'OR "leads to") AND hasabstract[text] AND english[lang]',
        12,
    ),
]


def _build_params(extra: dict, ncbi_key: str = "") -> dict:
    p = {"retmode": "json", "tool": "kan_transfer", "email": "jnariani@umass.edu"}
    if ncbi_key:
        p["api_key"] = ncbi_key
    p.update(extra)
    return p


def search_pmids(query: str, retmax: int, ncbi_key: str = "") -> List[str]:
    params = _build_params(
        {"db": "pubmed", "term": query, "retmax": retmax, "usehistory": "n"}, ncbi_key
    )
    resp = requests.get(BASE_ESEARCH, params=params, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    return data.get("esearchresult", {}).get("idlist", [])


def fetch_abstracts(pmids: List[str], ncbi_key: str = "") -> List[dict]:
    """Fetch abstracts for a list of PMIDs; returns list of {pmid, title, abstract}."""
    if not pmids:
        return []
    params = _build_params(
        {"db": "pubmed", "id": ",".join(pmids), "rettype": "abstract", "retmode": "xml"},
        ncbi_key,
    )
    resp = requests.get(BASE_EFETCH, params=params, timeout=60)
    resp.raise_for_status()

    root = ET.fromstring(resp.content)
    records = []
    for article in root.findall(".//PubmedArticle"):
        pmid_el = article.find(".//PMID")
        title_el = article.find(".//ArticleTitle")
        abstract_el = article.find(".//AbstractText")
        journal_el  = article.find(".//Journal/Title")
        year_el     = article.find(".//PubDate/Year")

        pmid     = pmid_el.text if pmid_el is not None else "?"
        title    = title_el.text if title_el is not None else ""
        abstract = "".join(abstract_el.itertext()) if abstract_el is not None else ""
        journal  = journal_el.text if journal_el is not None else ""
        year     = year_el.text if year_el is not None else ""

        if abstract and len(abstract) > 100:
            records.append(
                {"pmid": pmid, "title": title, "abstract": abstract,
                 "journal": journal, "year": year}
            )
    return records


def fetch_medical_corpus(
    out_dir: Path,
    ncbi_key: str = "",
    target_total: int = 50,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    meta_path = out_dir / "metadata.json"

    existing = {p.stem.replace("pmid_", "") for p in out_dir.glob("pmid_*.txt")}
    print(f"[PubMed] {len(existing)} abstracts already on disk.")

    all_meta = []
    total = 0

    for subdomain, query, retmax in QUERIES:
        if total >= target_total:
            break

        print(f"\n[PubMed] Subdomain: {subdomain} — searching for {retmax} PMIDs…")
        pmids = search_pmids(query, retmax + 10, ncbi_key)  # +10 buffer
        new_pmids = [p for p in pmids if p not in existing]
        print(f"[PubMed]   {len(pmids)} found, {len(new_pmids)} new.")

        if not new_pmids:
            continue

        time.sleep(0.35 if ncbi_key else 1.0)   # rate limit: 10/s with key, 3/s without
        records = fetch_abstracts(new_pmids[:retmax], ncbi_key)
        print(f"[PubMed]   {len(records)} abstracts fetched.")

        for rec in records:
            txt_path = out_dir / f"pmid_{rec['pmid']}.txt"
            txt_path.write_text(
                f"TITLE: {rec['title']}\n\nABSTRACT:\n{rec['abstract']}\n",
                encoding="utf-8",
            )
            all_meta.append({**rec, "subdomain": subdomain})
            existing.add(rec["pmid"])
            total += 1

        time.sleep(0.5)

    meta_path.write_text(json.dumps(all_meta, indent=2), encoding="utf-8")
    print(f"\n[PubMed] Done. {total} abstracts saved to {out_dir}")


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from config import MED_DIR, NCBI_API_KEY, PUBMED_TARGET_COUNT

    fetch_medical_corpus(MED_DIR, ncbi_key=NCBI_API_KEY, target_total=PUBMED_TARGET_COUNT)
