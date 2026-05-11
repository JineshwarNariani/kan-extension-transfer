"""
legal_fetcher.py

Downloads ~30 legal regulatory documents for the ablation experiment.
Sources:
  - FTC press releases / consent orders (causal "caused harm" language)
  - SEC litigation releases (enforcement actions)
  - CFPB enforcement actions

No API key required — all public government sources.
Output: data/corpora/legal/*.txt  +  metadata.json
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import List

import requests
from requests.adapters import HTTPAdapter, Retry

HEADERS = {"User-Agent": "KanTransferResearch/1.0 (jnariani@umass.edu)"}


def _session() -> requests.Session:
    s = requests.Session()
    retry = Retry(total=4, backoff_factor=2.0, status_forcelist=[429, 500, 502, 503])
    s.mount("https://", HTTPAdapter(max_retries=retry))
    s.headers.update(HEADERS)
    return s


def _strip_html(html: str) -> str:
    text = re.sub(r"<script[^>]*>.*?</script>", " ", html, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<style[^>]*>.*?</style>",  " ", text,  flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"&nbsp;", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ── FTC Press Releases ────────────────────────────────────────────────────────
# Enforcement actions with causal language

FTC_URLS = [
    "https://www.ftc.gov/news-events/news/press-releases/2023/01/ftc-takes-action-against-fortnite-maker-epic-games-deceptive-tactics",
    "https://www.ftc.gov/news-events/news/press-releases/2023/07/ftc-moves-block-amazons-acquisition-irobot",
    "https://www.ftc.gov/news-events/news/press-releases/2023/09/ftc-takes-action-ban-surveillance-pricing",
    "https://www.ftc.gov/news-events/news/press-releases/2024/01/ftc-bans-outlogic-x-mode-social-selling-sensitive-location-data",
    "https://www.ftc.gov/news-events/news/press-releases/2024/03/ftc-takes-action-against-avast-selling-browsing-data",
    "https://www.ftc.gov/news-events/news/press-releases/2024/05/ftc-finalizes-order-against-telehealth-firm-cerebral-misusing-sensitive-health-data",
    "https://www.ftc.gov/news-events/news/press-releases/2024/08/ftc-report-finds-surveillance-pricing-raises-concerns-about-competition-consumer-privacy",
    "https://www.ftc.gov/news-events/news/press-releases/2023/03/ftc-takes-action-against-malicious-spyware-stalkerware-app-developers",
    "https://www.ftc.gov/news-events/news/press-releases/2022/09/ftc-takes-action-against-data-broker-kochava",
    "https://www.ftc.gov/news-events/news/press-releases/2022/11/ftc-sues-amazon-ceo-jeff-bezos-founder-chairman",
]

# ── SEC Litigation Releases ────────────────────────────────────────────────────

SEC_URLS = [
    "https://www.sec.gov/litigation/litreleases/2023/lr25656.htm",
    "https://www.sec.gov/litigation/litreleases/2023/lr25700.htm",
    "https://www.sec.gov/litigation/litreleases/2023/lr25750.htm",
    "https://www.sec.gov/litigation/litreleases/2024/lr25900.htm",
    "https://www.sec.gov/litigation/litreleases/2024/lr25950.htm",
    "https://www.sec.gov/litigation/litreleases/2022/lr25500.htm",
    "https://www.sec.gov/litigation/litreleases/2022/lr25550.htm",
    "https://www.sec.gov/litigation/litreleases/2023/lr25600.htm",
    "https://www.sec.gov/litigation/litreleases/2023/lr25625.htm",
    "https://www.sec.gov/litigation/litreleases/2024/lr25800.htm",
]

# ── CFPB Enforcement Actions ──────────────────────────────────────────────────

CFPB_URLS = [
    "https://www.consumerfinance.gov/enforcement/actions/navient/",
    "https://www.consumerfinance.gov/enforcement/actions/wells-fargo-bank-na-3/",
    "https://www.consumerfinance.gov/enforcement/actions/jpmorgan-chase-bank-na/",
    "https://www.consumerfinance.gov/enforcement/actions/bank-of-america-na/",
    "https://www.consumerfinance.gov/enforcement/actions/citibank-na/",
    "https://www.consumerfinance.gov/enforcement/actions/trans-union/",
    "https://www.consumerfinance.gov/enforcement/actions/equifax-inc/",
    "https://www.consumerfinance.gov/enforcement/actions/experian-information-solutions-inc/",
    "https://www.consumerfinance.gov/enforcement/actions/ace-cash-express/",
    "https://www.consumerfinance.gov/enforcement/actions/enova-international-inc/",
]


def _fetch_url(url: str, session: requests.Session, max_chars: int = 5000) -> str:
    resp = session.get(url, timeout=30)
    resp.raise_for_status()
    text = _strip_html(resp.text)
    return text[:max_chars]


def fetch_legal_corpus(out_dir: Path, target_total: int = 30) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    session = _session()
    all_meta = []
    saved = 0

    all_urls = [
        ("ftc", u) for u in FTC_URLS
    ] + [
        ("sec", u) for u in SEC_URLS
    ] + [
        ("cfpb", u) for u in CFPB_URLS
    ]

    for source, url in all_urls:
        if saved >= target_total:
            break
        slug = re.sub(r"[^a-z0-9]+", "_", url.split("/")[-2] or url.split("/")[-1])[:40]
        out_path = out_dir / f"{source}_{slug}.txt"

        if out_path.exists():
            print(f"  [Legal] {out_path.name} already on disk.")
            all_meta.append({"source": source, "url": url, "file": out_path.name})
            saved += 1
            continue

        try:
            text = _fetch_url(url, session)
            if len(text) < 200:
                print(f"  [Legal] Too short ({len(text)} chars): {url}")
                continue
            out_path.write_text(text, encoding="utf-8")
            all_meta.append({"source": source, "url": url, "file": out_path.name, "chars": len(text)})
            saved += 1
            print(f"  [Legal] {out_path.name} saved ({len(text)} chars).")
            time.sleep(2.0)
        except Exception as exc:
            print(f"  [Legal] Failed {url}: {exc}")

    meta_path = out_dir / "metadata.json"
    meta_path.write_text(json.dumps(all_meta, indent=2), encoding="utf-8")
    print(f"\n[Legal] Done. {saved} documents saved to {out_dir}")


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from config import LEGAL_DIR, LEGAL_TARGET_COUNT

    fetch_legal_corpus(LEGAL_DIR, target_total=LEGAL_TARGET_COUNT)
