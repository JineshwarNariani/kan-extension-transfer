"""
config.py — Central configuration for the Kan Extension Transfer project.

All hyperparameters, paths, and API settings live here.
Pre-registered ranges prevent post-hoc tuning.
"""

from pathlib import Path
import os

# ── Paths ────────────────────────────────────────────────────────────────────

ROOT = Path(__file__).resolve().parent
DEMOCRITUS_ROOT = ROOT.parent / "Democritus_OpenAI"
CLIFF_ROOT      = ROOT.parent / "CLIFF_CatAgi"

CORPORA_DIR = ROOT / "data" / "corpora"
MED_DIR     = CORPORA_DIR / "medical"
ECON_DIR    = CORPORA_DIR / "economic"
LEGAL_DIR   = CORPORA_DIR / "legal"

EXTRACTION_DIR = ROOT / "results" / "extraction"
RESULTS_DIR    = ROOT / "results"
FIGURES_DIR    = ROOT / "results" / "figures"
TABLES_DIR     = ROOT / "results" / "tables"
LOGS_DIR       = ROOT / "logs"

for d in [MED_DIR, ECON_DIR, LEGAL_DIR, EXTRACTION_DIR, FIGURES_DIR, TABLES_DIR, LOGS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ── API ───────────────────────────────────────────────────────────────────────

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
OPENAI_API_KEY    = os.getenv("OPENAI_API_KEY", "")

# If UMass routes Claude through a proxy with an OpenAI-compatible API,
# set DEMOC_LLM_BASE_URL and DEMOC_LLM_MODEL in your shell instead.
# The AnthropicChatClient below uses the native SDK (no proxy needed).

# ── LLM ───────────────────────────────────────────────────────────────────────

PRIMARY_MODEL   = os.getenv("KAN_PRIMARY_MODEL",   "claude-sonnet-4-6")
SECONDARY_MODEL = os.getenv("KAN_SECONDARY_MODEL", "gpt-4o")
MAX_TOKENS      = int(os.getenv("KAN_MAX_TOKENS", "256"))
TEMPERATURE     = float(os.getenv("KAN_TEMPERATURE", "0.7"))

# ── Corpus acquisition ────────────────────────────────────────────────────────

PUBMED_TARGET_COUNT  = 100
ECON_TARGET_COUNT    = 100
LEGAL_TARGET_COUNT   = 30

NCBI_API_KEY = os.getenv("NCBI_API_KEY", "")   # optional — increases rate limit from 3 to 10 req/s

# ── Democritus extraction ─────────────────────────────────────────────────────

DEMOCRITUS_NUM_ROOT_TOPICS = 12    # topics per corpus
DEMOCRITUS_DEPTH_LIMIT     = 2     # BFS depth for topic graph
DEMOCRITUS_MAX_TOPICS      = 150   # global cap per corpus

# ── Functor / Kan ─────────────────────────────────────────────────────────────
# These are pre-registered before any experiment is run.

SBERT_MODEL = "all-MiniLM-L6-v2"   # 384-dim; same as Democritus

KAN_K_PRIMARY        = 10           # number of source topics used in Kan extension
KAN_K_RANGE          = [5, 10, 15]  # sensitivity analysis

KAN_SIM_THRESHOLD    = 0.25         # minimum cosine similarity to include a source topic
KAN_SIM_RANGE        = [0.15, 0.25, 0.35]

KAN_CONSENSUS_FRAC   = 0.6          # Right Kan: fraction of top-k sources that must support a claim
KAN_CONSENSUS_RANGE  = [0.4, 0.6, 0.8]

FUNCTOR_FUZZY_K      = 5            # top-k topics in F(query) lookup

# ── Evaluation ────────────────────────────────────────────────────────────────

TEST_FRACTION  = 0.4    # fraction of economic topics withheld as held-out queries
RANDOM_SEED    = 42
N_TEST_QUERIES = 20     # target number of held-out evaluation queries

SEMANTIC_MATCH_THRESHOLD = 0.80     # cosine sim for soft node matching

# ── Experiment methods ────────────────────────────────────────────────────────

METHODS = ["left_kan", "right_kan", "naive_rag"]
SOURCE_DOMAINS = ["medical", "legal"]
TARGET_DOMAIN  = "economic"
