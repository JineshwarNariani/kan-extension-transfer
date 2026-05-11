#!/bin/bash
#SBATCH --job-name=kan_transfer
#SBATCH --mem=128G
#SBATCH --cpus-per-task=16
#SBATCH --time=12:00:00
#SBATCH --partition=cpu
#SBATCH --output=/home/jnariani_umass_edu/cat_theory/kan_transfer/logs/unity_%j.log
#SBATCH --error=/home/jnariani_umass_edu/cat_theory/kan_transfer/logs/unity_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=jnariani@umass.edu

echo "=========================================="
echo "KAN TRANSFER JOB STARTED: $(date)"
echo "Node: $SLURMD_NODENAME"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "Memory: $SLURM_MEM_PER_NODE MB"
echo "=========================================="

# ── Environment ──────────────────────────────────────────────────────────────
source /home/jnariani_umass_edu/thesis_env/bin/activate

# UMass keymaker proxy — LiteLLM requires keys to start with 'sk-'
# Export OPENAI_API_KEY before sbatch, or set it here directly.
# This line normalises whatever key is set so it has the sk- prefix.
if [[ -n "${OPENAI_API_KEY}" ]]; then
    export OPENAI_API_KEY="sk-${OPENAI_API_KEY#sk-}"
fi

export DEMOC_LLM_BASE_URL="https://thekeymaker.umass.edu"
export DEMOC_LLM_MODEL="claude-sonnet-4-6"
export KAN_PRIMARY_MODEL="claude-sonnet-4-6"
export PYTHONPATH=/home/jnariani_umass_edu/cat_theory/Democritus_OpenAI:$PYTHONPATH
export PYTHONPATH=/home/jnariani_umass_edu/cat_theory/CLIFF_CatAgi:$PYTHONPATH
export CLIFF_DEMOCRITUS_ROOT=/home/jnariani_umass_edu/cat_theory/Democritus_OpenAI

cd /home/jnariani_umass_edu/cat_theory/kan_transfer

echo ""
echo "--- Step 0a: Fetch medical corpus (full PMC papers) ---"
python data/acquire/pubmed_fulltext_fetcher.py

echo ""
echo "--- Step 0b: Fetch economic corpus (full FOMC + FEDS/BIS/IMF) ---"
python data/acquire/econ_expanded_fetcher.py

echo ""
echo "--- Step 0c: Fetch legal corpus ---"
python data/acquire/legal_fetcher.py

echo ""
echo "--- Step 1a: Extract MEDICAL (parallel) ---"
python experiments/run_extraction.py --corpus medical --backend openai &
PID_MED=$!

echo "--- Step 1b: Extract ECONOMIC (parallel) ---"
python experiments/run_extraction.py --corpus economic --backend openai &
PID_ECON=$!

wait $PID_MED $PID_ECON
echo "--- Medical + Economic extraction done ---"

# Fail fast if either extraction did not produce triples
for corpus in medical economic; do
    if [[ ! -f "results/extraction/${corpus}/relational_triples.jsonl" ]]; then
        echo "ERROR: results/extraction/${corpus}/relational_triples.jsonl missing — extraction failed."
        echo "Check logs above for the error, fix it, then resubmit."
        exit 1
    fi
done

echo ""
echo "--- Step 1c: Extract LEGAL ---"
python experiments/run_extraction.py --corpus legal --backend openai

if [[ ! -f "results/extraction/legal/relational_triples.jsonl" ]]; then
    echo "ERROR: results/extraction/legal/relational_triples.jsonl missing — legal extraction failed."
    exit 1
fi

echo ""
echo "--- Step 2: Compute Kan extensions ---"
python experiments/run_kan.py --source all

echo ""
echo "--- Step 3: Evaluate ---"
python experiments/run_evaluation.py --source all

echo ""
echo "=========================================="
echo "JOB COMPLETE: $(date)"
echo "Results: /home/jnariani_umass_edu/cat_theory/kan_transfer/results/"
echo "=========================================="
