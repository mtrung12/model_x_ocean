#!/usr/bin/env bash
# Build essays-only submission/ folder (<25MB target).
# Excludes: mypersonality, ollama, logs, FAISS vector_db, profile_db,
#           false_predictions dumps, caches, backups.
set -euo pipefail
cd "$(dirname "$0")"

OUT=submission
rm -rf "$OUT"
mkdir -p "$OUT"

# --- root files ---
cp README.md requirements.txt .env.example LICENSE .gitignore "$OUT"/ 2>/dev/null || true

# --- source code packages ---
cp -r rag baselines ptd_model scripts "$OUT"/
mkdir -p "$OUT"/utils
cp utils/__init__.py utils/gpt_client.py utils/hf_client.py utils/log.py utils/parser.py "$OUT"/utils/
# (utils/ollama_client.py dropped — thesis uses GPT + LLaMA only)

# --- design (no backups / Copy) ---
mkdir -p "$OUT"/design
cp design/pipeline_overview.drawio design/demo_webapp_requirements.md "$OUT"/design/ 2>/dev/null || true

# --- stats (ocean trackers only) ---
mkdir -p "$OUT"/stats
cp stats/F1_score_tracker_ocean.xlsx stats/Retrieve_accuracy_tracker.xlsx stats/retrieved_sample_statistics.xlsx "$OUT"/stats/ 2>/dev/null || true

# --- classical / neural baseline notebooks ---
cp -r baseline_notebook_ML_DL "$OUT"/

# --- method notebooks (drop myp + ollama) ---
cp -r notebook "$OUT"/
rm -rf "$OUT"/notebook/data_myp "$OUT"/notebook/ollama

# --- data: essays only ---
# processed_essay.csv omitted: regenerated from data/raw by the processing notebook.
mkdir -p "$OUT"/data/raw "$OUT"/data/split
cp data/raw/wcpr_essays.csv "$OUT"/data/raw/
cp -r data/split/essays "$OUT"/data/split/
# drop redundant test shards (subsets of test.csv)
rm -f "$OUT"/data/split/essays/test_part_*.csv "$OUT"/data/split/essays/minor_test.csv

# --- results: keep CSV summaries + classification reports, drop big raw dumps ---
cp -r result "$OUT"/
find "$OUT"/result -name 'false_predictions.txt' -delete
find "$OUT"/result -name 'reasoning_log.jsonl' -delete   # raw per-run XML chains, regenerable
find "$OUT"/result -name 'predictions.csv' -delete       # raw per-essay outputs, regenerable

# --- demo web app (GR_demo, sibling repo) ---
# Self-contained runnable demo: FastAPI backend + JS frontend reusing the pipeline.
# Bundles ONLY the runtime index it loads (essays_dual) + test50 dropdown source.
DEMO_SRC=../GR_demo
if [ -d "$DEMO_SRC" ]; then
  mkdir -p "$OUT"/demo
  cp -r "$DEMO_SRC"/apps "$DEMO_SRC"/packages "$OUT"/demo/
  cp "$DEMO_SRC"/pyproject.toml "$DEMO_SRC"/uv.lock "$DEMO_SRC"/Makefile \
     "$DEMO_SRC"/demo_webapp_requirements.md "$DEMO_SRC"/README.md "$OUT"/demo/ 2>/dev/null || true
  # runtime data only (prebuilt index + dropdown essays); raw/processed/profile_db not needed
  mkdir -p "$OUT"/demo/data/split/essays "$OUT"/demo/data/vector_db
  cp "$DEMO_SRC"/data/split/essays/test50.csv "$OUT"/demo/data/split/essays/
  cp -r "$DEMO_SRC"/data/vector_db/essays_dual "$OUT"/demo/data/vector_db/
  printf 'OPENAI_API_KEY=\n' > "$OUT"/demo/.env.example
  rm -rf "$OUT"/demo/apps/backend/logs
fi

# --- prune junk: caches, checkpoints, backups ---
find "$OUT" -type d -name '__pycache__' -exec rm -rf {} + 2>/dev/null || true
find "$OUT" -type d -name '.ipynb_checkpoints' -exec rm -rf {} + 2>/dev/null || true
find "$OUT" -name '*.bkp' -delete 2>/dev/null || true
find "$OUT" -name '*.pyc' -delete 2>/dev/null || true

echo "=== submission built ==="
du -sh "$OUT"
echo "--- top-level sizes ---"
du -sh "$OUT"/* | sort -rh
