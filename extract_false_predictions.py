"""
Extract false predictions from a predictions.csv and the corresponding log file.

CLI usage:
    python extract_false_predictions.py \
        --log log/gpt-4o-mini/reasoned_rag_def_oneshot_30f/20260523-221029_log.txt \
        --pred result/gpt-4o-mini/reasoned_rag_def_oneshot_30f/20260523-221029/predictions.csv \
        [--out output_dir] \
        [--max 50]

Notebook usage:
    from extract_false_predictions import extract_false_predictions
    extract_false_predictions(log_path=..., pred_path=..., out_dir=..., max_per_file=50)

Output (6 markdown files):
    false_Openness.md / false_Conscientiousness.md / false_Extraversion.md
    false_Agreeableness.md / false_Neuroticism.md
    false_all5.md  — rows where ALL 5 traits were predicted wrong
"""

import argparse
import csv
import re
import sys
from pathlib import Path


TRAITS = ["Openness", "Conscientiousness", "Extraversion", "Agreeableness", "Neuroticism"]

GT_COLS = {
    "Openness": "cOPN", "Conscientiousness": "cCON",
    "Extraversion": "cEXT", "Agreeableness": "cAGR", "Neuroticism": "cNEU",
}
PRED_COLS = {
    "Openness": "pred_cOPN", "Conscientiousness": "pred_cCON",
    "Extraversion": "pred_cEXT", "Agreeableness": "pred_cAGR", "Neuroticism": "pred_cNEU",
}

_BLOCK_SEP   = "=" * 80
_LLM_MARKER  = "--- LLM RESPONSE ---"
_TEXT_MARKER = "Text to classify:"


# ---------------------------------------------------------------------------
# Core parsing
# ---------------------------------------------------------------------------

def parse_log_blocks(log_path: str | Path) -> dict[tuple[int, str], dict]:
    """Parse a log file into {(record_idx, trait): {"input_text": str, "llm_response": str}}."""
    header_re = re.compile(r"^\[Record number (\d+)-(\w+)\]")
    blocks: dict[tuple[int, str], dict] = {}
    current_key = None
    current_lines: list[str] = []

    with open(log_path, encoding="utf-8") as fh:
        lines = fh.readlines()

    def _flush(key, raw_lines):
        raw = "".join(raw_lines)

        input_text = ""
        text_idx = raw.find(_TEXT_MARKER)
        if text_idx != -1:
            after = raw[text_idx + len(_TEXT_MARKER):]
            end = after.find("\n\nReason step by step")
            if end == -1:
                end = after.find("\n\nDo not output")
            input_text = after[:end].strip() if end != -1 else after.strip()

        llm_response = ""
        llm_idx = raw.find(_LLM_MARKER)
        if llm_idx != -1:
            after_llm = raw[llm_idx + len(_LLM_MARKER):]
            sep_idx = after_llm.find(_BLOCK_SEP)
            llm_response = after_llm[:sep_idx].strip() if sep_idx != -1 else after_llm.strip()

        blocks[key] = {"input_text": input_text, "llm_response": llm_response}

    for line in lines:
        m = header_re.match(line)
        if m:
            if current_key is not None:
                _flush(current_key, current_lines)
            current_key = (int(m.group(1)), m.group(2))
            current_lines = [line]
        else:
            if current_key is not None:
                current_lines.append(line)

    if current_key is not None:
        _flush(current_key, current_lines)

    return blocks


def find_false_predictions(pred_path: str | Path) -> list[tuple[int, dict, list[str]]]:
    """Return [(row_idx, row_dict, [wrong_trait, ...])] for every row with at least one error."""
    results = []
    with open(pred_path, encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        for row_idx, row in enumerate(reader):
            wrong = [
                trait for trait in TRAITS
                if (gt := row.get(GT_COLS[trait], "").strip().lower())
                and (pred := row.get(PRED_COLS[trait], "").strip().lower())
                and gt != pred
            ]
            if wrong:
                results.append((row_idx, row, wrong))
    return results


# ---------------------------------------------------------------------------
# Markdown rendering
# ---------------------------------------------------------------------------

def _render_trait_entry(row_idx: int, row: dict, trait: str, block: dict | None) -> str:
    gt   = row.get(GT_COLS[trait],   "?")
    pred = row.get(PRED_COLS[trait], "?")
    input_text   = block["input_text"]   if block else "_Log block not found._"
    llm_response = block["llm_response"] if block else ""
    return "\n".join([
        f"## Sample {row_idx}  |  {trait}  |  gt=`{gt}`  pred=`{pred}`",
        "",
        "### Input text",
        "",
        input_text,
        "",
        "### LLM response",
        "",
        "```",
        llm_response,
        "```",
        "",
        "---",
        "",
    ])


def _render_all5_entry(row_idx: int, row: dict, log_blocks: dict) -> str:
    parts = [f"## Sample {row_idx}  |  ALL 5 TRAITS WRONG", ""]
    parts += ["| Trait | Ground truth | Predicted |", "|---|---|---|"]
    for trait in TRAITS:
        gt   = row.get(GT_COLS[trait],   "?")
        pred = row.get(PRED_COLS[trait], "?")
        parts.append(f"| {trait} | `{gt}` | `{pred}` |")
    parts.append("")

    for trait in TRAITS:
        block = log_blocks.get((row_idx, trait))
        gt   = row.get(GT_COLS[trait],   "?")
        pred = row.get(PRED_COLS[trait], "?")
        input_text   = block["input_text"]   if block else "_Log block not found._"
        llm_response = block["llm_response"] if block else ""
        parts += [
            f"### {trait}  |  gt=`{gt}`  pred=`{pred}`",
            "",
            "**Input text**",
            "",
            input_text,
            "",
            "**LLM response**",
            "",
            "```",
            llm_response,
            "```",
            "",
        ]
    parts += ["---", ""]
    return "\n".join(parts)


def _write_md(path: Path, title: str, entries: list[str]) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(f"# {title}\n\n")
        fh.write(f"Total samples: {len(entries)}\n\n")
        fh.write("---\n\n")
        fh.write("\n".join(entries))
    print(f"Saved: {path}  ({len(entries)} samples)")


# ---------------------------------------------------------------------------
# Main API
# ---------------------------------------------------------------------------

def extract_false_predictions(
    log_path: str | Path,
    pred_path: str | Path,
    out_dir: str | Path | None = None,
    max_per_file: int = 247,
) -> dict:
    """
    Parse false predictions from a log + predictions CSV and write 6 markdown files.

    Parameters
    ----------
    log_path     : path to the _log.txt file
    pred_path    : path to predictions.csv
    out_dir      : output directory (default: same folder as predictions.csv)
    max_per_file : max samples written per output file (default 247 = all)

    Returns
    -------
    dict with keys:
        per_trait   — {trait: count_of_wrong_predictions}
        all5_count  — number of samples wrong on all 5 traits
        out_dir     — Path of the output directory
    """
    log_path  = Path(log_path)
    pred_path = Path(pred_path)
    out_dir   = Path(out_dir) if out_dir else pred_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Parsing log blocks from:  {log_path}")
    log_blocks = parse_log_blocks(log_path)
    print(f"  -> {len(log_blocks)} blocks parsed")

    print(f"Reading predictions from: {pred_path}")
    false_rows = find_false_predictions(pred_path)
    print(f"  -> {len(false_rows)} rows with at least one wrong prediction")

    # Bucket by trait and by all-5
    per_trait: dict[str, list] = {t: [] for t in TRAITS}
    all5: list = []

    for row_idx, row, wrong_traits in false_rows:
        for trait in wrong_traits:
            per_trait[trait].append((row_idx, row, log_blocks.get((row_idx, trait))))
        if len(wrong_traits) == 5:
            all5.append((row_idx, row))

    # Per-trait files
    for trait in TRAITS:
        samples = per_trait[trait][:max_per_file]
        _write_md(
            out_dir / f"false_{trait}.md",
            f"False predictions — {trait}",
            [_render_trait_entry(ri, row, trait, blk) for ri, row, blk in samples],
        )

    # All-5 file
    _write_md(
        out_dir / "false_all5.md",
        "False predictions — ALL 5 traits wrong",
        [_render_all5_entry(ri, row, log_blocks) for ri, row in all5[:max_per_file]],
    )

    counts = {t: len(per_trait[t]) for t in TRAITS}
    print("\nSummary:")
    for trait, n in counts.items():
        print(f"  {trait:<20}: {n} wrong predictions")
    print(f"  {'All 5 wrong':<20}: {len(all5)} samples")

    return {"per_trait": counts, "all5_count": len(all5), "out_dir": out_dir}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract false predictions as markdown reports.")
    parser.add_argument("--log",  required=True)
    parser.add_argument("--pred", required=True)
    parser.add_argument("--out",  default=None)
    parser.add_argument("--max",  type=int, default=247)
    args = parser.parse_args()

    for p, label in [(args.log, "Log"), (args.pred, "Predictions")]:
        if not Path(p).exists():
            sys.exit(f"{label} file not found: {p}")

    extract_false_predictions(
        log_path=args.log,
        pred_path=args.pred,
        out_dir=args.out,
        max_per_file=args.max,
    )
