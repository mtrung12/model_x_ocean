import os
from typing import Dict

import pandas as pd
from sklearn.metrics import accuracy_score, classification_report

from utils.log import write_classification_report


TRAIT_COLUMNS = {
    "Openness": "cOPN",
    "Conscientiousness": "cCON",
    "Extraversion": "cEXT",
    "Agreeableness": "cAGR",
    "Neuroticism": "cNEU",
}


def build_report_df(y_true, y_pred) -> pd.DataFrame:
    report = classification_report(
        y_true,
        y_pred,
        output_dict=True,
        zero_division=0,
    )
    return pd.DataFrame(report).transpose()


def evaluate(
    prediction_csv: str,
    model_name: str,
    run_time: float,
    prompt_mode: str,
    res_dir: str,
    run_id: str,
    vector_db_dir: str = None,
    dataset: str = "N/A",
) -> Dict[str, object]:
    df = pd.read_csv(prediction_csv)

    required_cols = []
    for short_col in TRAIT_COLUMNS.values():
        required_cols.extend([short_col, f"pred_{short_col}"])

    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        raise ValueError(
            f"Missing required columns: {', '.join(missing_cols)}"
        )

    n_records = len(df)

    pred_cols = [f"pred_{c}" for c in TRAIT_COLUMNS.values()]
    normalized_preds = (
        df[pred_cols]
        .astype("string")
        .apply(lambda col: col.str.strip().str.lower())
    )

    fail_mask = (
        normalized_preds.isna().any(axis=1)
        | normalized_preds.eq("").any(axis=1)
        | ~normalized_preds.isin(["high", "low"]).all(axis=1)
    )

    fail_count = int(fail_mask.sum())

    summary_rows = []
    report_paths = {}

    output_dir = os.path.join(
        res_dir,
        model_name,
        prompt_mode,
        run_id,
    )
    os.makedirs(output_dir, exist_ok=True)

    for trait_name, true_col in TRAIT_COLUMNS.items():
        pred_col = f"pred_{true_col}"

        valid_df = df[[true_col, pred_col]].copy()

        valid_df[true_col] = (
            valid_df[true_col]
            .astype(str)
            .str.strip()
            .str.lower()
        )

        valid_df[pred_col] = (
            valid_df[pred_col]
            .astype(str)
            .str.strip()
            .str.lower()
        )

        valid_df = valid_df[
            valid_df[pred_col].isin(["high", "low"])
        ]

        y_true = valid_df[true_col]
        y_pred = valid_df[pred_col]

        if len(valid_df) == 0:
            accuracy = 0.0
            report_df = pd.DataFrame()
        else:
            accuracy = accuracy_score(y_true, y_pred)
            report_df = build_report_df(y_true, y_pred)

        report_path = os.path.join(
            output_dir,
            f"{trait_name}_classification_report.txt",
        )

        write_classification_report(
            report_path=report_path,
            save_df=report_df,
            model_name=model_name,
            test_csv=prediction_csv,
            n_records=n_records,
            fail_count=fail_count,
            prompt_mode=prompt_mode,
            vector_db_dir=vector_db_dir or "N/A",
            time=run_time,
            dataset=dataset,
        )

        report_paths[trait_name] = report_path

        summary_rows.append({
            "trait": trait_name,
            "n_samples": len(valid_df),
            "accuracy": accuracy,
            "macro_precision": (
                report_df.loc["macro avg", "precision"]
                if "macro avg" in report_df.index else None
            ),
            "macro_recall": (
                report_df.loc["macro avg", "recall"]
                if "macro avg" in report_df.index else None
            ),
            "macro_f1": (
                report_df.loc["macro avg", "f1-score"]
                if "macro avg" in report_df.index else None
            ),
            "weighted_precision": (
                report_df.loc["weighted avg", "precision"]
                if "weighted avg" in report_df.index else None
            ),
            "weighted_recall": (
                report_df.loc["weighted avg", "recall"]
                if "weighted avg" in report_df.index else None
            ),
            "weighted_f1": (
                report_df.loc["weighted avg", "f1-score"]
                if "weighted avg" in report_df.index else None
            ),
        })

    summary_df = pd.DataFrame(summary_rows)

    summary_path = os.path.join(
        output_dir,
        "evaluation_summary.csv",
    )

    summary_df.to_csv(summary_path, index=False)

    print(f"Loaded predictions from {prediction_csv}")
    print(f"Saved evaluation summary to {summary_path}")

    for trait_name, report_path in report_paths.items():
        print(f"Saved {trait_name} report to {report_path}")

    return {
        "summary_csv": summary_path,
        "report_paths": report_paths,
        "n_records": n_records,
        "fail_count": fail_count,
    }