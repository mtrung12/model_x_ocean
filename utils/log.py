from typing import List, Union
import os

import pandas as pd


def log_to_file(
    log_filepath: str,
    system_prompt: str,
    user_prompt: str,
    response: Union[str, List[str]],
    record_idx: int,
):
    os.makedirs(os.path.dirname(log_filepath), exist_ok=True)
    header = f"[Record number {record_idx}]\n"

    with open(log_filepath, "a", encoding="utf-8") as f:
        f.write(header)
        f.write("-" * 80 + "\n")
        f.write(f"--- SYSTEM PROMPT ---\n{system_prompt}\n\n")
        f.write(f"--- USER PROMPT ---\n{user_prompt}\n")
        f.write("\n")
        response_str = str(response)
        f.write(f"--- LLM RESPONSE ---\n{response_str}\n")
        f.write("=" * 80 + "\n\n")


def write_classification_report(
    report_path: str,
    save_df,
    model_name: str,
    test_csv: str,
    n_records: int,
    fail_count: int,
    prompt_mode: str,
    vector_db_dir: str,
    time: float,
    dataset: str = "N/A",
):
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    if isinstance(save_df, dict):
        save_df = pd.DataFrame(save_df).transpose()

    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"{'=' * 70}\n")
        f.write("Personality Trait Detection - Classification Report\n")
        f.write(f"{'=' * 70}\n\n")
        f.write(f"Model            : {model_name}\n")
        f.write(f"Test file        : {test_csv}\n")
        f.write(f"Dataset          : {dataset}\n")
        f.write(f"# Records        : {n_records}\n")
        f.write(f"# Failed         : {fail_count}\n")
        f.write(f"Prompt mode      : {prompt_mode}\n")
        f.write(f"Vector DB        : {vector_db_dir}\n")
        f.write(f"Prediction Time (sec) : {time:.2f}\n")
        f.write(f"{'-' * 70}\n\n")

        if save_df is None or getattr(save_df, "empty", False):
            f.write("No valid prediction rows were available for this report.\n\n")
        else:
            metric_cols = ["precision", "recall", "f1-score", "support"]
            metric_cols = [col for col in metric_cols if col in save_df.columns]
            printable_df = save_df.copy()

            for col in ["precision", "recall", "f1-score"]:
                if col in printable_df.columns:
                    printable_df[col] = printable_df[col].map(lambda x: f"{float(x):.4f}")
            if "support" in printable_df.columns:
                printable_df["support"] = printable_df["support"].map(
                    lambda x: f"{float(x):.0f}"
                )

            f.write(printable_df[metric_cols].to_string())
            f.write("\n\n")

        f.write(f"{'=' * 70}\n")
        f.write("End of Report\n")
