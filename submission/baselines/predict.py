import os
import time
import pandas as pd

from baselines.prompt import *
from utils.gpt_client import gpt_call
from utils.hf_client import hf_call
from utils.log import log_to_file
from utils.parser import extract_direct, extract_cot


TRAIT_TO_COLUMN = {
    "Openness": "pred_cOPN",
    "Conscientiousness": "pred_cCON",
    "Extraversion": "pred_cEXT",
    "Agreeableness": "pred_cAGR",
    "Neuroticism": "pred_cNEU",
}


def predict_text(
    text: str,
    trait_name: str,
    model_name: str,
    usr_prompt: str,
    max_new_tokens: int,
    record_idx: int,
    log_filepath: str,
    temperature: float,
):
    formatted_prompt = usr_prompt.replace("<text>", text)

    if model_name.startswith("gpt"):
        output = gpt_call(
            formatted_prompt,
            SYS_PROMPT,
            model_name,
            max_new_tokens,
            temperature,
        )
    else:
        output = hf_call(
            formatted_prompt,
            SYS_PROMPT,
            model_name,
            max_new_tokens,
            temperature,
        )

    log_to_file(
        log_filepath,
        SYS_PROMPT,
        formatted_prompt,
        output,
        f"{record_idx}-{trait_name}",
    )

    return output


def build_prompt(
    prompt_mode: str,
    trait_name: str,
    example_df: pd.DataFrame = None,
):
    trait_info = TRAITS[trait_name]

    if prompt_mode == "zeroshot":
        return ZEROSHOT_USR_PROMPT.format(
            trait_name=trait_name,
            high_definition=trait_info["high"],
            low_definition=trait_info["low"],
        )

    if prompt_mode == "cot":
        return COT_USR_PROMPT.format(
            trait_name=trait_name,
            high_definition=trait_info["high"],
            low_definition=trait_info["low"],
        )

    if prompt_mode == "oneshot":
        sample = example_df.sample(1).iloc[0]

        label_map = {
            "Openness": sample["cOPN"],
            "Conscientiousness": sample["cCON"],
            "Extraversion": sample["cEXT"],
            "Agreeableness": sample["cAGR"],
            "Neuroticism": sample["cNEU"],
        }

        return ONESHOT_USR_PROMPT.format(
            trait_name=trait_name,
            high_definition=trait_info["high"],
            low_definition=trait_info["low"],
            example_text=sample["text"],
            example_label=label_map[trait_name],
        )

    raise ValueError(f"Unsupported prompt mode: {prompt_mode}")


def predict(
    text_df: pd.DataFrame,
    model_name: str,
    log_dir: str,
    prompt_mode: str,
    max_new_tokens: int,
    res_dir: str,
    temperature: float,
    example_df: pd.DataFrame = None,
):
    run_id = time.strftime("%Y%m%d-%H%M%S")

    log_filepath = os.path.join(
        log_dir,
        model_name,
        prompt_mode,
        f"{run_id}_log.txt",
    )

    output_dir = os.path.join(
        res_dir,
        model_name,
        prompt_mode,
        run_id,
    )

    os.makedirs(output_dir, exist_ok=True)

    df = text_df.copy()

    for col in TRAIT_TO_COLUMN.values():
        df[col] = None

    n = len(df)
    print(f"Loaded {n} records.")

    t0 = time.time()

    for idx, row in df.iterrows():
        text = row["text"]

        for trait_name, pred_col in TRAIT_TO_COLUMN.items():
            usr_prompt = build_prompt(
                prompt_mode=prompt_mode,
                trait_name=trait_name,
                example_df=example_df,
            )

            output = predict_text(
                text=text,
                trait_name=trait_name,
                model_name=model_name,
                usr_prompt=usr_prompt,
                max_new_tokens=max_new_tokens,
                record_idx=idx,
                log_filepath=log_filepath,
                temperature=temperature,
            )

            normalized_output = output.strip()

            if prompt_mode in ("zeroshot", "oneshot"):
                prediction = extract_direct(normalized_output)
            else:
                prediction = extract_cot(normalized_output)

            df.at[idx, pred_col] = prediction

        if (idx + 1) % 10 == 0:
            print(f"Processed {idx + 1}/{n} records.")

    elapsed = time.time() - t0

    prediction_filepath = os.path.join(output_dir, "predictions.csv")
    df.to_csv(prediction_filepath, index=False)

    print(f"Finished in {elapsed:.2f} seconds.")
    print(f"Saved to: {prediction_filepath}")

    return run_id, elapsed, prediction_filepath