import gc
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch
from dotenv import load_dotenv

load_dotenv()
def get_HF_pipeline(model_name: str, max_new_tokens: int, temperature: float):
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        attn_implementation="sdpa",
    )
    generation_config = {
        "max_new_tokens": max_new_tokens,
        "temperature": temperature,
        "do_sample": True if temperature > 0 else False,
        "pad_token_id": tokenizer.pad_token_id,
    }
    return tokenizer, model, generation_config

def create_message(system_prompt_str: str, user_prompt_str: str):
    return [
        {"role": "system", "content": system_prompt_str},
        {"role": "user", "content": user_prompt_str},
    ]
    
def hf_call(
    user_prompt: str,
    system_prompt: str,
    model_name: str,
    max_new_tokens: int,
    temperature: float,
):
    tokenizer, model, generation_config = get_HF_pipeline(
        model_name, max_new_tokens, temperature
    )
    message = create_message(system_prompt, user_prompt)
    prompt = tokenizer.apply_chat_template(
        message, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    input_length = inputs["input_ids"].shape[-1]

    with torch.no_grad():
        outputs = model.generate(**inputs, **generation_config)

    generated_tokens = outputs[0][input_length:]
    content = tokenizer.decode(generated_tokens, skip_special_tokens=True)

    del inputs
    del outputs
    del model
    del tokenizer
    torch.cuda.empty_cache()
    gc.collect()
    return content
