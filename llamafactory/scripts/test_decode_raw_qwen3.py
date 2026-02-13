#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


def load_model(model_path: str, adapter_path: str | None = None):
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
        trust_remote_code=True,
    )

    if adapter_path:
        try:
            from peft import PeftModel

            model = PeftModel.from_pretrained(model, adapter_path)
        except Exception as e:
            raise RuntimeError(f"Failed to load adapter from {adapter_path}: {e}")

    model.eval()
    return tokenizer, model


def run_one(tokenizer, model, user_prompt: str, system_prompt: str | None = None, max_new_tokens: int = 256):
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_prompt})

    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_tensors="pt",
    ).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            eos_token_id=tokenizer.eos_token_id,
        )

    full_text = tokenizer.decode(outputs[0], skip_special_tokens=False)
    new_text = tokenizer.decode(outputs[0][inputs.shape[1]:], skip_special_tokens=False)

    print("=== PROMPT ===")
    print(user_prompt)
    print("=== FULL DECODE (with specials) ===")
    print(full_text)
    print("=== NEW DECODE (with specials) ===")
    print(new_text)


def main():
    parser = argparse.ArgumentParser(description="Test raw decode (including special tokens) for Qwen3 chat")
    parser.add_argument("--model", type=str, required=True, help="Base model path or HF repo id, e.g., Qwen/Qwen3-1.7B")
    parser.add_argument("--adapter", type=str, default=None, help="Optional LoRA adapter path")
    parser.add_argument("--max_new_tokens", type=int, default=256)
    args = parser.parse_args()

    tokenizer, model = load_model(args.model, args.adapter)

    system = "You are a helpful assistant."

    # Test 1: arithmetic
    run_one(tokenizer, model, "Compute 12345 * 6789.", system, max_new_tokens=args.max_new_tokens)

    # Test 2: summation
    run_one(tokenizer, model, "What is the sum of the first 100 integers?", system, max_new_tokens=args.max_new_tokens)


if __name__ == "__main__":
    main()


