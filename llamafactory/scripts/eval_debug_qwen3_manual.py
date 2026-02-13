#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import os
import re
import subprocess
import tempfile
from typing import List, Dict, Optional, Any

import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from llamafactory.data.tool_utils import QwenToolUtils


QWEN3_SYSTEM = "<|im_start|>system\n{content}<|im_end|>\n"
QWEN3_USER = "<|im_start|>user\n{content}<|im_end|>\n<|im_start|>assistant\n"


def _normalize_tools(tools: Any) -> list[dict]:
    try:
        if isinstance(tools, str):
            tools = json.loads(tools)
    except Exception:
        pass
    if tools is None:
        return []
    return tools if isinstance(tools, list) else [tools]


def build_qwen3_prompt_from_first_turn(conversations: List[Dict[str, str]], tools: Optional[Any]) -> Optional[str]:
    system_msg = ""
    user_msg = None

    if len(conversations) and conversations[0].get("role") == "system":
        system_msg = conversations[0].get("content", "")

    # append tool description into system (Qwen format)
    tool_list = _normalize_tools(tools)
    if len(tool_list) > 0:
        tool_text = QwenToolUtils.tool_formatter(tool_list)
        system_msg = QWEN3_SYSTEM.format(content=(system_msg.strip() + tool_text))

    for msg in conversations:
        if msg.get("role") == "user":
            # user_msg = msg.get("content", "")
            user_msg = "On $\\triangle ABC$ points $A,D,E$, and $B$ lie that order on side $\\overline{AB}$ with $AD=4, DE=16$, and $EB=8$. Points $A,F,G$, and $C$ lie in that order on side $\\overline{AC}$ with $AF=13, FG=52$, and $GC=26$. Let $M$ be the reflection of $D$ through $F$, and let $N$ be the reflection of $G$ through $E$. Quadrilateral $DEGF$ has area 288. Find the area of heptagon $AFNBCEM$."
            # user_msg = "Find the sum of all integer bases $b>9$ for which $17_{b}$ is a divisor of $97_{b}$."
            # user_msg = "The 9 members of a baseball team went to an ice cream parlor after their game. Each player had a singlescoop cone of chocolate, vanilla, or strawberry ice cream. At least one player chose each flavor, and the number of players who chose chocolate was greater than the number of players who chose vanilla, which was greater than the number of players who chose strawberry. Let $N$ be the number of different assignments of flavors to players that meet these conditions. Find the remainder when $N$ is divided by 1000."
            break

    if user_msg is None:
        return None

    return system_msg + QWEN3_USER.format(content=user_msg)


def load_model(model_path: str):
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    return tokenizer, model


def execute_python(code: str, timeout: int = 30) -> str:
    print("代码为：")
    print(code)

    has_print = 'print(' in code or 'print ' in code
    if not has_print:
        lines = code.strip().split('\n')
        if lines:
            last_line = lines[-1].strip()
            if (
                last_line
                and not last_line.startswith(('#', 'def ', 'class ', 'if ', 'for ', 'while ', 'try:', 'except', 'finally:', 'with ', 'import ', 'from '))
                and '=' not in last_line.split('#')[0]
                and not last_line.endswith(':')
                and not last_line.startswith('return ')
            ):
                lines[-1] = f"print({last_line})"
                code = '\n'.join(lines)
                print("自动添加print到最后一行的表达式：")
                print(code)

    try:
        compile(code, '<string>', 'exec')
    except SyntaxError as e:
        return f"Syntax error: {e}"

    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False, encoding='utf-8') as f:
        f.write(code)
        temp_file = f.name

    try:
        result = subprocess.run(
            ['python3', temp_file], capture_output=True, text=True, timeout=timeout
        )
        return (f"{result.stdout}" if result.returncode == 0 else f"error:{result.stderr}")
    except subprocess.TimeoutExpired:
        return f"timeout({timeout} seconds)"
    except FileNotFoundError:
        try:
            result = subprocess.run(
                ['python', temp_file], capture_output=True, text=True, timeout=timeout
            )
            return (f"{result.stdout}" if result.returncode == 0 else f"error:\n{result.stderr}")
        except Exception as e:
            return f"error: {e}"
    except Exception as e:
        return f"error: {e}"
    finally:
        try:
            os.unlink(temp_file)
        except Exception:
            pass


def main():
    parser = argparse.ArgumentParser(description="Eval debug without chat_template (Qwen3 first-turn only)")
    parser.add_argument("--model", type=str, required=True, help="Model path (your full-finetuned output_dir)")
    parser.add_argument("--dataset", type=str, required=True, help="Parquet path with 'conversations' column")
    parser.add_argument("--num_samples", type=int, default=5)
    parser.add_argument("--max_new_tokens", type=int, default=8192)
    args = parser.parse_args()

    tokenizer, model = load_model(args.model)
    ds: Dataset = Dataset.from_parquet(args.dataset)

    for idx in range(min(args.num_samples, len(ds))):
        record = ds[idx]
        conversations = record.get("conversations", [])
        prompt = build_qwen3_prompt_from_first_turn(conversations, record.get("tools"))
        if not prompt:
            continue

        # Step 1: generate assistant with potential <tool_call>
        input_ids = tokenizer.encode(prompt, add_special_tokens=False, return_tensors="pt").to(model.device)
        with torch.no_grad():
            output_ids = model.generate(
                input_ids,
                max_new_tokens=args.max_new_tokens,
                do_sample=True,
                eos_token_id=tokenizer.eos_token_id,
            )

        full_text_1 = tokenizer.decode(output_ids[0], skip_special_tokens=False)
        new_text_1 = tokenizer.decode(output_ids[0][input_ids.shape[1]:], skip_special_tokens=False)

        print(f"\n===== Sample {idx} =====")
        print("PROMPT (manual, with specials):")
        print(prompt)
        print('-'*100)
        print("NEW DECODE (with specials) - TURN 1:")
        print(new_text_1)

        # Try extract tool calls from assistant content
        # Strip <think>...</think> and trailing <|im_end|> before extracting tool calls
        clean_assistant = re.sub(r"<think>.*?</think>\s*", "", new_text_1, flags=re.DOTALL)
        clean_assistant = clean_assistant.replace("<|im_end|>", "")
        calls = QwenToolUtils.tool_extractor(clean_assistant)
        # If extraction fails (returns str), stop here
        if isinstance(calls, str) or not calls:
            continue

        # For simplicity handle first call and only code_interpreter
        tool_name, tool_args_json = calls[0]
        try:
            tool_args = json.loads(tool_args_json)
        except Exception:
            tool_args = {}

        observation = ""
        if tool_name == "code_interpreter":
            code = tool_args.get("code", "")
            observation = execute_python(code)
        else:
            observation = f"error: unsupported tool {tool_name}"

        # Append observation block and continue to get final assistant answer
        obs_block = (
            "<|im_start|>user\n<tool_response>\n" + observation + "\n</tool_response><|im_end|>\n<|im_start|>assistant\n"
        )
        next_prompt = full_text_1 + obs_block

        input_ids_2 = tokenizer.encode(next_prompt, add_special_tokens=False, return_tensors="pt").to(model.device)
        with torch.no_grad():
            output_ids_2 = model.generate(
                input_ids_2,
                max_new_tokens=args.max_new_tokens,
                do_sample=True,
                eos_token_id=tokenizer.eos_token_id,
            )

        new_text_2 = tokenizer.decode(output_ids_2[0][input_ids_2.shape[1]:], skip_special_tokens=False)
        print('-'*100)
        print("OBSERVATION (tool response):")
        print(observation)
        print('-'*100)
        print("NEW DECODE (with specials) - TURN 2:")
        print(new_text_2)


if __name__ == "__main__":
    main()

# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# """
# sglang 版「Qwen3-tool-call」调试脚本
# 用法：
# python sglang_tool.py --model /path/to/qwen3-8b-instruct \
#                       --dataset test.parquet \
#                       --num-samples 5
# """
# import argparse
# import json
# import os
# import re
# import subprocess
# import tempfile
# from typing import List, Dict, Optional, Any

# import sglang as sgl
# from datasets import Dataset
# from llamafactory.data.tool_utils import QwenToolUtils

# # ---------- 模板 ----------
# QWEN3_SYSTEM = "<|im_start|>system\n{content}<|im_end|>\n"
# QWEN3_USER   = "<|im_start|>user\n{content}<|im_end|>\n<|im_start|>assistant\n"


# # ---------- 工具 ----------
# def _normalize_tools(tools: Any) -> List[dict]:
#     try:
#         if isinstance(tools, str):
#             tools = json.loads(tools)
#     except Exception:
#         pass
#     if tools is None:
#         return []
#     return tools if isinstance(tools, list) else [tools]


# def build_prompt(conversations: List[Dict[str, str]],
#                  tools: Optional[Any] = None) -> Optional[str]:
#     """仅取第一轮 user 内容，拼成 qwen3 格式"""
#     system_msg = ""
#     if conversations and conversations[0].get("role") == "system":
#         system_msg = conversations[0].get("content", "").strip()

#     tool_list = _normalize_tools(tools)
#     if tool_list:
#         tool_text = QwenToolUtils.tool_formatter(tool_list)
#         system_msg = QWEN3_SYSTEM.format(content=system_msg + tool_text)

#     for msg in conversations:
#         if msg.get("role") == "user":
#             user_msg = msg["content"]
#             break
#     else:
#         return None
#     return system_msg + QWEN3_USER.format(content=user_msg)


# def execute_python(code: str, timeout: int = 30) -> str:
#     """与原版完全一致"""
#     has_print = "print(" in code or "print " in code
#     if not has_print:
#         lines = code.strip().splitlines()
#         if lines:
#             last = lines[-1].strip()
#             if (
#                 last
#                 and not last.startswith(
#                     ("#", "def ", "class ", "if ", "for ", "while ", "try:", "except", "finally:", "with ", "import ",
#                      "from "))
#                 and "=" not in last.split("#")[0]
#                 and not last.endswith(":")
#                 and not last.startswith("return ")
#             ):
#                 lines[-1] = f"print({last})"
#                 code = "\n".join(lines)

#     with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False, encoding="utf-8") as f:
#         f.write(code)
#         tmp = f.name

#     try:
#         proc = subprocess.run(
#             ["python3", tmp], capture_output=True, text=True, timeout=timeout
#         )
#         return proc.stdout if proc.returncode == 0 else f"error:\n{proc.stderr}"
#     except Exception as e:
#         return f"error: {e}"
#     finally:
#         try:
#             os.unlink(tmp)
#         except Exception:
#             pass


# # ---------- SGLang 主流程 ----------
# @sgl.function
# def multi_turn(s, prompt: str, max_new_tokens: int=32768):
#     """两轮 generate：第一轮带 tool_call，第二轮拼结果"""
#     # 第一轮
#     s += sgl.user(prompt)
#     s += sgl.assistant(sgl.gen("t1", max_tokens=max_new_tokens))

#     # 如果模型返回了 <tool_call>，提取并执行
#     raw1 = s["t1"]
#     print("raw1:")
#     print(raw1)
#     print("-"*100)
#     clean = re.sub(r"<think>.*?</think>\s*", "", raw1, flags=re.DOTALL).replace("<|im_end|>", "")
#     calls = QwenToolUtils.tool_extractor(clean)

#     if isinstance(calls, list) and calls:
#         tool_name, args_json = calls[0]
#         try:
#             args = json.loads(args_json)
#         except Exception:
#             args = {}
#         obs = ""
#         if tool_name == "code_interpreter":
#             obs = execute_python(args.get("code", ""))
#         else:
#             obs = f"error: unsupported tool {tool_name}"

#         # 把 observation 拼回去
#         obs_block = f"<|im_start|>user\n<tool_response>\n{obs}\n</tool_response><|im_end|>\n<|im_start|>assistant\n"
#         s += obs_block
#         s += sgl.assistant(sgl.gen("t2", max_tokens=max_new_tokens))
#     else:
#         s["t2"] = ""      # 没有 tool_call 就不跑第二轮


# def main():
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--model", required=True, help="HF 格式模型路径")
#     parser.add_argument("--dataset", required=True, help="含 conversations 列的 parquet")
#     parser.add_argument("--num-samples", type=int, default=5)
#     parser.add_argument("--max-new-tokens", type=int, default=8192)
#     parser.add_argument("--tp", type=int, default=1, help="tensor-parallel size")
#     args = parser.parse_args()

#     # 启动 SGLang 服务
#     sgl.set_default_backend(
#         sgl.Runtime(
#             model_path=args.model,
#             tp_size=args.tp,
#             trust_remote_code=True,
#         )
#     )

#     ds: Dataset = Dataset.from_parquet(args.dataset)
#     for idx in range(min(args.num_samples, len(ds))):
#         rec = ds[idx]
#         prompt = build_prompt(rec.get("conversations", []), rec.get("tools"))
#         if not prompt:
#             continue

#         state = multi_turn.run(prompt, max_new_tokens=args.max_new_tokens)

#         print(f"\n===== Sample {idx} =====")
#         print("PROMPT:")
#         print(prompt)
#         print("-" * 80)
#         print("TURN 1 assistant:")
#         print(state["t1"])
#         if state["t2"]:
#             print("-" * 80)
#             print("OBSERVATION:")
#             print(state.get("observation", ""))
#             print("-" * 80)
#             print("TURN 2 assistant:")
#             print(state["t2"])


# if __name__ == "__main__":
#     main()

# """
# python scripts/eval_debug_qwen3_manual.py \
#   --model ckpts/qwen3_1_7b_thinking_sft_full_epoch3_SFT_wo_sysprompt_wo_tools_reasoning \
#   --dataset data/debug.parquet \
#   --num-samples 5 \
#   --tp 1
# """