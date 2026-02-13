#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
from pathlib import Path
from typing import Any, Iterable
import datasets

from omegaconf import OmegaConf


def _ensure_list(obj: Any) -> list:
    if obj is None:
        return []
    if isinstance(obj, list):
        return obj
    return [obj]


def _wrap_think(text: str | None) -> str:
    text = text.rstrip("\n")
    return f"<think>\n{text}\n</think>\n\n"


def _normalize_tool_calls(tool_calls: Any) -> str:
    calls = _ensure_list(tool_calls)
    normalized: list[dict[str, Any]] = []
    for call in calls:
        if not isinstance(call, dict):
            continue
        if isinstance(call.get("function"), dict):
            inner = call["function"]
            name = inner.get("name")
            arguments = inner.get("arguments", {})
        else:
            name = call.get("name")
            arguments = call.get("arguments", {})
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except Exception:
                pass
        normalized.append({"name": name, "arguments": arguments})
    if len(normalized) == 1:
        return json.dumps(normalized[0], ensure_ascii=False)
    return json.dumps(normalized, ensure_ascii=False)


def _normalize_tools(tools: Any) -> str:
    """Normalize tools into a JSON string of plain function objects list.

    Accepts one of:
      - list[ {"name":..., "parameters":...} ] (already plain)
      - list[ {"type":"function", "function": {"name":..., ...}} ] (OpenAI style)
      - JSON string of either forms
    Returns: json.dumps(list_of_plain_function_objects)
    """
    try:
        if isinstance(tools, str):
            tools = json.loads(tools)
    except Exception:
        pass

    tool_list = _ensure_list(tools)
    normalized: list[dict[str, Any]] = []
    for tool in tool_list:
        if isinstance(tool, dict) and tool.get("type") == "function" and isinstance(tool.get("function"), dict):
            normalized.append(tool["function"])  # unwrap
        else:
            normalized.append(tool)

    return json.dumps(normalized, ensure_ascii=False)


def convert_messages_to_conversations(messages: list[dict[str, Any]]) -> list[dict[str, str]]:
    conversations: list[dict[str, str]] = []
    # 如果conversion的第一个不是system，那么补上：
    SYSTEM_PROMPT = "You are a math expert. You are given a question and you need to solve it step by step with logical reasoning. If needed, use the code interpreter to perform calculations or solve equations. Before any tool call, reason through the problem step by step."
    if messages[0]["role"] != "system":
        messages.insert(0, {"role": "system", "content": SYSTEM_PROMPT})
    
    messages = messages[1:]

    for msg in messages:
        role = msg.get("role")
        content = msg.get("content", "") or ""
        reasoning = msg.get("reasoning_content") or msg.get("reasoning") or ""
        think = _wrap_think(reasoning)

        # Map tool messages directly
        if role in ("tool", "observation"):
            conversations.append({"role": "tool", "content": str(content)})
            continue

        # Convert assistant with tool_calls into function message
        if role == "assistant" and msg.get("tool_calls"):
            tool_payload = _normalize_tool_calls(msg.get("tool_calls"))

            conversations.append({"role": "function", "content": think + tool_payload})
            # If assistant also provides final content in the same step, keep it
            final_answer = (content or "").strip()
            if final_answer:
                conversations.append({"role": "assistant", "content": think + final_answer})
            continue

        # Regular assistant / user / system messages
        if role == "assistant":
            conversations.append({"role": "assistant", "content": think + str(content)})
        elif role == "user":
            conversations.append({"role": "user", "content": str(content)})
        elif role == "system":
            conversations.append({"role": "system", "content": str(content)})
        else:
            # Unknown role: skip silently to be robust
            continue

    return conversations


def _load_items(input_path: Path) -> Iterable[dict[str, Any]]:
    if input_path.suffix.lower() == ".jsonl":
        with input_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                yield json.loads(line)
    else:
        data = json.loads(input_path.read_text(encoding="utf-8"))
        if isinstance(data, list):
            for item in data:
                yield item
        else:
            yield data


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert data with reasoning_content/tool_calls to ShareGPT for qwen3")
    # 输入源：文件 或 HF 数据集 二选一
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--input", type=str, default=None, help="Path to input JSON/JSONL file")
    group.add_argument("--hf_dataset", type=str, default=None, help="HF dataset repo id, e.g., JoeYing/ReTool-SFT")
    parser.add_argument("--hf_subset", type=str, default=None, help="Optional subset name of HF dataset")
    parser.add_argument("--split", type=str, default="train", help="Split name when using HF dataset (default: train)")

    tools_config_file = "scripts/tools.yaml"
    tools_config = OmegaConf.load(tools_config_file)
    tool_schema = OmegaConf.to_container(tools_config["tools"][0]["tool_schema"])
    tools = json.dumps([tool_schema])

    # 输出 Parquet：支持位置参数或 --output
    # parser.add_argument("output", nargs="?", default=None, help="Path to output Parquet file")
    parser.add_argument("--output", dest="output_opt", type=str, default=None, help="Path to output Parquet file")
    args = parser.parse_args()

    out_value = args.output_opt or args.output
    if not out_value:
        raise SystemExit("Please specify output path via positional 'output' or --output.")
    output_path = Path(out_value)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, Any]] = []

    if args.hf_dataset:
        ds = datasets.load_dataset(path=args.hf_dataset, name=args.hf_subset, split=args.split)
        for row in ds:
            msgs = (
                row.get("messages")
                or row.get("conversations")
                or row.get("dialogue")
                or row.get("conversation")
            )
            if isinstance(msgs, list) and (len(msgs) == 0 or isinstance(msgs[0], dict) and "role" in msgs[0]):
                conversations = convert_messages_to_conversations(msgs)
                obj: dict[str, Any] = {"conversations": conversations}
                tools_inline = row.get("tools", None)
                if tools_inline is not None:
                    obj["tools"] = _normalize_tools(tools_inline)
                elif tools is not None:
                    obj["tools"] = tools
                records.append(obj)
            else:
                continue
    else:
        input_path = Path(args.input)
        for item in _load_items(input_path):
            msgs = (
                item.get("messages")
                or item.get("conversations")
                or item.get("dialogue")
                or item.get("conversation")
                or item
            )
            if isinstance(msgs, list) and (len(msgs) == 0 or isinstance(msgs[0], dict) and "role" in msgs[0]):
                conversations = convert_messages_to_conversations(msgs)
                obj: dict[str, Any] = {"conversations": conversations}
                tools_inline = item.get("tools", None)
                if tools_inline is not None:
                    obj["tools"] = _normalize_tools(tools_inline)
                elif tools is not None:
                    obj["tools"] = tools
                records.append(obj)
            else:
                continue

    if len(records) == 0:
        raise RuntimeError("No valid records found to write.")

    print(f"Writing {len(records)} records to {output_path}")
    # 随便打印一个demo
    print(records[0])
    ds = datasets.Dataset.from_list(records)
    ds.to_parquet(str(output_path))


if __name__ == "__main__":
    # python scripts/convert_qwen3_thinking_wo_tools.py --input gpt-oss-20b-reasoning/merged.jsonl --output data/my_qwen3_thinking_tools.parquet
    main()
