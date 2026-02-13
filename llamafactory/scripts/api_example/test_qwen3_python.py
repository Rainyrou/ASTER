# Copyright 2025 the LlamaFactory team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import os

from openai import OpenAI
from transformers.utils.versions import require_version
import subprocess
import tempfile
import pandas as pd
import re

require_version("openai>=1.5.0", "To fix: pip install openai>=1.5.0")


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


SYSTEM_PROMPT = """You are a math expert. You are given a question and you need to solve it step by step with logical reasoning. If needed, use the code interpreter to perform calculations or solve equations. Before any tool call, reason through the problem step by step."""


def load_aime25_data():
    """加载AIME25数据集"""
    try:
        # 从本地JSONL加载AIME2025数据集
        path1 = "./aime25/I.jsonl"
        path2 = "./aime25/II.jsonl"

        df1 = pd.read_json(path1, lines=True)
        df2 = pd.read_json(path2, lines=True)

        examples = []
        for _, row in df1.iterrows():
            examples.append({
                "question": row["question"],
                "answer": str(row["answer"]).strip()
            })
        for _, row in df2.iterrows():
            examples.append({
                "question": row["question"],
                "answer": str(row["answer"]).strip()
            })

        print(f"成功加载AIME25数据集，共{len(examples)}道题目")
        return examples
    except Exception as e:
        print(f"加载AIME25数据集失败: {e}")
        # 返回一些示例题目作为备选
        return [
            {
                "question": "Let $A$ be the set of positive integer divisors of $2025$. Let $B$ be a randomly selected subset of $A$. The probability that $B$ is a nonempty set with the property that the least common multiple of its element is $2025$ is $\\frac{m}{n}$, where $m$ and $n$ are relatively prime positive integers. Find $m+n$.",
                "answer": "237"
            },
            {
                "question": "Find the smallest positive integer $n$ such that $n^2$ is divisible by $2025$.",
                "answer": "45"
            }
        ]


def extract_boxed_answer(text: str) -> str:
    """从回答中提取boxed答案"""
    if not text:
        return None

    # 尝试匹配 \boxed{...} 格式
    pattern = r'\\boxed\{([^}]+)\}'
    matches = re.findall(pattern, text)
    if matches:
        return matches[-1].strip()

    # 尝试匹配 boxed{...} 格式（无反斜杠）
    pattern = r'boxed\{([^}]+)\}'
    matches = re.findall(pattern, text)
    if matches:
        return matches[-1].strip()

    # 如果都没有，尝试提取最后一个数字
    pattern = r'\d+'
    matches = re.findall(pattern, text)
    if matches:
        return matches[-1]

    return None


def normalize_answer(answer: str) -> str:
    """标准化答案格式"""
    if not answer:
        return ""

    # 移除多余的空格
    answer = answer.strip()

    # 处理可能的LaTeX格式
    if '\\' in answer:
        answer = re.sub(r'\\[a-zA-Z]+\{([^}]+)\}', r'\1', answer)

    # 再次清理空格
    return answer.strip()


def solve_one_question_multiturn(client: OpenAI, tools, tool_map, question: str, model: str = "test", max_rounds: int = 8):
    """
    使用多轮对话（while循环）求解单个问题。
    支持函数调用工具（code_interpreter），直到模型给出boxed答案或达到最大轮数。
    """
    messages = []
    messages.append({"role": "system", "content": SYSTEM_PROMPT})
    messages.append({"role": "user", "content": question})

    final_content = ""
    boxed = None
    rounds = 0

    while rounds < max_rounds:
        rounds += 1
        result = client.chat.completions.create(messages=messages, model=model, tools=tools, max_tokens=32768)
        msg = result.choices[0].message
        
        print("Round", rounds)
        print(msg)
        print("-"*100)
        # 如果模型请求调用工具
        if getattr(msg, "tool_calls", None):
            messages.append(msg)
            for tc in msg.tool_calls:
                name = tc.function.name
                args_json = tc.function.arguments or "{}"
                try:
                    arguments = json.loads(args_json)
                except Exception:
                    arguments = {}
                if name in tool_map:
                    tool_output = tool_map[name](arguments['code'])
                    # 回传工具结果，携带tool_call_id
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id if hasattr(tc, "id") else None,
                        "content": tool_output,
                    })
                else:
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id if hasattr(tc, "id") else None,
                        "content": f"error: unknown tool {name}",
                    })
            # 工具调用后继续下一轮
            continue

        # 普通assistant文本回复
        content = msg.content or ""
        messages.append(msg)
        final_content = content
        boxed = extract_boxed_answer(content)
        break

    return final_content, (boxed or extract_boxed_answer(final_content)), rounds, messages


def run_aime25_test():
    """运行AIME25多轮对话测试并统计准确率"""
    client = OpenAI(
        api_key="{}".format(os.getenv("API_KEY", "0")),
        base_url="http://localhost:{}/v1".format(os.getenv("API_PORT", 8000)),
    )
    tools = [
        {
            "type": "function",
            "function": {
                "name": "code_interpreter",
                "description": "A tool for executing code.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "code": {"type": "string", "description": "The code to execute"},
                    },
                    "required": ["code"],
                },
            },
        }
    ]
    tool_map = {"code_interpreter": execute_python}

    model = os.getenv("MODEL_NAME", "test")
    max_rounds = int(os.getenv("AIME25_MAX_ROUNDS", "8"))
    limit = os.getenv("AIME25_LIMIT")
    limit = int(limit) if (limit and limit.isdigit()) else None

    data = load_aime25_data()
    if limit:
        data = data[:limit]

    total = len(data)
    correct = 0

    print(f"开始评测：共{total}题，模型={model}，最大轮数={max_rounds}")

    for idx, ex in enumerate(data, 1):
        q = ex["question"]
        gt = normalize_answer(str(ex["answer"]))
        pred_content, pred_boxed, rounds, _messages = solve_one_question_multiturn(
            client, tools, tool_map, q, model=model, max_rounds=max_rounds
        )
        pred = normalize_answer(pred_boxed or "")
        is_correct = (pred == gt) if (pred and gt) else False
        correct += int(is_correct)

        print(f"[{idx}/{total}] 回合={rounds} 正确={is_correct} | 预测={pred} | 标准={gt}")

    acc = correct / total if total else 0.0
    print(f"AIME25 准确率: {correct}/{total} = {acc:.2%}")


def main():
    client = OpenAI(
        api_key="{}".format(os.getenv("API_KEY", "0")),
        base_url="http://localhost:{}/v1".format(os.getenv("API_PORT", 8000)),
    )
    tools = [
        {
            "type": "function",
            "function": {
                "name": "code_interpreter",
                "description": "A tool for executing code.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "code": {"type": "string", "description": "The code to execute"},
                    },
                    "required": ["code"],
                },
            },
        }
    ]
    tool_map = {"code_interpreter": execute_python}

    messages = []
    messages.append({"role": "system", "content": SYSTEM_PROMPT})
    messages.append({"role": "user", "content": "Please execute the code: print('Hello, world!')"})
    result = client.chat.completions.create(messages=messages, model="test", tools=tools)
    if result.choices[0].message.tool_calls is None:
        raise ValueError("Cannot retrieve function call from the response.")

    messages.append(result.choices[0].message)
    tool_call = result.choices[0].message.tool_calls[0].function
    print(tool_call)
    # Function(arguments='{\"code\": \"print(\\'Hello, world!\\')\"}', name='code_interpreter')
    name, arguments = tool_call.name, json.loads(tool_call.arguments)
    tool_result = tool_map[name](**arguments)
    messages.append({"role": "tool", "content": tool_result})
    result = client.chat.completions.create(messages=messages, model="test", tools=tools)
    print(result.choices[0].message.content)
    # Hello, world!


if __name__ == "__main__":
    if os.getenv("RUN_AIME25", "0") == "1":
        run_aime25_test()
    else:
        main()

"""
export RUN_AIME25=1
# 可选：限制题目数量
export AIME25_LIMIT=1
# 可选：每题最大对话轮数
export AIME25_MAX_ROUNDS=8
# 可选：模型名
export MODEL_NAME=test
python3 scripts/api_example/test_qwen3_python.py
"""
