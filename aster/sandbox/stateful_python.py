# Copyright 2025 Bytedance Ltd. and/or its affiliates
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

import logging
import os
import threading
from contextlib import ExitStack
from enum import Enum
from typing import Any, Callable, Optional, TypeVar
from uuid import uuid4
import re
import ast
import re

import ray

from verl.tools.base_tool import BaseTool
from aster.sandbox.stateful_utils import process_single_case
from verl.utils.rollout_trace import rollout_trace_op

from verl.tools.schemas import OpenAIFunctionToolSchema

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

T = TypeVar("T")


class PoolMode(Enum):
    ThreadMode = 1
    ProcessMode = 2


@ray.remote(concurrency_groups={"acquire": 1, "release": 10})
class TokenBucketWorker:
    def __init__(self, rate_limit: int):
        self.rate_limit = rate_limit
        # this only used for observalability
        self.current_count = 0
        self._semaphore = threading.Semaphore(rate_limit)

    @ray.method(concurrency_group="acquire")
    def acquire(self):
        self._semaphore.acquire()
        self.current_count += 1

    @ray.method(concurrency_group="release")
    def release(self):
        self._semaphore.release()
        self.current_count -= 1

    def get_current_count(self):
        return self.current_count


class ExecutionWorker:
    def __init__(self, enable_global_rate_limit=True, rate_limit=10):
        self.rate_limit_worker = self._init_rate_limit(rate_limit) if enable_global_rate_limit else None

    def _init_rate_limit(self, rate_limit):
        # TODO validation for rate_limit
        # A Singleton Rate Limitor
        return TokenBucketWorker.options(name="rate-limiter", get_if_exists=True).remote(rate_limit)

    def ping(self):
        return True

    def execute(self, fn: Callable[..., T], *fn_args, **fn_kwargs) -> T:
        with ExitStack() as stack:
            stack.callback(self.rate_limit_worker.release.remote)
            ray.get(self.rate_limit_worker.acquire.remote())
            try:
                return fn(*fn_args, **fn_kwargs)
            except Exception as e:
                # TODO we should make this available to the tool caller
                logger.warning(f"Error when executing code: {e}")


def init_execution_pool(
    num_workers: int, enable_global_rate_limit=True, rate_limit=10, mode: PoolMode = PoolMode.ThreadMode
):
    if mode == PoolMode.ThreadMode:
        return (
            ray.remote(ExecutionWorker)
            .options(max_concurrency=num_workers)
            .remote(enable_global_rate_limit=enable_global_rate_limit, rate_limit=rate_limit)
        )
    else:
        raise NotImplementedError("Process mode is not implemented yet")
        # return ray.util.multiprocessing.Pool(processes=num_workers)


def patch_code(code: str) -> str:
    """
    Add print patch to the code
    """
    if not code.strip():
        return code
    try:
        # Use AST parsing for more accurate processing, only append print when the last statement is an expression
        tree = ast.parse(code)
        if not tree.body:
            return code

        last_stmt = tree.body[-1]
        if isinstance(last_stmt, ast.Expr):
            last_expr = last_stmt.value
            # If the last expression itself is a print call, do not modify
            is_print_call = (
                isinstance(last_expr, ast.Call)
                and isinstance(getattr(last_expr, 'func', None), ast.Name)
                and getattr(last_expr.func, 'id', None) == 'print'
            )
            if is_print_call:
                return code

            # Try to get expression source code
            expr_src = None
            try:
                expr_src = ast.get_source_segment(code, last_expr)
            except Exception:
                try:
                    # Fallback to use unparse (Py3.9+)
                    expr_src = ast.unparse(last_expr)
                except Exception:
                    expr_src = None

            if expr_src:
                # Single line pure expression: directly wrap the whole thing with print()
                if len(tree.body) == 1 and '\n' not in code:
                    return f"print({code})"
                # Multi-line: append print at the end, do not destroy original structure
                return f"{code.rstrip()}\nprint({expr_src})"

        # Parsing successful but last statement is not expression: keep as is
        return code
    except SyntaxError:
        # If AST parsing fails, fallback to heuristic method
        pass

    return _fallback_patch_code(code)

def _fallback_patch_code(code: str) -> str:
    """Fallback code patching method"""
    lines = code.split('\n')
    if not lines:
        return code

    # Find the last non-empty and non-comment line
    idx = len(lines) - 1
    while idx >= 0 and (not lines[idx].strip() or lines[idx].lstrip().startswith('#')):
        idx -= 1
    if idx < 0:
        return code

    last_line = lines[idx].strip()

    # If the last line itself is a print call, do not process
    if re.match(r'^\s*print\s*\(', last_line):
        return code

    # Do not append print to keyword-like statements
    if last_line in {"pass", "continue", "break"}:
        return code

    # More comprehensive exclusion conditions
    assignment_re = re.compile(r'(?<![:<>=!])=(?!=)')  # Single assignment operator, exclude ==, <=, >=, !=, :=
    exclusion_checks = [
        lambda line: line.startswith('#'),
        lambda line: any(line.startswith(k) for k in ['def ', 'class ', 'async def ']),
        lambda line: any(line.startswith(k) for k in ['if ', 'for ', 'while ', 'try:', 'except', 'finally:', 'with ', 'elif ', 'else:']),
        lambda line: any(line.startswith(k) for k in ['import ', 'from ']),
        lambda line: line.startswith('return '),
        lambda line: bool(assignment_re.search(line.split('#')[0])),
        lambda line: line.startswith('@'),
        lambda line: not line or line.endswith(':'),
    ]

    should_exclude = any(check(last_line) for check in exclusion_checks)

    if not should_exclude and last_line:
        if last_line.endswith('\\'):
            return code
        # Append print at the end of original text, avoid destroying structure
        return f"{code.rstrip()}\nprint({last_line})"

    return code


class SandboxFusionTool(BaseTool):
    """A tool for executing the code using sanbox fusion image.

    - `to_openai_function_tool_schema`: return the tool schema in OpenAI format.
    - `create`: create a tool instance for a trajectory.
    - `execute`: execute the tool.
    - `calc_reward`: calculate the reward respect to tool state.
    - `release`: release the tool instance.
    """

    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        """
        _tool_schema = OpenAIFunctionToolSchema.model_validate({
            "type": "function",
            "function": {
                "name": "code_interpreter",
                "description": "A tool for execute code",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "code": {
                            "type": "string",
                            "description": "code needs to be execute and grad",
                        },
                    },
                    "required": ["code"],
                },
            }
        })
        """
        super().__init__(config, tool_schema)
        self._instance_dict = {}
        self.num_workers = config.get("num_workers", 10)
        self.rate_limit = config.get("rate_limit", 10)
        self.default_timeout = config.get("default_timeout", 30)
        self.default_language = config.get("default_language", "python")
        self.enable_global_rate_limit = config.get("enable_global_rate_limit", True)
        self.execution_pool = init_execution_pool(
            num_workers=self.num_workers,
            enable_global_rate_limit=self.enable_global_rate_limit,
            rate_limit=self.rate_limit,
            mode=PoolMode.ThreadMode,
        )
        self.sandbox_fusion_url = config.get("sandbox_fusion_url", "")
        self.memory_limit_mb = config.get("memory_limit_mb", 1024)
        if self.sandbox_fusion_url == "":
            raise ValueError("sandbox_fusion_url is not set")
        log_msg = f"Init SandboxFusionTool with config: {config}"
        logger.info(log_msg)

        # TODO: Is this correct?
        self.code_pattern = re.compile(
            r"```(?:py|python)?\s*([\s\S]*?)```", re.MULTILINE
        )


    def get_openai_tool_schema(self) -> OpenAIFunctionToolSchema:
        return self.tool_schema

    async def create(self, instance_id: Optional[str] = None, ground_truth: Optional[str] = None, **kwargs) -> str:
        if instance_id is None:
            instance_id = str(uuid4())
        self._instance_dict[instance_id] = {
            "response": "",
            "ground_truth": ground_truth,
            "reward": [],
        }
        return instance_id

    @rollout_trace_op
    async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs) -> tuple[str, float, dict]:
        session_id = parameters.get("session_id", "")
        assert session_id, "session_id is required for stateful python"
        code = parameters.get("code", "")
        timeout = parameters.get("timeout", self.default_timeout)
        language = parameters.get("language", self.default_language)
        if not isinstance(code, str):
            code = str(code)

        code_lines = code.count("\n") + 1 if code else 0
        code_to_execute = patch_code(code)
        result, result_status = await self.execution_pool.execute.remote(self.execute_code, instance_id, session_id, code_to_execute, timeout, language)

        metrics = {
            "code_lines": code_lines,
        }
        
        return result, result_status, metrics

    def execute_code(self, instance_id, session_id, code, timeout=30, language="python"):
        result_status, metadata = process_single_case(
            session_id=session_id,
            sandbox_fusion_url=self.sandbox_fusion_url,
            code=code,
            timeout=timeout,
            memory_limit_mb=self.memory_limit_mb,
            language=language,
        )
        
        # Return meaningful feedback based on different execution statuses
        if metadata["run_status"] == "Finished":
            # Check if execution was truly successful
            if result_status == True:
                # Successful execution
                actual_output = metadata["stdout"] + metadata["stderr"]
                logger.debug(f"actual_output from sandbox fusion: {actual_output},{instance_id}")
                return actual_output, result_status
            elif result_status == -2:
                # Runtime error (Finished but return_code is not 0)
                error_msg = "Runtime error occurred during execution."
                if metadata.get("stderr"):
                    error_msg += f"\n\nError details:\n{metadata['stderr']}"
                if metadata.get("stdout"):
                    error_msg += f"\n\nOutput before error:\n{metadata['stdout']}"
                return error_msg, result_status
            else:
                actual_output = metadata["stdout"] + metadata["stderr"]
                return actual_output, result_status
        elif metadata["run_status"] == "TimeLimitExceeded":
            # Timeout error
            error_msg = f"Execution timeout after {timeout} seconds. Consider optimizing your code for better performance or reducing computational complexity."
            if metadata.get("stderr"):
                error_msg += f"\n\nError output:\n{metadata['stderr']}"
            return error_msg, result_status
        elif metadata["run_status"] == "Error":
            # Runtime error
            error_msg = "Runtime error occurred during execution."
            if metadata.get("stderr"):
                error_msg += f"\n\nError details:\n{metadata['stderr']}"
            if metadata.get("stdout"):
                error_msg += f"\n\nOutput before error:\n{metadata['stdout']}"
            return error_msg, result_status
        elif metadata.get("api_request_error"):
            # API request error
            error_msg = f"Sandbox API error: {metadata['api_request_error']}"
            return error_msg, result_status
        else:
            # Other unknown error
            status = metadata.get("status", "unknown")
            error_msg = f"Unknown execution error (status: {status})"
            if metadata.get("stderr"):
                error_msg += f"\n\nError output:\n{metadata['stderr']}"
            return error_msg, result_status

    def extract_code_blocks(self, text: str) -> list[str]:
        return [match.strip() for match in self.code_pattern.findall(text)]

    async def calc_reward(self, instance_id: str, **kwargs) -> str:
        return self._instance_dict[instance_id]["reward"]

    async def release(self, instance_id: str, **kwargs) -> None:
        del self._instance_dict[instance_id]


if __name__ == '__main__':
    code_test1 ="""
for i in range(10):
    pass
"""
    # before patch
    print(code_test1)
    print("*"*20)
    # after patch
    print(patch_code(code_test1))

    code_test2 ="""
def func():
    pass
"""
    # before patch
    print(code_test2)
    print("*"*20)
    # after patch
    print(patch_code(code_test2))

    # Single-line pure expression
    code_expr1 = """
1+2
"""
    print(code_expr1)
    print("*"*20)
    print(patch_code(code_expr1))

    # Multi-line, last line is an expression
    code_expr2 = """
x = 1
y = 2
x + y
"""
    print(code_expr2)
    print("*"*20)
    print(patch_code(code_expr2))

    # Already has print
    code_print = """
print('hello')
"""
    print(code_print)
    print("*"*20)
    print(patch_code(code_print))

    # Assignment only at the end
    code_assign = """
a = 123
"""
    print(code_assign)
    print("*"*20)
    print(patch_code(code_assign))

    # Control flow at the end
    code_control = """
if True:
    x = 1
else:
    x = 2
"""
    print(code_control)
    print("*"*20)
    print(patch_code(code_control))

    # for + continue at the end
    code_for_continue = """
for i in range(3):
    if i == 1:
        continue
"""
    print(code_for_continue)
    print("*"*20)
    print(patch_code(code_for_continue))

    # try/except at the end
    code_try_except = """
try:
    1/0
except ZeroDivisionError:
    pass
"""
    print(code_try_except)
    print("*"*20)
    print(patch_code(code_try_except))

    # Multi-line expression with parentheses line breaks
    code_paren_multiline = """
(
1 +
2
)
"""
    print(code_paren_multiline)
    print("*"*20)
    print(patch_code(code_paren_multiline))

    # Expression with trailing comment (ensuring comment is not broken)
    code_trailing_comment = """
1 + 2  # sum
"""
    print(code_trailing_comment)
    print("*"*20)
    print(patch_code(code_trailing_comment))

    code_backslash = """
s = "hello " \
    "world"
s
"""
    print(code_backslash)
    print("*"*20)
    print(patch_code(code_backslash))