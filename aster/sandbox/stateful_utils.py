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
import json
import logging
import threading
import time
import traceback
from typing import Any, Optional
import requests

DEFAULT_TIMEOUT = 10  # Default compile and run timeout
MAX_RETRIES = 3
INITIAL_RETRY_DELAY = 1
API_TIMEOUT = 10

logger = logging.getLogger(__name__)

def call_sandbox_api(
    sandbox_fusion_url: str,
    session_id: str,
    code: str,
    compile_timeout: int,
    run_timeout: int,
    memory_limit_mb: int,
    language: str = "python",
) -> tuple[Optional[dict[str, Any]], Optional[str]]:
    """
    调用远端 stateful sandbox API 执行代码，带 504 重试。

    返回 (response_json, error_message)。成功时 error_message 为 None。
    """
    log_prefix = f"[Session ID: {session_id}] "
    assert language == "python"

    payload = json.dumps(
        {
            "compile_timeout": compile_timeout,
            "run_timeout": run_timeout,
            "code": code,
            "memory_limit_MB": memory_limit_mb,
            "language": language,
            "files": {},
            "fetch_files": [],
            "session_id": session_id or None,
        }
    )
    headers = {"Content-Type": "application/json", "Accept": "application/json"}
    request_timeout = compile_timeout + run_timeout + API_TIMEOUT

    last_error = None

    for attempt in range(MAX_RETRIES):
        try:
            logger.info(
                f"{log_prefix}Attempt {attempt + 1}/{MAX_RETRIES}: Calling sandbox API at {sandbox_fusion_url}"
            )
            response = requests.post(
                sandbox_fusion_url,
                headers=headers,
                data=payload,
                timeout=request_timeout,
                proxies={'http': None, 'https': None},  # 禁用系统代理
            )

            if response.status_code == 504:
                last_error = (
                    f"{log_prefix}API Request Error: Gateway Timeout (504) on attempt "
                    f"{attempt + 1}/{MAX_RETRIES}"
                )
                logger.warning(last_error)
                if attempt < MAX_RETRIES - 1:
                    delay = INITIAL_RETRY_DELAY * (attempt + 1)
                    logger.info(f"{log_prefix}Retrying after {delay} seconds...")
                    time.sleep(delay)
                continue

            response.raise_for_status()
            logger.info(f"{log_prefix}Sandbox API call successful on attempt {attempt + 1}")
            return response.json(), None

        except requests.exceptions.RequestException as e:
            last_error = f"{log_prefix}API Request Error: {e}"
            break
        except json.JSONDecodeError as e:
            last_error = f"{log_prefix}API Response JSON Decode Error: {e}"
            break
        except Exception as e:
            last_error = f"{log_prefix}Unexpected Error: {e}"
            break

    logger.error(f"{log_prefix}Sandbox API call failed. Last error: {last_error}")
    return None, last_error.replace(log_prefix, "API Call Failed: ") if last_error else "API Call Failed after retries"


def process_single_case(
    session_id: str,
    sandbox_fusion_url: str,
    code: str,
    timeout: int,
    memory_limit_mb: int,
    language: str,
    concurrent_semaphore: Optional[threading.Semaphore] = None,
) -> tuple[int, dict[str, Any]]:
    """执行一次有状态代码并分类结果，返回 (result_status, metadata)。"""
    api_response = None
    error_msg = None
    logger.info(f"Processing code for session {session_id}.")

    try:
        if concurrent_semaphore:
            logger.debug(f"Attempting to acquire semaphore for session {session_id}")
            with concurrent_semaphore:
                logger.debug(f"Semaphore acquired. Calling API for session {session_id}")
                api_response, error_msg = call_sandbox_api(
                    sandbox_fusion_url=sandbox_fusion_url,
                    session_id=session_id,
                    code=code,
                    compile_timeout=timeout,
                    run_timeout=timeout,
                    memory_limit_mb=memory_limit_mb,
                    language=language,
                )
            logger.debug(f"Semaphore released for session {session_id}")
        else:
            api_response, error_msg = call_sandbox_api(
                sandbox_fusion_url=sandbox_fusion_url,
                session_id=session_id,
                code=code,
                compile_timeout=timeout,
                run_timeout=timeout,
                memory_limit_mb=memory_limit_mb,
                language=language,
            )
    except Exception as e:
        error_msg = f"API Request Exception during stateful run for session {session_id}: {e}"
        logger.error(f"{error_msg}")
        traceback.print_exc()

    metadata = {
        "session_id": session_id,
        "api_request_error": error_msg,
        "api_response": None,
        "status": "unknown",
        "api_status": None,
        "message": None,
        "stdout": None,
        "stderr": None,
        "exit_code": None,
        "duration": None,
        "compile_duration": None,
        "compile_stderr": None,
        "compile_status": None,
        "run_status": None,
    }
    result_status = -1

    if error_msg:
        metadata["status"] = "api_error"
        result_status = -1
    elif api_response:
        metadata["api_response"] = api_response
        metadata["api_status"] = api_response.get("status")
        metadata["message"] = api_response.get("message")
        compile_result = api_response.get("compile_result")
        run_result = api_response.get("run_result")

        if compile_result:
            metadata["compile_status"] = compile_result.get("status")
            metadata["compile_duration"] = compile_result.get("execution_time")
            metadata["compile_stderr"] = compile_result.get("stderr")

        if run_result:
            metadata["run_status"] = run_result.get("status")
            metadata["stdout"] = run_result.get("stdout")
            metadata["stderr"] = run_result.get("stderr")
            metadata["exit_code"] = run_result.get("return_code")
            metadata["duration"] = run_result.get("execution_time")

        api_status = metadata["api_status"]

        if api_status == "SandboxError":
            metadata["status"] = "sandbox_error"
            result_status = -1
        elif api_status == "Failed":
            # Compile 失败或超时
            is_compile_error = compile_result and (
                metadata["compile_status"] in ["Error", "TimeLimitExceeded"]
                or (metadata["compile_status"] == "Finished" and compile_result.get("return_code") != 0)
            )
            if is_compile_error:
                if metadata["compile_status"] == "TimeLimitExceeded":
                    metadata["status"] = "compile_timeout"
                else:
                    metadata["status"] = "compile_error"
                result_status = -4
            elif run_result:
                is_runtime_error = (
                    metadata["run_status"] == "TimeLimitExceeded"
                    or metadata["run_status"] == "Error"
                    or (metadata["run_status"] == "Finished" and run_result.get("return_code") != 0)
                )
                if is_runtime_error:
                    if metadata["run_status"] == "TimeLimitExceeded":
                        metadata["status"] = "timeout"
                        result_status = -3
                    else:
                        metadata["status"] = "runtime_error"
                        result_status = -2
                else:
                    metadata["status"] = "unknown_failure"
                    result_status = -1
            else:
                metadata["status"] = "unknown_failure_state"
                result_status = -1
        elif api_status == "Success":
            if run_result and metadata["run_status"] == "Finished":
                metadata["status"] = "success"
                result_status = True
            else:
                metadata["status"] = "unexpected_success_state"
                result_status = -1
        else:
            logger.warning(f"Unknown API status received: {api_status}")
            metadata["status"] = f"unknown_api_status_{api_status}"
            result_status = -1
    else:
        metadata["status"] = "unknown_api_state"
        result_status = -1

    return result_status, metadata
