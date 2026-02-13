# Copyright 2024 Bytedance Ltd. and/or its affiliates
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

import os
import traceback
import uuid
from enum import Enum
from pathlib import Path
from typing import Dict, Any, Optional
from typing import List, Tuple

import dill
import structlog
from fastapi import APIRouter
from pydantic import BaseModel, Field

from sandbox.runners import (
    CODE_RUNNERS,
    CellRunResult,
    CodeRunArgs,
    CodeRunResult,
    CommandRunResult,
    CommandRunStatus,
    Language,
    RunJupyterRequest,
    run_jupyter,
)
from starlette.concurrency import run_in_threadpool

sandbox_router = APIRouter()
logger = structlog.stdlib.get_logger()


class RunCodeRequest(BaseModel):
    compile_timeout: float = Field(10, description='compile timeout for compiled languages')
    run_timeout: float = Field(30, description='code run timeout')
    memory_limit_MB: int = Field(-1, description='maximum memory allowed in megabytes')
    code: str = Field(..., examples=['print("hello")'], description='the code to run')
    stdin: Optional[str] = Field(None, examples=[''], description='optional string to pass into stdin')
    language: Language = Field(..., examples=['python'], description='the language or execution mode to run the code')
    files: Dict[str, Optional[str]] = Field({}, description='a dict from file path to base64 encoded file content')
    fetch_files: List[str] = Field([], description='a list of file paths to fetch after code execution')


class RunStatus(str, Enum):
    # all command finished successfully
    Success = 'Success'
    # one of the process has non-zero return code
    Failed = 'Failed'
    # error on sandbox side
    SandboxError = 'SandboxError'


class RunCodeResponse(BaseModel):
    status: RunStatus
    message: str
    compile_result: Optional[CommandRunResult] = None
    run_result: Optional[CommandRunResult] = None
    executor_pod_name: Optional[str] = None
    files: Dict[str, str] = {}


class RunJupyterResponse(BaseModel):
    status: RunStatus
    message: str
    driver: Optional[CommandRunResult] = None
    cells: List[CellRunResult] = []
    executor_pod_name: Optional[str] = None
    files: Dict[str, str] = {}


def parse_run_status(result: CodeRunResult) -> Tuple[RunStatus, str]:
    outcomes = []
    retcodes = []
    err_msgs = []
    if result.compile_result is not None:
        outcomes.append(result.compile_result.status)
        err_msgs.append(result.compile_result.stderr or '')
        if result.compile_result.return_code is not None:
            retcodes.append(result.compile_result.return_code)
    if result.run_result is not None:
        outcomes.append(result.run_result.status)
        err_msgs.append(result.run_result.stderr or '')
        if result.run_result.return_code is not None:
            retcodes.append(result.run_result.return_code)

    for o, m in zip(outcomes, err_msgs):
        if o == CommandRunStatus.Error:
            return RunStatus.SandboxError, m
    if any([o == CommandRunStatus.TimeLimitExceeded for o in outcomes]):
        return RunStatus.Failed, ''
    if any([r != 0 for r in retcodes]):
        return RunStatus.Failed, ''
    # no error, no tle and no non-zero return codes -> success
    return RunStatus.Success, ''


@sandbox_router.post("/run_code", response_model=RunCodeResponse, tags=['sandbox'])
async def run_code(request: RunCodeRequest):
    resp = RunCodeResponse(status=RunStatus.Success, message='', executor_pod_name=os.environ.get('MY_POD_NAME'))
    try:
        logger.debug(
            f'start processing {request.language} request with code ```\n{request.code[:100]}\n``` and files {list(request.files.keys())}...(memory_limit: {request.memory_limit_MB}MB)'
        )
        result = await CODE_RUNNERS[request.language](CodeRunArgs(**request.model_dump()))

        resp.compile_result = result.compile_result
        resp.run_result = result.run_result
        resp.files = result.files
        resp.status, message = parse_run_status(result)
        if resp.status == RunStatus.SandboxError:
            resp.message = message
    except Exception as e:
        message = f'exception on running code {request.code}: {e} {traceback.print_tb(e.__traceback__)}'
        logger.warning(message)
        resp.message = message
        resp.status = RunStatus.SandboxError

    return resp


@sandbox_router.post("/run_jupyter", name='Run Code in Jupyter', response_model=RunJupyterResponse, tags=['sandbox'])
async def run_jupyter_handler(request: RunJupyterRequest):
    resp = RunJupyterResponse(status=RunStatus.Success, message='', executor_pod_name=os.environ.get('MY_POD_NAME'))
    code_repr = "\n".join(request.cells)[:100]
    try:
        logger.debug(
            f'start processing jupyter request with code ```\n{code_repr}\n``` and files {list(request.files.keys())}...'
        )
        result = await run_jupyter(request)
        resp.driver = result.driver
        if result.status != CommandRunStatus.Finished:
            resp.status = RunStatus.Failed
        else:
            resp.status = RunStatus.Success
            resp.cells = result.cells
            resp.files = result.files
    except Exception as e:
        message = f'exception on running jupyter {code_repr}: {e} {traceback.print_tb(e.__traceback__)}'
        logger.warning(message)
        resp.message = message
        resp.status = RunStatus.SandboxError

    return resp


# =============== Stateful START ===============
class RunStatefulCodeRequest(RunCodeRequest):
    session_id: str | None = Field(
        None,
        description="Create a new session when empty, "
                    "otherwise reuse the existing session."
    )

class RunStatefulCodeResponse(RunCodeResponse):
    session_id: str


class _StatefulRunner:
    def __init__(self) -> None:
        self._sessions: Dict[str, Dict[str, Any]] = {}
        self._sessions_set = set()
        self._state_dir = Path(os.environ.get("STATEFUL_SESSIONS_DIR", "/tmp/stateful_sessions"))
        self._state_dir.mkdir(parents=True, exist_ok=True, mode=0o700)

    def _state_path(self, sid: str) -> Path:
        return self._state_dir / f"{sid}.pkl"

    def _new_session(self) -> str:
        sid = str(uuid.uuid4())
        return sid

    def _load_snapshot(self, sid: str) -> Dict[str, Any]:
        pkl = self._state_path(sid)
        if pkl.exists():
            try:
                return dill.loads(pkl.read_bytes())
            except Exception:
                logger.warning(f"Failed to load pickle file for session {sid}")
                pass
        return {}

    async def _load_snapshot_async(self, sid: str) -> Dict[str, Any]:
        # Run sync function in thread pool
        return await run_in_threadpool(self._load_snapshot, sid)
    
    def _generate_restore_code(self, snapshot: Dict[str, Any]) -> str:
        """Dump the variable into executable code."""
        if not snapshot:
            return "# first run, nothing to restore"
        lines = []
        for k, v in snapshot.items():
            if k in {"__builtins__", "__loader__", "__spec__"}:
                continue
            lines.append(f"{k} = dill.loads({dill.dumps(v)!r})")
        return "\n".join(lines)

    def _generate_persist_code(self, state_path: str) -> str:
        """
        Generating Persistence Code with atomic write.
        """
        # Define a temporary file path next to the original state file
        temp_state_path = f"{state_path}.tmp"
        
        return f"""
import dill
import os

# A dictionary to hold the global state
snapshot = {{k: v for k, v in globals().items()
            if not k.startswith('__') or k == '__builtins__'}}

try:
    # 1. Write to a temporary file first
    with open({repr(temp_state_path)}, 'wb') as file:
        dill.dump(snapshot, file)
    
    # 2. If writing is successful, atomically rename it to the final destination
    os.rename({repr(temp_state_path)}, {repr(state_path)})

except Exception as e:
    # If anything goes wrong, log it (optional) and clean up the temp file if it exists
    print(f"Failed to persist state: {{e}}")
    if os.path.exists({repr(temp_state_path)}):
        os.remove({repr(temp_state_path)})
"""


_stateful_runner = _StatefulRunner()


@sandbox_router.post("/run_stateful_code", response_model=RunStatefulCodeResponse, tags=['sandbox'])
async def run_stateful_code(req: RunStatefulCodeRequest):
    # Create or reuse session
    session_id = req.session_id or _stateful_runner._new_session()
    if session_id not in _stateful_runner._sessions_set:
        logger.debug(f"First occurrence of session: {session_id}")
        _stateful_runner._sessions_set.add(session_id)

    # Local Persistent Path
    state_path = str(_stateful_runner._state_path(session_id))

    try:
        snapshot = await _stateful_runner._load_snapshot_async(session_id)
        restore_code = _stateful_runner._generate_restore_code(snapshot)
    except Exception as e:
        logger.warning(f"Snapshot loading failed for session {session_id}", sid=session_id, exc=e)
        return RunStatefulCodeResponse(
            session_id=session_id,
            status=RunStatus.SandboxError,
            message=f"State restore failed: {e}",
            executor_pod_name=os.environ.get("MY_POD_NAME")
    )

    # Generate persistent code
    persist_code = _stateful_runner._generate_persist_code(state_path)

    full_code = f"""
import dill, os
import sympy as sp
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"]        = "1"
os.environ["OMP_NUM_THREADS"]        = "1"
# os.environ["NUMEXPR_NUM_THREADS"]    = "1"
import math
import sympy
import itertools
{restore_code}
{req.code}
{persist_code}
"""

    # Execute the code
    args = CodeRunArgs(
        code=full_code,
        compile_timeout=req.compile_timeout,
        run_timeout=req.run_timeout,
        memory_limit_MB=req.memory_limit_MB,
        stdin=req.stdin,
        files=req.files,
        fetch_files=[],
        session_id=session_id
    )
    result = await CODE_RUNNERS[req.language](args)

    status, message = parse_run_status(result)
    if status == RunStatus.Success and os.path.exists(state_path):
        logger.debug(f"State persistence successful for session {session_id}")
    elif status != RunStatus.Success:
        logger.warning(f"Code execution failed for session {session_id}: {status} - {message}")
    elif not os.path.exists(state_path):
        logger.warning(f"State persistence failed for session {session_id}: state file not found")

    return RunStatefulCodeResponse(
        session_id=session_id,
        status=status,
        message=message,
        compile_result=result.compile_result,
        run_result=result.run_result,
        files=result.files,
        executor_pod_name=os.environ.get("MY_POD_NAME")
    )
# =============== Stateful END ===============