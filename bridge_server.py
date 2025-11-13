# /app/bridge_server.py
import os
import re
import json
import asyncio
import subprocess
import shutil
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Body, Query
from fastapi.middleware.cors import CORSMiddleware

# ============================
# Local imports
# ============================
from managers.db_manager import insert_repo_to_db, get_connection
from managers.prompt_agent import LLMAgent
from managers.llm_manager import LLMManager
from managers.context_manager import ContextManager
from utils.torch_version_loader import TorchVersionLoader


# ============================
# Base setup
# ============================
BASE_DIR = Path(__file__).parent.resolve()
GIT_CLONE_DIR = (BASE_DIR / "workspace").resolve()
GIT_CLONE_DIR.mkdir(parents=True, exist_ok=True)
BRIDGE_PORT = 9013

app = FastAPI(title="Bridge Server (React ↔ FastAPI)")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================
# Global managers & locks
# ============================
clients: List[WebSocket] = []
clients_lock = asyncio.Lock()
llm_lock = asyncio.Lock()
llm_manager = LLMManager()
agent = LLMAgent(llm_manager)
torch_loader = TorchVersionLoader(base_dir="/app/pytorch_versions")
context_manager = ContextManager()
RG_BINARY = shutil.which("rg")
PRIMARY_TASK = "assistant"


# ============================
# Utility Functions
# ============================
def _normalize_trigger_text(text: str | None) -> str:
    if not text:
        return ""
    cleaned = text.replace("“", '"').replace("”", '"').replace("’", "'")
    cleaned = re.sub(r"\s+", " ", cleaned).strip().lower()
    return cleaned


CLOSING_BEHAVIOR_TRIGGER = (
    llm_manager.prompts.get(PRIMARY_TASK, {}).get("closing_behavior_trigger")
    or "Would you like me to refactor this for the latest PyTorch version, or a specific version you prefer?"
)
CLOSING_BEHAVIOR_TRIGGER_NORMALIZED = _normalize_trigger_text(CLOSING_BEHAVIOR_TRIGGER)
LAST_USER_REQUESTS: Dict[Any, Dict[str, Any]] = {}
FUNCTION_DEFINITIONS: List[Dict[str, Any]] = llm_manager.prompts.get("functions", []) or []
HAS_SELF_CHECKER = "self_checker" in llm_manager.prompts
HAS_FUNCTION_ROUTER = "function_router" in llm_manager.prompts
WORKSPACE_ROOT = Path("/app").resolve()
FILE_TOKEN_PATTERN = re.compile(r"([A-Za-z0-9_\-./]+?\.(?:ya?ml|json|py|ts|js|md|txt|c|cpp|java|rs|go|sh))")


def _find_file_by_name(token: str) -> Path | None:
    try:
        iterator = WORKSPACE_ROOT.rglob(token)
    except Exception:
        return None
    for match in iterator:
        if match.is_file():
            return match
    return None


def _tab_storage_key(tab_id: int | None):
    return tab_id if tab_id is not None else "__default__"


def _closing_trigger_matches(text: str | None) -> bool:
    if not text or not CLOSING_BEHAVIOR_TRIGGER_NORMALIZED:
        return False
    return _normalize_trigger_text(text) == CLOSING_BEHAVIOR_TRIGGER_NORMALIZED


def _remember_user_request(tab_id: int | None, user_text: str, response_text: str | None):
    LAST_USER_REQUESTS[_tab_storage_key(tab_id)] = {
        "user_text": user_text,
        "response_text": response_text,
        "timestamp": datetime.utcnow().isoformat(),
    }


def _get_last_user_request(tab_id: int | None):
    return LAST_USER_REQUESTS.get(_tab_storage_key(tab_id))


def _extract_symbol_candidates(text: str, max_symbols: int = 5) -> List[str]:
    if not text:
        return []
    candidates: List[str] = []
    torch_refs = re.findall(r"(torch(?:\.[A-Za-z_][\w]*)+)", text)
    for ref in torch_refs:
        last = ref.split(".")[-1]
        if last:
            candidates.append(last)
    candidates.extend(re.findall(r"\bdef\s+([A-Za-z_][\w]*)", text))
    candidates.extend(re.findall(r"\bclass\s+([A-Za-z_][\w]*)", text))
    ordered: List[str] = []
    seen = set()
    for name in candidates:
        clean = name.strip()
        if not clean or clean in seen:
            continue
        seen.add(clean)
        ordered.append(clean)
        if len(ordered) >= max_symbols:
            break
    return ordered


def _search_symbol_definitions(symbol: str, torch_root: Path, max_matches: int = 1) -> List[tuple[Path, int]]:
    if not symbol or not torch_root.exists() or not RG_BINARY:
        return []
    pattern = rf"^\s*(?:class|def)\s+{re.escape(symbol)}\b"
    cmd = [
        RG_BINARY,
        "--line-number",
        "--no-heading",
        "--max-count",
        str(max_matches),
        "--color",
        "never",
        pattern,
        str(torch_root),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode not in (0, 1):
        print(f"[TorchLookup] ⚠️ rg error for symbol {symbol}: {proc.stderr.strip()}")
        return []
    matches: List[tuple[Path, int]] = []
    for line in proc.stdout.strip().splitlines():
        if not line:
            continue
        parts = line.split(":", 2)
        if len(parts) < 2:
            continue
        path_str, line_no = parts[0], parts[1]
        try:
            line_idx = int(line_no)
        except ValueError:
            continue
        matches.append((Path(path_str).resolve(), line_idx))
    return matches


def _read_file_snippet(file_path: Path, center_line: int, before: int = 20, after: int = 40) -> str | None:
    try:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            lines = f.readlines()
    except Exception as exc:
        print(f"[TorchLookup] ⚠️ snippet read failed for {file_path}: {exc}")
        return None
    if not lines:
        return None
    start = max(center_line - 1 - before, 0)
    end = min(center_line - 1 + after, len(lines))
    snippet = "".join(lines[start:end]).strip()
    return snippet or None


def _collect_symbol_snippets(user_text: str, repo_root: Path, max_total: int = 3):
    torch_root = repo_root / "torch"
    symbols = _extract_symbol_candidates(user_text)
    if not torch_root.exists() or not symbols:
        return [], []
    snippets: List[str] = []
    used_symbols: List[str] = []
    for symbol in symbols:
        matches = _search_symbol_definitions(symbol, torch_root, max_matches=1)
        for match_path, line_no in matches:
            try:
                rel_path = match_path.relative_to(repo_root)
            except ValueError:
                rel_path = match_path
            snippet = _read_file_snippet(match_path, line_no)
            if not snippet:
                continue
            block = f"### {rel_path}:{line_no}\n{snippet}"
            snippets.append(block)
            used_symbols.append(symbol)
            if len(snippets) >= max_total:
                return snippets, used_symbols
        if len(snippets) >= max_total:
            break
    return snippets, used_symbols


def _build_torch_source_prompt(user_text: str) -> str | None:
    context = torch_loader.build_context_from_text(user_text)
    if not context:
        return None
    repo_root = context.root_path / "repo"
    if not repo_root.exists():
        return None
    snippets, symbols = _collect_symbol_snippets(user_text, repo_root)
    if not snippets:
        return None
    header = [
        "[Torch Source Lookup]",
        f"Version: {context.version}",
    ]
    if symbols:
        header.append(f"Symbols: {', '.join(symbols)}")
    return "\n".join(header) + "\n\n" + "\n\n".join(snippets)


async def handle_closing_behavior_request(tab_id: int | None):
    last_request = _get_last_user_request(tab_id)
    if not last_request:
        await broadcast(
            {
                "type": "intent_notice",
                "text": "마지막 사용자 요청을 찾을 수 없습니다. 먼저 작업 요청을 전달해 주세요.",
                "tabId": tab_id,
                "timestamp": current_timestamp(),
            }
        )
        return

    try:
        torch_prompt = await asyncio.to_thread(_build_torch_source_prompt, last_request["user_text"])
    except Exception as exc:
        await broadcast(
            {
                "type": "error",
                "text": f"PyTorch 소스 탐색 중 오류가 발생했습니다: {exc}",
                "tabId": tab_id,
                "timestamp": current_timestamp(),
            }
        )
        print(f"[TorchLookup] ⚠️ build prompt failed: {exc}")
        return

    if not torch_prompt:
        await broadcast(
            {
                "type": "intent_notice",
                "text": "요청과 일치하는 PyTorch 소스 코드를 찾지 못했습니다. 함수명이나 torch.* 경로를 포함해 주세요.",
                "tabId": tab_id,
                "timestamp": current_timestamp(),
            }
        )
        return

    combined_prompt = f"{torch_prompt}\n\n[User Request]\n{last_request['user_text']}"
    response_text = await run_llm_call(combined_prompt, task=PRIMARY_TASK)
    clean_response = _strip_reasoning_output(response_text)
    context_manager.add_message(tab_id, "assistant", clean_response)
    await broadcast(
        {
            "type": "llm_response",
            "text": clean_response,
            "data": {"task": PRIMARY_TASK, "context": "torch_lookup"},
            "tabId": tab_id,
            "timestamp": current_timestamp(),
        }
    )
    _remember_user_request(tab_id, last_request["user_text"], clean_response)


async def broadcast(msg: Dict[str, Any]):
    print(f"[Bridge] 📨 {msg}")
    dead = []
    async with clients_lock:
        for ws in clients:
            try:
                await ws.send_json(json.loads(json.dumps(msg, default=str)))
            except Exception:
                dead.append(ws)
        for d in dead:
            clients.remove(d)


def current_timestamp() -> str:
    return datetime.utcnow().isoformat()


# ============================
# LLM Handling
# ============================
def _strip_reasoning_output(text: str) -> str:
    """Remove <think>...</think> reasoning traces."""
    if not text:
        return text
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


async def run_llm_call(
    prompt: str,
    *,
    task: str,
    max_new_tokens: int = 512,
    system_override: str | None = None,
) -> str:
    async with llm_lock:
        result = await asyncio.to_thread(
            llm_manager.generate,
            prompt,
            task=task,
            max_new_tokens=max_new_tokens,
            system_override=system_override,
        )
        return _strip_reasoning_output(result)


# ============================
# Self-check & Function Routing
# ============================
def _safe_json_object(text: str) -> Dict[str, Any] | None:
    if not text:
        return None
    try:
        start = text.index("{")
        end = text.rindex("}") + 1
        return json.loads(text[start:end])
    except (ValueError, json.JSONDecodeError):
        return None


def _parse_self_checker_output(raw: str) -> Dict[str, Any] | None:
    if not raw:
        return None
    confidence = None
    freshness = None
    for line in raw.splitlines():
        clean = line.strip()
        if clean.lower().startswith("confidence:"):
            try:
                confidence = float(clean.split(":", 1)[1].strip())
            except ValueError:
                confidence = None
        elif clean.lower().startswith("freshness_need:"):
            value = clean.split(":", 1)[1].strip().lower()
            freshness = "yes" if value == "yes" else "no"
    if confidence is None and freshness is None:
        return None
    return {"confidence": confidence, "freshness_need": freshness}


def _should_trigger_followup(check_result: Dict[str, Any]) -> bool:
    if not check_result:
        return False
    if check_result.get("freshness_need") == "yes":
        return True
    confidence = check_result.get("confidence")
    return confidence is not None and confidence < 0.5


async def run_self_checker(text: str) -> Dict[str, Any] | None:
    if not HAS_SELF_CHECKER or not text or not text.strip():
        return None
    raw = await run_llm_call(text, task="self_checker", max_new_tokens=64)
    return _parse_self_checker_output(raw)


def _build_function_catalog_text() -> str:
    if not FUNCTION_DEFINITIONS:
        return ""
    return json.dumps(FUNCTION_DEFINITIONS, ensure_ascii=False, indent=2)


async def select_function_call(user_text: str) -> Dict[str, Any] | None:
    if not FUNCTION_DEFINITIONS:
        return None
    if HAS_FUNCTION_ROUTER:
        catalog = _build_function_catalog_text()
        planner_prompt = (
            "다음은 사용할 수 있는 함수 목록과 스키마입니다.\n"
            f"{catalog}\n\n"
            "사용자 요청:\n"
            f"{user_text}\n\n"
            "위 요청을 해결하기 위해 호출할 최적의 함수를 JSON 한 줄로만 응답하세요. "
            '형식: {"name": "<함수명 또는 none>", "arguments": {...}}'
        )
        raw = await run_llm_call(planner_prompt, task="function_router", max_new_tokens=256)
        parsed = _safe_json_object(raw)
        if parsed and isinstance(parsed, dict) and parsed.get("name"):
            return parsed
    return None


def _fallback_function_from_text(user_text: str) -> Dict[str, Any] | None:
    if not user_text:
        return None
    matches = FILE_TOKEN_PATTERN.findall(user_text)
    for token in matches:
        cleaned = token.strip().lstrip("./")
        if not cleaned:
            continue
        final_path: Path | None = None
        try:
            candidate = _resolve_workspace_path(cleaned)
        except ValueError:
            continue
        if candidate.exists() and candidate.is_file():
            final_path = candidate
        else:
            alt = _find_file_by_name(Path(cleaned).name)
            if alt:
                final_path = alt
        if final_path:
            return {"name": "read_file", "arguments": {"path": str(final_path)}}
    if matches:
        return {"name": "search_file", "arguments": {"keyword": matches[0]}}
    return None


def _resolve_workspace_path(path_value: str) -> Path:
    if not path_value:
        raise ValueError("path is required")
    candidate = Path(path_value.strip())
    if not candidate.is_absolute():
        candidate = (WORKSPACE_ROOT / candidate).resolve()
    if not str(candidate).startswith(str(WORKSPACE_ROOT)):
        raise ValueError("Access outside workspace is not allowed")
    return candidate


def _execute_load_files(arguments: Dict[str, Any]) -> str:
    target = _resolve_workspace_path(arguments.get("path", ""))
    if not target.exists():
        raise FileNotFoundError(f"Path not found: {target}")
    if target.is_dir():
        entries = sorted(p.name for p in target.iterdir())
        snippet = "\n".join(entries[:100])
        if len(entries) > 100:
            snippet += "\n... (truncated)"
        return f"[Directory Listing] {target}\n{snippet}"
    data = target.read_text(encoding="utf-8", errors="ignore")
    preview = data[:4000]
    if len(data) > 4000:
        preview += "\n... (truncated)"
    return f"[File Content] {target}\n{preview}"


def _execute_connect_db(arguments: Dict[str, Any]) -> str:
    query = (arguments.get("query") or "").strip()
    if not query:
        raise ValueError("query is required")
    if not query.lower().startswith("select"):
        raise ValueError("Only SELECT queries are allowed for safety.")
    conn = get_connection()
    try:
        cur = conn.cursor()
        cur.execute(query)
        rows = cur.fetchall()
        columns = [desc[0] for desc in cur.description] if cur.description else []
        limited = rows[:50]
        result_rows = [
            {columns[idx]: value for idx, value in enumerate(row)}
            for row in limited
        ]
        return json.dumps({"rows": result_rows, "rowCount": len(rows)}, ensure_ascii=False, indent=2)
    finally:
        try:
            cur.close()
        except Exception:
            pass
        conn.close()


def _execute_search_web(arguments: Dict[str, Any]) -> str:
    query = arguments.get("query") or ""
    return (
        "[Search Placeholder] Web search is not available in this environment. "
        f"Intended query: {query}"
    )


def _execute_search_file(arguments: Dict[str, Any]) -> str:
    keyword = (arguments.get("keyword") or "").strip()
    if not keyword:
        raise ValueError("keyword is required")
    max_results = arguments.get("max_results")
    try:
        max_results = int(max_results) if max_results is not None else 50
    except (TypeError, ValueError):
        max_results = 50
    max_results = max(1, min(max_results, 200))

    matches: List[str] = []
    for root, _dirs, files in os.walk(WORKSPACE_ROOT):
        if len(matches) >= max_results:
            break
        for fname in files:
            if len(matches) >= max_results:
                break
            if keyword.lower() in fname.lower():
                full_path = Path(root) / fname
                try:
                    rel = full_path.relative_to(WORKSPACE_ROOT)
                except ValueError:
                    continue
                matches.append(str(rel))

    if not matches:
        return f"[Search File] No files found matching '{keyword}'."
    return json.dumps({"keyword": keyword, "results": matches}, ensure_ascii=False, indent=2)


def _execute_read_file(arguments: Dict[str, Any]) -> str:
    target = _resolve_workspace_path(arguments.get("path", ""))
    if not target.exists():
        raise FileNotFoundError(f"Path not found: {target}")
    if target.is_dir():
        raise IsADirectoryError(f"Path is a directory: {target}")
    data = target.read_text(encoding="utf-8", errors="ignore")
    preview = data[:8000]
    if len(data) > 8000:
        preview += "\n... (truncated)"
    return f"[Read File] {target}\n{preview}"


async def execute_function_call(name: str, arguments: Dict[str, Any]) -> str:
    name = (name or "").strip()
    if not name or name == "none":
        raise ValueError("No executable function selected.")
    if name == "self_check_answer":
        answer = arguments.get("answer")
        if not answer:
            raise ValueError("answer is required for self_check_answer")
        result = await run_llm_call(answer, task="self_checker", max_new_tokens=64)
        return result

    sync_map = {
        "load_files": _execute_load_files,
        "connect_db": _execute_connect_db,
        "search_web": _execute_search_web,
        "search_file": _execute_search_file,
        "read_file": _execute_read_file,
    }
    func = sync_map.get(name)
    if not func:
        raise ValueError(f"Unsupported function: {name}")
    return await asyncio.to_thread(func, arguments)


# ============================
# User Message Handling
# ============================
async def handle_user_message(user_text: str, tab_id: int | None):
    try:
        # 1️⃣ 사용자 입력 저장
        context_manager.add_message(tab_id, "user", user_text)

        # 2️⃣ closing behavior 트리거 체크
        if _closing_trigger_matches(user_text):
            await handle_closing_behavior_request(tab_id)
            return

        # 3️⃣ 도구 계획 (답변 금지 단계)
        planned_call = await select_function_call(user_text)
        if not planned_call or planned_call.get("name") in (None, "", "none"):
            planned_call = _fallback_function_from_text(user_text)

        # 4️⃣ self-check: 사용자 질문 자체 평가 -> 라우팅
        self_check = await run_self_checker(user_text)
        if self_check:
            await broadcast(
                {
                    "type": "self_check",
                    "text": (
                        f"Self-check → confidence={self_check.get('confidence')} "
                        f"freshness={self_check.get('freshness_need')}"
                    ),
                    "data": self_check,
                    "tabId": tab_id,
                    "timestamp": current_timestamp(),
                }
            )

        force_search = False
        if self_check:
            fresh = self_check.get("freshness_need")
            conf = self_check.get("confidence")
            if fresh == "yes" or (conf is not None and conf < 0.7):
                force_search = True

        if force_search and (not planned_call or planned_call.get("name") not in {"read_file", "search_file"}):
            planned_call = {"name": "search_web", "arguments": {"query": user_text}}

        tool_block = ""
        executed_function = None
        if planned_call and planned_call.get("name") not in (None, "", "none"):
            try:
                func_output = await execute_function_call(
                    planned_call.get("name", ""), planned_call.get("arguments") or {}
                )
                executed_function = {**planned_call}
                tool_block = f"[Tool Result: {planned_call.get('name')}]\n{func_output}"
                context_manager.add_message(tab_id, "tool", tool_block)
                await broadcast(
                    {
                        "type": "tool_result",
                        "text": func_output,
                        "data": {
                            "function": executed_function,
                        },
                        "tabId": tab_id,
                        "timestamp": current_timestamp(),
                    }
                )
            except Exception as exc:
                await broadcast(
                    {
                        "type": "error",
                        "text": f"Function call failed: {exc}",
                        "data": {"function": planned_call},
                        "tabId": tab_id,
                        "timestamp": current_timestamp(),
                    }
                )
                print(f"[Bridge] ⚠️ function call failed: {exc}")
                tool_block = ""
                executed_function = None

        # 5️⃣ 전체 문맥 + 도구 결과 기반 프롬프트 구성
        full_prompt = context_manager.build_prompt(tab_id, user_text)
        if tool_block:
            full_prompt = f"{full_prompt}\n\n{tool_block}"

        # 6️⃣ LLM 호출 (최종 답변)
        response_text = await run_llm_call(full_prompt, task=PRIMARY_TASK)
        clean_response = _strip_reasoning_output(response_text)
        context_manager.add_message(tab_id, "assistant", clean_response)

        # 7️⃣ 응답 브로드캐스트
        await broadcast(
            {
                "type": "llm_response",
                "text": clean_response,
                "data": {
                    "task": PRIMARY_TASK,
                    "function": executed_function,
                    "selfCheck": self_check,
                },
                "tabId": tab_id,
                "timestamp": current_timestamp(),
            }
        )

        # 8️⃣ closing behavior 대비 마지막 요청 저장
        _remember_user_request(tab_id, user_text, clean_response)

    except Exception as exc:
        await broadcast(
            {
                "type": "error",
                "text": f"LLM 처리 중 오류가 발생했습니다: {exc}",
                "tabId": tab_id,
                "timestamp": current_timestamp(),
            }
        )
        print(f"[Bridge] ⚠️ handle_user_message failed: {exc}")


# ============================
# Git & Repo Handling
# ============================
def extract_github_url(text: str) -> str | None:
    match = re.search(r"(https?://github\.com/[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+)", text)
    return match.group(1) if match else None


IGNORE_DIRS = {".git", "venv", "__pycache__", "node_modules"}


def build_dir_tree(base_path: Path, root_path: Path | None = None, max_depth: int = 5, depth: int = 0):
    if root_path is None:
        root_path = base_path
    tree = {"name": base_path.name, "path": str(base_path.relative_to(GIT_CLONE_DIR)), "type": "folder", "children": []}
    if depth > max_depth or not base_path.is_dir():
        return tree
    for entry in sorted(base_path.iterdir(), key=lambda e: (e.is_file(), e.name.lower())):
        if entry.name in IGNORE_DIRS:
            continue
        if entry.is_dir():
            tree["children"].append(build_dir_tree(entry, root_path, max_depth, depth + 1))
        else:
            tree["children"].append({"name": entry.name, "path": str(entry.relative_to(GIT_CLONE_DIR)), "type": "file"})
    return tree


async def clone_repo_and_broadcast(url: str):
    """GitHub 저장소를 클론하고 요약/청크/심볼링크 생성 작업을 수행"""
    repo_name = url.split("/")[-1].replace(".git", "")
    dest = GIT_CLONE_DIR / repo_name
    git_dir = dest / ".git"

    try:
        # ✅ 폴더 존재 + .git 폴더도 있으면 pull
        if dest.exists() and git_dir.exists():
            await asyncio.to_thread(subprocess.run, ["git", "-C", str(dest), "pull"], check=True)
        else:
            # ⚠️ 기존 폴더가 남아있고 .git이 없으면 제거 후 재clone
            if dest.exists():
                shutil.rmtree(dest)
            await asyncio.to_thread(subprocess.run, ["git", "clone", url, str(dest)], check=True)

        # ✅ DB 기록 및 분석 단계
        repo_id = await asyncio.to_thread(insert_repo_to_db, repo_name, url, dest)

        await broadcast({"type": "git_status", "text": "Summarizing files..."})
        await asyncio.to_thread(agent.summarize_repo_files, repo_id, dest)

        await broadcast({"type": "git_status", "text": "Generating chunks..."})
        await asyncio.to_thread(agent.chunk_repo_files, repo_id, dest)

        await broadcast({"type": "git_status", "text": "Extracting symbol links..."})
        await asyncio.to_thread(agent.extract_symbol_links, repo_id, dest)

        await broadcast({"type": "git_status", "text": "✅ Done."})

    except subprocess.CalledProcessError as e:
        # pull/clone 명령이 실패할 경우 재시도
        await broadcast({"type": "git_status", "text": f"⚠️ Git command failed: {e}. Retrying..."})
        if dest.exists():
            shutil.rmtree(dest)
        await asyncio.to_thread(subprocess.run, ["git", "clone", url, str(dest)], check=True)
        await broadcast({"type": "git_status", "text": "✅ Repository re-cloned successfully."})

    except Exception as e:
        await broadcast({"type": "error", "text": f"❌ Repository clone failed: {e}"})
        print(f"[Bridge] ⚠️ clone_repo_and_broadcast failed: {e}")


# ============================
# FastAPI Routes
# ============================
@app.on_event("startup")
async def startup_event():
    print("========== DEBUG PATH CHECK ==========")
    print(f"[DEBUG] BASE_DIR: {BASE_DIR}")
    print(f"[DEBUG] GIT_CLONE_DIR: {GIT_CLONE_DIR}")
    print(f"[DEBUG] Exists(GIT_CLONE_DIR): {GIT_CLONE_DIR.exists()}")
    print("======================================")


@app.post("/reset_db")
async def reset_db():
    try:
        conn = get_connection()
        cur = conn.cursor()
        cur.execute("""
            TRUNCATE TABLE repo_meta, files_meta, repo_chunks, symbol_links
            RESTART IDENTITY CASCADE;
        """)
        conn.commit()
        cur.close()
        conn.close()
        return {"status": "ok", "message": "All tables truncated"}
    except Exception as e:
        return {"status": "error", "message": str(e)}


@app.post("/send")
async def from_react(payload: Dict[str, Any] = Body(...)):
    text = payload.get("text", "")
    msg_type = payload.get("type", "")
    tab_id = payload.get("tabId")
    github_url = extract_github_url(text)

    if github_url:
        asyncio.create_task(clone_repo_and_broadcast(github_url))
        return {"status": "ok", "message": "Repository cloning and analysis started."}

    if msg_type == "user_input" and text.strip():
        asyncio.create_task(handle_user_message(text.strip(), tab_id))
        return {"status": "ok", "message": "User message queued for processing."}

    return {"status": "ok", "message": "No actionable content found."}


@app.websocket("/ws/client")
async def ws_client(ws: WebSocket):
    await ws.accept()
    async with clients_lock:
        clients.append(ws)
    try:
        while True:
            await ws.receive_text()
    except WebSocketDisconnect:
        clients.remove(ws)


@app.get("/init_tree")
async def get_initial_tree():
    if not GIT_CLONE_DIR.exists():
        return {"status": "error", "message": "workspace directory not found"}

    entries = [e for e in GIT_CLONE_DIR.iterdir() if e.name not in IGNORE_DIRS]
    if not entries:
        return {"status": "empty", "message": "workspace is empty"}

    trees = [build_dir_tree(e) for e in entries]
    return {"status": "ok", "trees": trees}


@app.get("/file")
async def get_file_content(path: str = Query(...)):
    target = (GIT_CLONE_DIR / path).resolve()
    if not target.exists():
        return {"status": "error", "message": f"file not found: {target}"}
    if target.is_dir():
        return {"status": "error", "message": "cannot open directory"}
    try:
        content = target.read_text(encoding="utf-8", errors="ignore")
    except Exception as e:
        return {"status": "error", "message": f"read failed: {e}"}
    return {"status": "ok", "content": content}


@app.get("/history")
async def get_history(limit: int = 100):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("""
        SELECT repo_name, repo_url, description, language, total_files, indexed_at
        FROM repo_meta
        ORDER BY indexed_at DESC
        LIMIT %s;
    """, (limit,))
    rows = cur.fetchall()
    cur.close()
    conn.close()

    history = [
        {
            "repo_name": r[0],
            "repo_url": r[1],
            "description": r[2],
            "language": r[3],
            "total_files": r[4],
            "indexed_at": r[5],
        }
        for r in rows
    ]
    return {"status": "ok", "history": history}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("bridge_server:app", host="0.0.0.0", port=BRIDGE_PORT, reload=False)
