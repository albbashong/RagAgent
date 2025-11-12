import ast, re, json
from pathlib import Path
from typing import List, Dict, Any

class SymbolExtractor:
    def __init__(self, llm):
        self.llm = llm  # LLMManager instance with generate()

    # ----------------------------------------
    # [1] AST 기반 추출 (기본)
    # ----------------------------------------
    def extract_links_ast(self, code: str, file_path: Path, repo_id: int) -> List[Dict[str, Any]]:
        try:
            tree = ast.parse(code)
        except SyntaxError:
            print(f"[SymbolExtractor] ⚠️ SyntaxError: {file_path}")
            return []

        filename = file_path.name
        links = []

        # imports
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    links.append({
                        "repo_id": repo_id,
                        "source_symbol": filename,
                        "target_symbol": alias.name,
                        "relation_type": "imports",
                        "file_path": str(file_path)
                    })
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                for alias in node.names:
                    target = f"{module}.{alias.name}".strip(".")
                    links.append({
                        "repo_id": repo_id,
                        "source_symbol": filename,
                        "target_symbol": target,
                        "relation_type": "imports",
                        "file_path": str(file_path)
                    })

        # function calls
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                func_name = node.name
                for subnode in ast.walk(node):
                    if isinstance(subnode, ast.Call):
                        target = None
                        if isinstance(subnode.func, ast.Name):
                            target = subnode.func.id
                        elif isinstance(subnode.func, ast.Attribute):
                            target = subnode.func.attr
                        if target:
                            links.append({
                                "repo_id": repo_id,
                                "source_symbol": func_name,
                                "target_symbol": target,
                                "relation_type": "calls",
                                "file_path": str(file_path)
                            })

        # inheritance
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                for base in node.bases:
                    parent = getattr(base, "id", None) or getattr(base, "attr", None)
                    if parent:
                        links.append({
                            "repo_id": repo_id,
                            "source_symbol": node.name,
                            "target_symbol": parent,
                            "relation_type": "inherits",
                            "file_path": str(file_path)
                        })
        return links

    # ----------------------------------------
    # [2] LLM 보조 (config 기반)
    # ----------------------------------------
    def extract_links_llm(self, code: str, file_path: Path, repo_id: int) -> List[Dict[str, Any]]:
        user_prompt = f"File: {file_path.name}\n\nCode:\n```python\n{code}\n```"
        try:
            res = self.llm.generate(user_prompt, task="symbol_links", max_new_tokens=512)
            print(f"[SymbolExtractor-LLM Raw] {res[:300]}")

            parsed = []
            for line in res.splitlines():
                if "|" not in line:
                    continue
                parts = [p.strip() for p in line.split("|")]
                if len(parts) != 3:
                    continue
                src, tgt, rel = parts
                if rel not in {"calls", "imports", "inherits"}:
                    continue
                parsed.append({
                    "repo_id": repo_id,
                    "source_symbol": src,
                    "target_symbol": tgt,
                    "relation_type": rel,
                    "file_path": str(file_path)
                })
            return parsed
        except Exception as e:
            print(f"[SymbolExtractor] ⚠️ LLM parse failed: {file_path} ({e})")
            return []


    # ----------------------------------------
    # [3] 통합 호출 (AST + LLM hybrid)
    # ----------------------------------------
    def extract_links(self, file_path: Path, repo_id: int) -> List[Dict[str, Any]]:
        try:
            code = file_path.read_text(encoding="utf-8", errors="ignore")
        except Exception as e:
            print(f"[SymbolExtractor] ❌ File read failed: {file_path} ({e})")
            return []

        ast_links = self.extract_links_ast(code, file_path, repo_id)
        print(f"[SymbolExtractor] ✅ AST found {len(ast_links)} links in {file_path.name}")

        # alias/dynamic pattern 감지
        if re.search(r"import\s+\w+\s+as\s+\w+|torch\.|tf\.|np\.", code):
            llm_links = self.extract_links_llm(code, file_path, repo_id)
            print(f"[SymbolExtractor] 🧠 LLM refined {len(llm_links)} links in {file_path.name}")

            all_links = {
                (l["source_symbol"], l["target_symbol"], l["relation_type"]): l
                for l in ast_links + llm_links
            }
            return list(all_links.values())

        return ast_links
