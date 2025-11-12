import ast
import re
from pathlib import Path

class CodeChunker:
    def __init__(self):
        pass

    # ---------------------------------------
    # [1] AST 기반 함수/클래스 분리 + main_body 병합
    # ---------------------------------------
    def chunk_structured_code(self, code: str, filename: str):
        try:
            tree = ast.parse(code)
        except SyntaxError:
            print(f"[Chunker] ⚠️ SyntaxError in {filename}, fallback to procedural.")
            return self.chunk_procedural_code(code, filename)

        lines = code.splitlines()
        chunks = []
        covered_lines = set()

        # ✅ 1️⃣ 함수/클래스 단위 chunk
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                start, end = node.lineno - 1, getattr(node, "end_lineno", node.lineno)
                func_code = "\n".join(lines[start:end])
                chunks.append({
                    "semantic_scope": f"function: {node.name}",
                    "hierarchical_context": f"{filename} > function {node.name}",
                    "content": func_code.strip(),
                })
                covered_lines.update(range(start, end))

            elif isinstance(node, ast.ClassDef):
                start, end = node.lineno - 1, getattr(node, "end_lineno", node.lineno)
                class_code = "\n".join(lines[start:end])
                chunks.append({
                    "semantic_scope": f"class: {node.name}",
                    "hierarchical_context": f"{filename} > class {node.name}",
                    "content": class_code.strip(),
                })
                covered_lines.update(range(start, end))

        # ✅ 2️⃣ 함수/클래스 밖 나머지 코드(main_body)
        main_body_lines = [
            line for i, line in enumerate(lines)
            if i not in covered_lines and line.strip()
        ]
        if main_body_lines:
            procedural_chunks = self.chunk_procedural_code("\n".join(main_body_lines), filename)
            chunks.extend(procedural_chunks)
            print(f"[Chunker] ✅ Detected main_body in {filename} ({len(main_body_lines)} lines)")

        return chunks

    # ---------------------------------------
    # [2] Procedural script용 휴리스틱 분리
    # ---------------------------------------
    def chunk_procedural_code(self, code: str, filename: str):
        lines = code.splitlines()
        chunks, current = [], []
        section_name = "main_body"

        def flush():
            if current:
                chunks.append({
                    "semantic_scope": f"section: {section_name}",
                    "hierarchical_context": f"{filename} > section > {section_name}",
                    "content": "\n".join(current).strip(),
                })

        for line in lines:
            stripped = line.strip()
            if not stripped:
                current.append(line)
                continue

            if re.match(r"^(import|from)\s+\w", stripped):
                if section_name != "imports":
                    flush(); current = []; section_name = "imports"
            elif re.match(r"^#{3,}", stripped):
                # 의미 없는 구분선은 무시
                header_text = re.sub(r"#+\s*", "", stripped).strip()
                if not header_text or re.match(r"^#+$", stripped):
                    current.append(line)
                    continue

                # ✅ 지금까지의 chunk를 종료하고 새 섹션 시작
                flush()
                current = []  # 주석도 포함 (상하 문맥 보존)
                section_name = header_text.lower().replace(" ", "_")
            elif re.match(r"^with\s+(tf\.|torch\.)", stripped):
                flush(); current = []
                matches = re.findall(r"['\"](.*?)['\"]", stripped)
                section_name = matches[0] if matches else "block"
            elif re.match(r"^with\s+tf\.Session", stripped) or re.match(r"^for\s+", stripped):
                flush(); current = []; section_name = "training_loop"
            elif re.match(r"^(t1\s*=|if\s+__name__|print\(|TRAIN\s*=|noise_mag\s*=|modeldir|np\.|os\.|tools\.|tf\.)", stripped):
                if section_name != "main_body":
                    flush(); current = []; section_name = "main_body"

            current.append(line)

        flush()

        if not chunks:
            chunks.append({
                "semantic_scope": "section: main_body",
                "hierarchical_context": f"{filename} > section > main_body",
                "content": code.strip()
            })

        return chunks

    # ---------------------------------------
    # [3] Dispatcher
    # ---------------------------------------
    def extract_chunks(self, file_path: Path):
        try:
            code = file_path.read_text(encoding="utf-8", errors="ignore")
        except Exception as e:
            print(f"[Chunker] ❌ 파일 읽기 실패: {file_path} ({e})")
            return []

        # ✅ Python 파일이면 AST + procedural 병합
        if file_path.suffix == ".py":
            print(f"[Chunker] 🧩 Running hybrid chunker for {file_path.name}")
            chunks = self.chunk_structured_code(code, file_path.name)
        else:
            chunks = self.chunk_procedural_code(code, file_path.name)

        print(f"[Chunker] ✅ {file_path.name}: {len(chunks)} chunks total")
        return chunks
