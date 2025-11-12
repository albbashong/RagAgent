import json
import re
from pathlib import Path
from managers.llm_manager import LLMManager
from managers.db_manager import get_connection
from psycopg2.extras import execute_values
from managers.chunker import CodeChunker
from managers.symbol import SymbolExtractor
from managers.embedding import EmbeddingManager


class LLMAgent:
    def __init__(self, llm: LLMManager | None = None):
        # Allow sharing a single LLMManager instance to avoid duplicate model loads.
        self.llm = llm or LLMManager()

    # -------------------------------------------------------------
    # 🔹 파일 요약
    # -------------------------------------------------------------
    def summarize_file(self, file_path: Path) -> str:
        """LLM 기반 코드/문서 요약"""
        ext = file_path.suffix.lower()

        # 비코드 파일 고정 문장
        if ext in [".png", ".jpg", ".jpeg", ".gif"]:
            return "이미지 리소스 파일입니다."
        if ext in [".npy", ".npz", ".pt", ".pkl", ".h5"]:
            return "머신러닝 모델의 데이터 또는 가중치 파일입니다."
        if ext in [".csv", ".xlsx"]:
            return "데이터셋 파일입니다."
        if ext in [".md", ".txt"]:
            return f"{file_path.name} 문서 파일입니다."

        try:
            text = file_path.read_text(encoding="utf-8", errors="ignore")[:4000]
        except Exception:
            return "파일을 읽을 수 없습니다."

        user_prompt = f"File name: {file_path.name}\n\nCode content:\n{text}"
        result = self.llm.generate(user_prompt, task="summarization", max_new_tokens=512)
        if "<summary>" in result:
            result = result.split("<summary>")[-1].split("</summary>")[0]
        return result.strip()

    # -------------------------------------------------------------
    # 🔹 코드 semantic chunk 생성
    # -------------------------------------------------------------
    def safe_json_parse(self, raw: str):
        """LLM 출력 문자열을 안전하게 JSON으로 변환 (깨짐 보정 포함)"""
        match = re.search(r"\[.*\]", raw, re.DOTALL)
        if not match:
            return []
        clean = match.group(0)

        # 기본 문자 정리
        clean = clean.replace("’", "'").replace("“", '"').replace("”", '"')
        clean = re.sub(r"^```(json)?|```$", "", clean.strip(), flags=re.MULTILINE)

        # 🧩 content 내부의 " escape 처리
        def escape_quotes_in_content(m):
            content = m.group(1)
            # \ 먼저 escape → " escape
            content = content.replace("\\", "\\\\").replace('"', '\\"')
            return f'"content": "{content}"'

        # "content": " ... " 부분을 찾아 내부 따옴표 이스케이프
        clean = re.sub(r'"content":\s*"(.*?)"', escape_quotes_in_content, clean, flags=re.DOTALL)

        # 객체 간 쉼표 누락 보정 (}{ → },{)
        clean = re.sub(r'(?<=\})(\s*)(?=\{)', ', ', clean)

        # 배열 또는 객체 끝의 트레일링 콤마 제거
        clean = re.sub(r",\s*(\]|\})", r"\1", clean)

        try:
            return json.loads(clean)
        except json.JSONDecodeError as e:
            print(f"[Chunk] ⚠️ Safe JSON decode error: {e}")
            print("---- raw json ----")
            print(clean[:])
            print("------------------")
            return []



    def extract_chunks(self, file_path: Path):
        chunker = CodeChunker()
        chunks = chunker.extract_chunks(file_path)
        if not chunks:
            print(f"[Chunk] ⚠️ {file_path.name}: no chunks found")
            return []
        print(f"[Chunk] ✅ {file_path.name}: {len(chunks)} chunks parsed locally")
        return chunks
        
    # -------------------------------------------------------------
    # 🔹 symbol_links (AST + LLM hybrid)
    # -------------------------------------------------------------
    def extract_symbol_links(self, repo_id: int, repo_dir: Path):
        """AST + LLM hybrid 방식으로 symbol_links 채우기"""
        from managers.symbol import SymbolExtractor  # 이미 상단 import되어 있으면 생략 가능
        from managers.db_manager import get_connection
        from psycopg2.extras import execute_values

        extractor = SymbolExtractor(llm=self.llm)
        all_links = []

        for py_file in repo_dir.rglob("*.py"):
            try:
                links = extractor.extract_links(py_file, repo_id)
                if links:
                    all_links.extend(links)
            except Exception as e:
                print(f"[SymbolExtractor] ⚠️ {py_file} skipped: {e}")

        if not all_links:
            print(f"[SymbolExtractor] ⚠️ No symbol links found for repo_id={repo_id}")
            return

        conn = get_connection()
        cur = conn.cursor()
        execute_values(cur, """
            INSERT INTO symbol_links (repo_id, source_symbol, target_symbol, relation_type, file_path)
            VALUES %s
        """, [
            (l["repo_id"], l["source_symbol"], l["target_symbol"], l["relation_type"], l["file_path"])
            for l in all_links
        ])
        conn.commit()
        cur.close()
        conn.close()
        print(f"[SymbolExtractor] ✅ Inserted {len(all_links)} symbol links for repo_id={repo_id}")

    # -------------------------------------------------------------
    # 🔹 repo_id 기준으로 파일 전체 요약
    # -------------------------------------------------------------
    def summarize_repo_files(self, repo_id: int, repo_dir: Path):
        conn = get_connection()
        cur = conn.cursor()
        cur.execute("SELECT id, file_path FROM files_meta WHERE repo_id = %s;", (repo_id,))
        files = cur.fetchall()

        for file_id, rel_path in files:
            fpath = repo_dir / rel_path
            if not fpath.exists():
                continue
            try:
                summary = self.summarize_file(fpath)
                cur.execute("UPDATE files_meta SET summary = %s WHERE id = %s;", (summary, file_id))
                conn.commit()
                print(f"[Summary] ✅ {rel_path}")
            except Exception as e:
                print(f"[Summary] ⚠️ {rel_path}: {e}")

        cur.close()
        conn.close()
        print(f"[Summary] ✅ repo_id={repo_id} summaries complete")

    # -------------------------------------------------------------
    # 🔹 repo_id 기준으로 전체 chunk 생성 후 DB 저장
    # -------------------------------------------------------------

    def chunk_repo_files(self, repo_id: int, repo_dir: Path):
        conn = get_connection()
        cur = conn.cursor()

        cur.execute("""
            SELECT id, file_path, file_type
            FROM files_meta
            WHERE repo_id = %s;
        """, (repo_id,))
        files = cur.fetchall()

        embedder = EmbeddingManager()  # ✅ SentenceTransformer 사용
        all_values = []
        total_chunks = 0

        for file_id, rel_path, file_type in files:
            path = repo_dir / rel_path
            if not path.exists() or file_type not in ["py", "js", "ts", "java", "cpp"]:
                continue

            chunks = self.extract_chunks(path)
            if not chunks:
                continue

            for c in chunks:
                emb = embedder.embed_text(c["content"])  # ✅ content 임베딩
                all_values.append((
                    repo_id,
                    file_id,
                    str(path),
                    file_type,
                    c["semantic_scope"],
                    c["hierarchical_context"],
                    c["content"],
                    len(c["content"].split()),
                    emb.tolist(),  # ✅ vector(1024)
                ))

            total_chunks += len(chunks)
            print(f"[Chunk+Embed] ✅ {path.name}: {len(chunks)} chunks embedded")

        if all_values:
            execute_values(cur, """
                INSERT INTO repo_chunks
                (repo_id, file_id, file_path, file_type,
                semantic_scope, hierarchical_context, content, token_count, embedding)
                VALUES %s;
            """, all_values)
            print(f"[Chunk+Embed] 🚀 Inserted {len(all_values)} chunks (with embeddings) for repo_id={repo_id}")

            # ✅ repo_meta 업데이트
            cur.execute("""
                UPDATE repo_meta
                SET total_chunks = %s
                WHERE id = %s;
            """, (total_chunks, repo_id))

        conn.commit()
        cur.close()
        conn.close()
