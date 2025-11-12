import os
import re
import psycopg2
from psycopg2.extras import execute_values
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any

# ---------------------------------------------
# DB 설정
# ---------------------------------------------
DB_CONFIG = {
    "host": "172.17.0.2",
    "port": 5432,
    "user": "root",
    "password": "0000",
    "dbname": "Rag",
}

IGNORE_DIRS = {".git", "venv", "node_modules", "__pycache__"}


# ---------------------------------------------
# DB 연결
# ---------------------------------------------
def get_connection():
    """PostgreSQL 연결 객체 생성"""
    return psycopg2.connect(**DB_CONFIG)


# ---------------------------------------------
# 전체 테이블 데이터 리셋 (TRUNCATE)
# ---------------------------------------------
def reset_all_tables():
    """모든 주요 테이블 데이터 초기화"""
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("""
        TRUNCATE TABLE repo_meta, files_meta, repo_chunks, symbol_links
        RESTART IDENTITY CASCADE;
    """)
    conn.commit()
    cur.close()
    conn.close()
    print("✅ All tables truncated (data reset complete)")


# ---------------------------------------------
# README 요약 추출
# ---------------------------------------------
def extract_readme_summary(repo_path: Path) -> str:
    """README.md에서 첫 문단 추출"""
    for name in ["README.md", "README.MD", "readme.md"]:
        readme = repo_path / name
        if readme.exists():
            text = readme.read_text(encoding="utf-8", errors="ignore").strip()
            paragraphs = re.split(r"\n\s*\n", text)
            return paragraphs[0][:500]
    return ""


# ---------------------------------------------
# 디렉터리 기반 구조 요약
# ---------------------------------------------
def generate_structure_summary(repo_path: Path) -> str:
    dirs, files = [], []
    for root, _, fs in os.walk(repo_path):
        if any(ig in root for ig in IGNORE_DIRS):
            continue
        rel_root = os.path.relpath(root, repo_path)
        if rel_root != ".":
            dirs.append(rel_root)
        for f in fs:
            files.append(Path(f).suffix)
    file_types = {ext for ext in files if ext}
    desc = f"이 저장소는 {len(dirs)}개의 폴더와 {len(files)}개의 파일로 구성되어 있으며, 주요 확장자는 {', '.join(sorted(file_types))}입니다."
    return desc


# ---------------------------------------------
# 주요 언어 감지
# ---------------------------------------------
def detect_main_language(repo_path: Path) -> str:
    """확장자 기반 언어 감지"""
    code_ext_map = {
        ".py": "Python", ".js": "JavaScript", ".ts": "TypeScript", ".java": "Java",
        ".cpp": "C++", ".c": "C", ".cs": "C#", ".go": "Go", ".rs": "Rust",
        ".rb": "Ruby", ".php": "PHP", ".swift": "Swift", ".kt": "Kotlin",
        ".m": "Objective-C", ".scala": "Scala", ".r": "R", ".jl": "Julia",
        ".ipynb": "Python",
    }

    lang_count = {}
    for root, _, files in os.walk(repo_path):
        if any(ig in root for ig in IGNORE_DIRS):
            continue
        for f in files:
            ext = Path(f).suffix.lower()
            if ext in code_ext_map:
                lang = code_ext_map[ext]
                lang_count[lang] = lang_count.get(lang, 0) + 1

    if not lang_count:
        readme_path = repo_path / "README.md"
        if readme_path.exists():
            readme = readme_path.read_text(encoding="utf-8", errors="ignore").lower()
            if "tensorflow" in readme or "pytorch" in readme:
                return "Python"
            if "react" in readme or "node" in readme:
                return "JavaScript"
        return "Unknown"

    return max(lang_count, key=lang_count.get)


# ---------------------------------------------
# Repo 설명 생성
# ---------------------------------------------
def generate_repo_description(repo_path: Path) -> str:
    description = extract_readme_summary(repo_path)
    if not description:
        description = generate_structure_summary(repo_path)
    return description


# ---------------------------------------------
# Repo + Files 삽입
# ---------------------------------------------
def insert_repo_to_db(repo_name: str, repo_url: str, dest: Path):
    """
    - repo_meta: 메타데이터 등록
    - files_meta: 파일 목록 저장
    - description / language 자동 감지
    """
    try:
        conn = get_connection()
        cur = conn.cursor()

        description = generate_repo_description(dest)
        language = detect_main_language(dest)
        total_chunks = 0

        # ✅ repo_meta 삽입 또는 갱신
        cur.execute("""
            INSERT INTO repo_meta (repo_name, repo_url, description, language, total_files, total_chunks, indexed_at)
            VALUES (%s, %s, %s, ARRAY[%s], %s, %s, NOW())
            ON CONFLICT (repo_name)
            DO UPDATE SET
                repo_url = EXCLUDED.repo_url,
                description = EXCLUDED.description,
                language = EXCLUDED.language,
                total_files = EXCLUDED.total_files,
                total_chunks = EXCLUDED.total_chunks,
                indexed_at = NOW()
            RETURNING id;
        """, (repo_name, repo_url, description, language, 0, total_chunks))
        repo_id = cur.fetchone()[0]

        # ✅ 파일 목록 수집 및 삽입
        file_records = []
        for root, _, files in os.walk(dest):
            if any(ig in root for ig in IGNORE_DIRS):
                continue
            for f in files:
                file_path = os.path.relpath(os.path.join(root, f), dest)
                ext = Path(f).suffix.replace(".", "")
                file_records.append((repo_id, file_path, ext, None))

        if file_records:
            execute_values(cur, """
                INSERT INTO files_meta (repo_id, file_path, file_type, summary)
                VALUES %s;
            """, file_records)

        cur.execute(
            "UPDATE repo_meta SET total_files = %s WHERE id = %s;",
            (len(file_records), repo_id),
        )

        conn.commit()
        cur.close()
        conn.close()

        print(f"[DB] ✅ repo_meta + files_meta 등록 완료 ({repo_name}, {len(file_records)} files, lang={language})")

    except Exception as e:
        print(f"[DB] ❌ DB 삽입 오류: {e}")

    return repo_id
