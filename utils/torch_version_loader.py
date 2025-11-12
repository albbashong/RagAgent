from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple


@dataclass(slots=True)
class TorchCodeContext:
    """Container describing which torch files were loaded for the LLM."""

    version: str
    root_path: Path
    included_files: List[str] = field(default_factory=list)
    missing_files: List[str] = field(default_factory=list)
    content: str = ""

    def as_prompt_block(self) -> str:
        """Return a human-readable block that can be appended to an LLM prompt."""
        header = [
            "[Torch Code Context]",
            f"Version: {self.version}",
            f"Loaded files: {', '.join(self.included_files) if self.included_files else 'n/a'}",
        ]
        if self.missing_files:
            header.append(f"Missing files: {', '.join(self.missing_files)}")
        return "\n".join(header) + "\n\n" + self.content


class TorchVersionLoader:
    """Utility to resolve torch code across multiple downloaded PyTorch versions."""

    VERSION_DIR_PREFIX = "pytorch_"
    DEFAULT_FILES = ["torch/__init__.py"]
    VERSION_PATTERN = re.compile(
        r"(?:torch|pytorch)\s*(?:==|=|version|v)?\s*(v?\d+(?:\.\d+){1,3}(?:[a-z0-9\-]+)?)",
        re.IGNORECASE,
    )
    FILE_PATTERN = re.compile(r"(torch[\\/][\w.\-/]+\.py)", re.IGNORECASE)

    def __init__(
        self,
        base_dir: str | Path = "/app/pytorch_versions",
        *,
        max_chars: int = 20000,
        max_files: int = 3,
    ):
        self.base_dir = Path(base_dir)
        self.max_chars = max_chars
        self.max_files = max_files

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------
    def build_context_from_text(self, user_text: str) -> Optional[TorchCodeContext]:
        """Infer version and relevant files from the user text."""
        if not self.base_dir.exists():
            return None

        versions = self._list_available_versions()
        if not versions:
            return None

        version_hint = self._extract_version_hint(user_text)
        version_tag, version_path = self._resolve_version(version_hint, versions)

        rel_paths = self._extract_requested_files(user_text)
        if not rel_paths:
            rel_paths = list(self.DEFAULT_FILES)

        context = self._load_files(version_tag, version_path, rel_paths)
        if not context.content and rel_paths != self.DEFAULT_FILES:
            fallback_context = self._load_files(version_tag, version_path, self.DEFAULT_FILES)
            fallback_context.missing_files = list(set(context.missing_files + fallback_context.missing_files))
            fallback_context.included_files = list(dict.fromkeys(context.included_files + fallback_context.included_files))
            context = fallback_context
        return context

    # ------------------------------------------------------------------
    # Version handling
    # ------------------------------------------------------------------
    def _list_available_versions(self) -> List[Tuple[str, Path]]:
        result: List[Tuple[str, Path]] = []
        for path in sorted(self.base_dir.glob(f"{self.VERSION_DIR_PREFIX}*")):
            if not path.is_dir():
                continue
            version_tag = path.name.replace(self.VERSION_DIR_PREFIX, "", 1)
            result.append((version_tag, path))
        result.sort(key=lambda item: self._version_sort_key(item[0]))
        return result

    def _resolve_version(
        self,
        version_hint: Optional[str],
        versions: Sequence[Tuple[str, Path]],
    ) -> Tuple[str, Path]:
        if not versions:
            raise FileNotFoundError("No PyTorch versions found in base directory.")

        if version_hint:
            normalized_hint = self._normalize_version_tag(version_hint)
            for version_tag, version_path in versions:
                if self._normalize_version_tag(version_tag) == normalized_hint:
                    return version_tag, version_path

        # fallback -> latest available
        return versions[-1]

    def _extract_version_hint(self, text: str) -> Optional[str]:
        match = self.VERSION_PATTERN.search(text)
        if not match:
            return None
        return match.group(1)

    def _normalize_version_tag(self, tag: str) -> str:
        cleaned = tag.strip().lower()
        cleaned = cleaned.replace("pytorch", "").replace("torch", "")
        cleaned = cleaned.replace("_", "")
        cleaned = cleaned.lstrip("=")
        cleaned = re.sub(r"[^0-9a-z.\-v]", "", cleaned)
        if not cleaned.startswith("v"):
            cleaned = f"v{cleaned}"
        return cleaned

    def _version_sort_key(self, tag: str):
        numeric = self._normalize_version_tag(tag).lstrip("v")
        parts = []
        for chunk in numeric.split("."):
            if not chunk:
                continue
            chunk_clean = re.match(r"(\d+)", chunk)
            if chunk_clean:
                parts.append(int(chunk_clean.group(1)))
        return tuple(parts)

    # ------------------------------------------------------------------
    # File path handling
    # ------------------------------------------------------------------
    def _extract_requested_files(self, text: str) -> List[str]:
        matches = self.FILE_PATTERN.findall(text)
        if not matches:
            return []
        sanitized = []
        for raw in matches:
            rel = self._sanitize_rel_path(raw)
            if rel and rel not in sanitized:
                sanitized.append(rel)
        return sanitized[: self.max_files]

    def _sanitize_rel_path(self, raw: str) -> Optional[str]:
        candidate = raw.replace("\\", "/").strip().lstrip("/")
        if ".." in Path(candidate).parts:
            return None
        if not candidate.startswith("torch/"):
            candidate = f"torch/{candidate}"
        return candidate

    # ------------------------------------------------------------------
    # File loading
    # ------------------------------------------------------------------
    def _load_files(
        self,
        version_tag: str,
        version_root: Path,
        rel_paths: Iterable[str],
    ) -> TorchCodeContext:
        repo_root = version_root / "repo"
        included, missing = [], []
        remaining = self.max_chars
        chunks: List[str] = []

        for rel in rel_paths:
            if remaining <= 0:
                break
            file_path = repo_root / rel
            if not file_path.exists():
                missing.append(rel)
                continue
            try:
                data = file_path.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                missing.append(rel)
                continue
            snippet = data[:remaining]
            remaining -= len(snippet)
            included.append(rel)
            chunks.append(f"### {rel}\n{snippet}")

        content = "\n\n".join(chunks)
        return TorchCodeContext(
            version=version_tag,
            root_path=version_root,
            included_files=included,
            missing_files=missing,
            content=content,
        )


__all__ = ["TorchVersionLoader", "TorchCodeContext"]
