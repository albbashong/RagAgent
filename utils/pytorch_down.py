import os
import subprocess
from pathlib import Path
import requests


class PyTorchTorchDownloader:
    """
    PyTorch GitHub에서 모든 버전의 torch/ 폴더만 sparse-checkout으로 가져오는 도구.
    """

    def __init__(self, base_dir: str = "pytorch_versions"):
        self.repo_url = "https://github.com/pytorch/pytorch.git"
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------
    # 1) GitHub에서 PyTorch 모든 태그 가져오기
    # ------------------------------
    def get_all_tags(self):
        print("[+] Fetching PyTorch tags...")

        url = "https://api.github.com/repos/pytorch/pytorch/tags?per_page=300"
        tags = []
        while url:
            r = requests.get(url)
            if r.status_code != 200:
                raise RuntimeError(f"GitHub API 요청 실패: {url}")

            data = r.json()
            for item in data:
                tags.append(item["name"])

            # GitHub API pagination 지원
            url = r.links.get("next", {}).get("url")

        # v숫자로 시작하는 태그만 필터링
        tags = [t for t in tags if t.startswith("v") and t[1].isdigit()]

        print(f"[+] Total PyTorch release tags: {len(tags)}")
        return sorted(tags)

    # ------------------------------
    # 2) 특정 버전의 torch/ 폴더만 sparse checkout
    # ------------------------------
    def download_torch_only(self, version_tag: str):
        print(f"\n==============================")
        print(f"  ✅ Downloading PyTorch {version_tag}")
        print(f"==============================")

        version_dir = self.base_dir / f"pytorch_{version_tag}"
        if version_dir.exists():
            print(f"[→] Skip: already exists {version_dir}")
            return

        version_dir.mkdir(parents=True, exist_ok=True)
        repo_dir = version_dir / "repo"

        # sparse clone
        cmds = [
            ["git", "clone", "--filter=blob:none", "--no-checkout", self.repo_url, str(repo_dir)],
            ["git", "-C", str(repo_dir), "sparse-checkout", "init", "--cone"],
            ["git", "-C", str(repo_dir), "sparse-checkout", "set", "torch"],
            ["git", "-C", str(repo_dir), "checkout", version_tag],
        ]

        for cmd in cmds:
            try:
                subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            except subprocess.CalledProcessError as e:
                print(f"❌ Error downloading {version_tag}: {e}")
                # 실패 시 디렉토리 삭제
                subprocess.run(["rm", "-rf", str(version_dir)])
                return

        print(f"[✅] Completed: {version_dir}/repo/torch")

    # ------------------------------
    # 3) 전체 버전 일괄 다운로드
    # ------------------------------
    def download_all(self):
        tags = self.get_all_tags()

        for tag in tags:
            self.download_torch_only(tag)

        print("\n=============================================")
        print("✅ All torch/ folders for all PyTorch versions downloaded!")
        print("=============================================")


# --------------------------------------
# ✅ 실행 예시
# --------------------------------------
if __name__ == "__main__":
    downloader = PyTorchTorchDownloader("pytorch_versions")
    downloader.download_all()
