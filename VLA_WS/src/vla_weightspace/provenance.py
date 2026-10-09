"""Verify the research dependency rather than merely claiming an expected pin."""
import importlib.metadata
import json
from pathlib import Path
import subprocess
from urllib.parse import urlparse
from urllib.request import url2pathname


LEROBOT_COMMIT = "b9cb121cb7d4e3e26ec5c906d08dda68d151ba52"


def verify_lerobot():
    distribution = importlib.metadata.distribution("lerobot")
    text = distribution.read_text("direct_url.json")
    revision = None
    if text:
        direct = json.loads(text)
        revision = direct.get("vcs_info", {}).get("commit_id")
        if revision is None and direct.get("dir_info", {}).get("editable"):
            path = Path(url2pathname(urlparse(direct["url"]).path))
            revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=path, text=True).strip()
            dirty = subprocess.check_output(["git", "status", "--porcelain", "--", "src", "pyproject.toml"], cwd=path, text=True).strip()
            if dirty:
                raise RuntimeError("Editable LeRobot source has uncommitted changes")
    if revision != LEROBOT_COMMIT:
        raise RuntimeError(f"Expected LeRobot commit {LEROBOT_COMMIT}; found {revision!r}. Install the repository's research extra.")
    return {"commit": revision, "version": distribution.version}
