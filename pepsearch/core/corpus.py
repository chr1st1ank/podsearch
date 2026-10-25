"""Load and chunk the PEP documents."""

import re
from dataclasses import dataclass
from pathlib import Path

# Directory containing the tarball extraction created by setup.
DATA_DIR = Path(__file__).resolve().parents[2] / "data"


@dataclass
class Pep:
    number: int
    title: str
    text: str


def _title(text: str) -> str:
    for line in text.splitlines():
        if line.startswith("Title:"):
            return line.split(":", 1)[1].strip()
    return "(untitled)"


def load_peps(root: Path | None = None) -> list[Pep]:
    root = root or DATA_DIR
    files = sorted(root.glob("peps-main/peps/pep-*.rst"))
    if not files:
        files = sorted(root.glob("peps-main/pep-*.rst"))
    if not files:
        raise FileNotFoundError(f"no pep-*.rst files under {root}")
    return [
        Pep(int(f.stem.split("-")[1]), _title(text), text)
        for f in files
        for text in [f.read_text(encoding="utf-8")]
    ]


def clean(text: str) -> str:
    """Strip RST markup and the metadata header."""
    lines = text.splitlines()
    for i, line in enumerate(lines):
        if line.startswith(("Abstract:", "Status:")) and i > 0:
            text = "\n".join(lines[i:])
            break
    text = re.sub(r":pep:`(\d+)`", r"PEP \1", text)
    text = re.sub(r":\w+:`([^`]+)`", r"\1", text)
    text = re.sub(r"``([^`]+)``", r"\1", text)
    text = re.sub(r"`([^`]+)`", r"\1", text)
    text = re.sub(
        r"^\s*\.\. (code-block|note|versionadded|versionchanged|deprecated|seealso"
        r"|important|attention|warning|math|contents)::.*$",
        "",
        text,
        flags=re.MULTILINE,
    )
    text = re.sub(r"^[^\w\s]{4,}\s*$", "", text, flags=re.MULTILINE)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def chunk(text: str, size: int = 200, overlap: int = 40) -> list[str]:
    """Split into word chunks with overlap."""
    words = text.split()
    return [" ".join(words[i : i + size]) for i in range(0, len(words), size - overlap)]
