"""Read local markdown with conservative publication policy.

An absent, broken, or unverified frontmatter means the file is not indexed.
This stage is a local-file prototype; a durable ingest worker comes later.
"""
from dataclasses import dataclass
from pathlib import Path
import re
import yaml


@dataclass(frozen=True)
class Chunk:
    id: str
    source: str
    title: str
    text: str


def _split_frontmatter(raw: str) -> tuple[dict, str]:
    match = re.match(r'\A---\s*\n(.*?)\n---\s*\n(.*)\Z', raw, re.S)
    if not match:
        return {}, ''
    try:
        metadata = yaml.safe_load(match.group(1))
    except yaml.YAMLError:
        return {}, ''
    if not isinstance(metadata, dict):
        return {}, ''
    return metadata, match.group(2).strip()


def _chunk_text(text: str, max_chars: int = 700) -> list[str]:
    paragraphs = [x.strip() for x in re.split(r'\n\s*\n', text) if x.strip()]
    chunks = []
    current = ''
    for paragraph in paragraphs:
        # Hard bound even for documents without paragraph breaks.
        pieces = [paragraph[i:i+max_chars] for i in range(0, len(paragraph), max_chars)]
        for piece in pieces:
            if current and len(current) + len(piece) + 2 > max_chars:
                chunks.append(current)
                current = ''
            current = f'{current}\n\n{piece}'.strip() if current else piece
    if current:
        chunks.append(current)
    return chunks


def load_knowledge_base(folder: Path) -> list[Chunk]:
    """Idempotent local loading; only verified, publishable documents are indexed."""
    chunks: list[Chunk] = []
    if not folder.exists():
        return chunks
    for file in sorted(folder.rglob('*.md')):
        meta, body = _split_frontmatter(file.read_text(encoding='utf-8'))
        if meta.get('status') != 'verified' or meta.get('publishable') is not True:
            continue
        relative = file.relative_to(folder).as_posix()
        for i, text in enumerate(_chunk_text(body)):
            chunks.append(Chunk(id=f'{relative}::{i}', source=relative,
                                title=str(meta.get('title') or file.stem), text=text))
    return chunks
