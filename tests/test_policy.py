from pathlib import Path
from ingest.pipeline import load_knowledge_base
from config import ROOT


def test_only_verified_publishable_docs():
    chunks = load_knowledge_base(ROOT / 'data' / 'docs')
    assert chunks
    assert not any('unverified_note.md' == chunk.source for chunk in chunks)


def test_missing_metadata_fails_closed(tmp_path: Path):
    (tmp_path / 'raw.md').write_text('secret text', encoding='utf-8')
    (tmp_path / 'draft.md').write_text('---\nstatus: draft\npublishable: true\n---\nDraft', encoding='utf-8')
    (tmp_path / 'bad.md').write_text('---\nstatus: [invalid\n---\nBroken', encoding='utf-8')
    assert not load_knowledge_base(tmp_path)
