from pathlib import Path

import pytest

from atlasrag.ingestion.loader import (
    detect_document_format,
    load_local_document,
)
from atlasrag.ingestion.models import (
    DocumentFormat,
    DocumentSource,
)


@pytest.mark.parametrize(
    ("filename", "expected"),
    [
        ("policy.pdf", DocumentFormat.PDF),
        ("procedure.docx", DocumentFormat.DOCX),
        ("runbook.md", DocumentFormat.MARKDOWN),
        ("handbook.markdown", DocumentFormat.MARKDOWN),
        ("POLICY.PDF", DocumentFormat.PDF),
    ],
)
def test_detect_document_format(
    filename: str,
    expected: DocumentFormat,
) -> None:
    assert detect_document_format(Path(filename)) is expected


@pytest.mark.parametrize(
    "filename",
    [
        "notes.txt",
        "archive.zip",
        "README",
    ],
)
def test_detect_document_format_rejects_unsupported_files(
    filename: str,
) -> None:
    with pytest.raises(
        ValueError,
        match="Unsupported document extension",
    ):
        detect_document_format(Path(filename))


def test_load_local_document_reads_content_and_builds_version(
    tmp_path: Path,
) -> None:
    path = tmp_path / "policy.md"
    content = b"# Security Policy\n\nReport incidents immediately."
    path.write_bytes(content)

    source = DocumentSource(
        source_namespace="local-manifest",
        source_key="security/policy",
    )

    loaded = load_local_document(
        path=path,
        source=source,
    )

    assert loaded.path == path
    assert loaded.content == content
    assert loaded.version.source == source
    assert loaded.version.source_format is DocumentFormat.MARKDOWN
    assert loaded.version.size_bytes == len(content)


def test_load_local_document_rejects_empty_file(
    tmp_path: Path,
) -> None:
    path = tmp_path / "empty.md"
    path.write_bytes(b"")

    source = DocumentSource(
        source_namespace="local-manifest",
        source_key="empty",
    )

    with pytest.raises(ValueError, match="Document is empty"):
        load_local_document(path=path, source=source)


def test_load_local_document_rejects_unsupported_extension(
    tmp_path: Path,
) -> None:
    path = tmp_path / "notes.txt"
    path.write_text("Notes", encoding="utf-8")

    source = DocumentSource(
        source_namespace="local-manifest",
        source_key="notes",
    )

    with pytest.raises(ValueError, match="Unsupported document extension"):
        load_local_document(path=path, source=source)


def test_load_local_document_reports_missing_file(
    tmp_path: Path,
) -> None:
    path = tmp_path / "missing.md"

    source = DocumentSource(
        source_namespace="local-manifest",
        source_key="missing",
    )

    with pytest.raises(FileNotFoundError):
        load_local_document(path=path, source=source)
