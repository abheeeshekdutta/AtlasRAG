import hashlib
import re
from uuid import NAMESPACE_URL, UUID, uuid5

ATLASRAG_NAMESPACE = uuid5(NAMESPACE_URL, "urn:atlasrag")
DOCUMENT_NAMESPACE = uuid5(ATLASRAG_NAMESPACE, "document")
CHUNK_NAMESPACE = uuid5(ATLASRAG_NAMESPACE, "chunk")

SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")


def calculate_content_sha256(content: bytes) -> str:
    """Return the lowercase hexadecimal SHA-256 digest."""
    return hashlib.sha256(content).hexdigest()


def create_document_id(source_namespace: str, source_key: str) -> UUID:
    """Create a deterministic ID for one logical source document."""
    if not source_namespace.strip():
        raise ValueError("source_namespace must not be blank")

    if not source_key.strip():
        raise ValueError("source_key must not be blank")

    source_namespace_id = uuid5(DOCUMENT_NAMESPACE, source_namespace)
    return uuid5(source_namespace_id, source_key)


def create_document_version_id(
    document_id: UUID,
    content_sha256: str,
) -> UUID:
    """Create a deterministic ID for one exact document version."""
    if SHA256_PATTERN.fullmatch(content_sha256) is None:
        raise ValueError(
            "content_sha256 must be a 64-character lowercase hexadecimal digest"
        )

    return uuid5(document_id, content_sha256)


def create_chunk_id(
    document_version_id: UUID,
    chunker_fingerprint: str,
    chunk_index: int,
    contextualized_text: str,
) -> UUID:
    """Create a deterministic ID for one chunk."""
    if not chunker_fingerprint.strip():
        raise ValueError("chunker_fingerprint must not be blank")

    if chunk_index < 0:
        raise ValueError("chunk_index must not be negative")

    if not contextualized_text.strip():
        raise ValueError("contextualized_text must not be blank")

    document_version_namespace = uuid5(
        CHUNK_NAMESPACE,
        str(document_version_id),
    )
    text_sha256 = calculate_content_sha256(contextualized_text.encode("utf-8"))
    chunk_key = f"{chunker_fingerprint}:{chunk_index}:{text_sha256}"

    return uuid5(document_version_namespace, chunk_key)
