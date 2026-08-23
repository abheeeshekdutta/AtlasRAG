import hashlib
import re
from uuid import NAMESPACE_URL, UUID, uuid5

ATLASRAG_NAMESPACE = uuid5(NAMESPACE_URL, "urn:atlasrag")
DOCUMENT_NAMESPACE = uuid5(ATLASRAG_NAMESPACE, "document")

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
