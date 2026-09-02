import json
from importlib.metadata import version as distribution_version
from typing import Annotated, Literal, Self, cast
from uuid import UUID

from docling_core.transforms.chunker.doc_chunk import DocChunk
from docling_core.transforms.chunker.hybrid_chunker import (
    HybridChunker as DoclingCoreHybridChunker,
)
from docling_core.transforms.chunker.tokenizer.huggingface import (
    HuggingFaceTokenizer,
)
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

from atlasrag.ingestion.identity import calculate_content_sha256, create_chunk_id
from atlasrag.ingestion.models import DocumentVersion
from atlasrag.ingestion.parser import ParsedDocument

CHUNKER_NAME = "docling-hybrid"
CHUNKER_ALGORITHM_VERSION = 1


class ChunkingConfig(BaseModel):
    """Configuration that controls repeatable document chunking."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    tokenizer_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_max_tokens: int = Field(default=256, gt=0)
    chunk_max_tokens: int = Field(default=200, gt=0)
    merge_peers: bool = True

    @model_validator(mode="after")
    def validate_configuration(self) -> Self:
        if not self.tokenizer_model.strip():
            raise ValueError("tokenizer_model must not be blank")

        if self.chunk_max_tokens > self.embedding_max_tokens:
            raise ValueError("chunk_max_tokens must not exceed embedding_max_tokens")

        return self


def create_chunker_fingerprint(
    config: ChunkingConfig,
    *,
    docling_version: str,
) -> str:
    """Hash every setting that can change chunking output."""
    if not docling_version.strip():
        raise ValueError("docling_version must not be blank")

    payload = {
        "algorithm_version": CHUNKER_ALGORITHM_VERSION,
        "chunker_name": CHUNKER_NAME,
        "config": config.model_dump(mode="json"),
        "docling_version": docling_version,
    }
    canonical_json = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
    )

    return calculate_content_sha256(canonical_json.encode("utf-8"))


PositivePageNumber = Annotated[int, Field(gt=0)]


class DocumentChunk(BaseModel):
    """One searchable piece of a parsed document."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    schema_version: Literal[1] = 1
    chunk_id: UUID
    document_id: UUID
    document_version_id: UUID
    chunk_index: int = Field(ge=0)
    text: str
    contextualized_text: str
    token_count: int = Field(gt=0)
    headings: tuple[str, ...] = ()
    source_item_refs: tuple[str, ...] = ()
    page_numbers: tuple[PositivePageNumber, ...] = ()

    @field_validator("text", "contextualized_text")
    @classmethod
    def reject_blank_text(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("text values must not be blank")

        return value


class ChunkedDocument(BaseModel):
    """Complete chunking result for one document version."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    schema_version: Literal[1] = 1
    version: DocumentVersion
    config: ChunkingConfig
    docling_version: str
    chunker_fingerprint: str
    chunks: tuple[DocumentChunk, ...]

    @model_validator(mode="after")
    def validate_artifact(self) -> Self:
        if not self.docling_version.strip():
            raise ValueError("docling_version must not be blank")

        if not self.chunks:
            raise ValueError("chunks must not be empty")

        expected_indexes = tuple(range(len(self.chunks)))
        actual_indexes = tuple(chunk.chunk_index for chunk in self.chunks)
        if actual_indexes != expected_indexes:
            raise ValueError("chunk indexes must be contiguous and ordered")

        expected_fingerprint = create_chunker_fingerprint(
            self.config,
            docling_version=self.docling_version,
        )
        if self.chunker_fingerprint != expected_fingerprint:
            raise ValueError(
                "chunker_fingerprint does not match the chunking configuration"
            )

        document_id = self.version.source.document_id
        document_version_id = self.version.document_version_id

        for chunk in self.chunks:
            if chunk.document_id != document_id:
                raise ValueError("chunk document_id does not match the document")

            if chunk.document_version_id != document_version_id:
                raise ValueError(
                    "chunk document_version_id does not match the document version"
                )

            if chunk.token_count > self.config.embedding_max_tokens:
                raise ValueError("chunk exceeds the embedding token limit")

            expected_chunk_id = create_chunk_id(
                document_version_id,
                self.chunker_fingerprint,
                chunk.chunk_index,
                chunk.contextualized_text,
            )
            if chunk.chunk_id != expected_chunk_id:
                raise ValueError("chunk_id does not match the chunk content")

        return self


class DocumentChunkingError(RuntimeError):
    """Raised when Docling cannot produce safe chunks."""


class DoclingHybridChunker:
    """Create validated, token-aware chunks from a parsed document."""

    def __init__(self, config: ChunkingConfig | None = None) -> None:
        self.config = config or ChunkingConfig()
        self.docling_version = distribution_version("docling-core")
        self.fingerprint = create_chunker_fingerprint(
            self.config,
            docling_version=self.docling_version,
        )

        self._tokenizer = HuggingFaceTokenizer.from_pretrained(
            model_name=self.config.tokenizer_model,
            max_tokens=self.config.chunk_max_tokens,
        )
        self._chunker = DoclingCoreHybridChunker(
            tokenizer=self._tokenizer,
            merge_peers=self.config.merge_peers,
        )

    def chunk(self, parsed: ParsedDocument) -> ChunkedDocument:
        """Chunk one parsed document and preserve its lineage."""
        document_id = parsed.version.source.document_id
        document_version_id = parsed.version.document_version_id
        chunks: list[DocumentChunk] = []

        for chunk_index, base_chunk in enumerate(self._chunker.chunk(parsed.document)):
            docling_chunk = cast(DocChunk, base_chunk)
            contextualized_text = self._chunker.contextualize(chunk=docling_chunk)
            token_count = self._tokenizer.count_tokens(text=contextualized_text)

            if token_count > self.config.embedding_max_tokens:
                raise DocumentChunkingError(
                    f"Chunk {chunk_index} has {token_count} tokens; "
                    f"the embedding limit is {self.config.embedding_max_tokens}"
                )

            source_item_refs = tuple(
                dict.fromkeys(item.self_ref for item in docling_chunk.meta.doc_items)
            )
            page_numbers = tuple(
                sorted(
                    {
                        provenance.page_no
                        for item in docling_chunk.meta.doc_items
                        for provenance in item.prov
                    }
                )
            )

            chunks.append(
                DocumentChunk(
                    chunk_id=create_chunk_id(
                        document_version_id,
                        self.fingerprint,
                        chunk_index,
                        contextualized_text,
                    ),
                    document_id=document_id,
                    document_version_id=document_version_id,
                    chunk_index=chunk_index,
                    text=docling_chunk.text,
                    contextualized_text=contextualized_text,
                    token_count=token_count,
                    headings=tuple(docling_chunk.meta.headings or ()),
                    source_item_refs=source_item_refs,
                    page_numbers=page_numbers,
                )
            )

        if not chunks:
            raise DocumentChunkingError("Docling produced no chunks")

        return ChunkedDocument(
            version=parsed.version,
            config=self.config,
            docling_version=self.docling_version,
            chunker_fingerprint=self.fingerprint,
            chunks=tuple(chunks),
        )
