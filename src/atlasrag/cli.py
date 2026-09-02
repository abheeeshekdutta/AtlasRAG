import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

DEFAULT_ARTIFACT_ROOT = Path(".atlasrag/artifacts")


def build_parser() -> argparse.ArgumentParser:
    """Build the AtlasRAG command-line parser."""
    parser = argparse.ArgumentParser(
        prog="atlasrag",
        description="Ingest local documents and search them semantically.",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    ingest = commands.add_parser("ingest", help="Ingest one local document")
    ingest.add_argument("path", type=Path, help="PDF, DOCX, or Markdown document")
    ingest.add_argument(
        "--artifacts",
        type=Path,
        default=DEFAULT_ARTIFACT_ROOT,
        help=f"Artifact directory (default: {DEFAULT_ARTIFACT_ROOT})",
    )
    ingest.add_argument(
        "--namespace",
        default="local-filesystem",
        help="Stable source namespace",
    )
    ingest.add_argument(
        "--source-key",
        help="Stable source key (default: absolute document path)",
    )

    search = commands.add_parser("search", help="Search ingested documents")
    search.add_argument("query", help="Natural-language search query")
    search.add_argument(
        "--artifacts",
        type=Path,
        default=DEFAULT_ARTIFACT_ROOT,
        help=f"Artifact directory (default: {DEFAULT_ARTIFACT_ROOT})",
    )
    search.add_argument("--top-k", type=_positive_int, default=5)
    search.add_argument("--minimum-score", type=_score, default=-1.0)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the AtlasRAG command-line application."""
    arguments = build_parser().parse_args(argv)

    try:
        if arguments.command == "ingest":
            return _run_ingest(arguments)
        return _run_search(arguments)
    except (OSError, ValueError, RuntimeError) as error:
        print(f"atlasrag: {error}", file=sys.stderr)
        return 1


def _run_ingest(arguments: argparse.Namespace) -> int:
    from atlasrag.ingestion.chunker import DoclingHybridChunker
    from atlasrag.ingestion.models import DocumentSource
    from atlasrag.ingestion.parser import DoclingParser
    from atlasrag.ingestion.pipeline import ingest_local_document
    from atlasrag.ingestion.sentence_transformer_embedder import (
        SentenceTransformerEmbedder,
    )

    path: Path = arguments.path
    source_key = arguments.source_key or str(path.resolve())
    embedder = SentenceTransformerEmbedder()
    result = ingest_local_document(
        path=path,
        source=DocumentSource(
            source_namespace=arguments.namespace,
            source_key=source_key,
        ),
        artifact_root=arguments.artifacts,
        parser=DoclingParser(),
        chunker=DoclingHybridChunker(),
        embedder=embedder,
    )
    _print_json(
        {
            "document_id": str(result.parsed.version.source.document_id),
            "document_version_id": str(result.parsed.version.document_version_id),
            "chunks": len(result.chunked.chunks),
            "embedding_model": result.embedded.config.model,
            "artifacts": {
                "parsed": str(result.artifact_path),
                "chunks": str(result.chunk_artifact_path),
                "embeddings": str(result.embedding_artifact_path),
            },
        }
    )
    return 0


def _run_search(arguments: argparse.Namespace) -> int:
    from atlasrag.answering.context import ContextBuilder, ContextConfig
    from atlasrag.ingestion.sentence_transformer_embedder import (
        SentenceTransformerEmbedder,
    )
    from atlasrag.retrieval.artifact_corpus import load_artifact_corpus

    embedder = SentenceTransformerEmbedder()
    corpus = load_artifact_corpus(
        artifact_root=arguments.artifacts,
        embedder=embedder,
    )
    hits = corpus.index.search(
        arguments.query,
        top_k=arguments.top_k,
        minimum_score=arguments.minimum_score,
    )
    context = ContextBuilder(ContextConfig(max_chunks=arguments.top_k)).build(hits)
    _print_json(
        {
            "query": arguments.query,
            "corpus": {
                "documents": corpus.document_count,
                "chunks": corpus.chunk_count,
            },
            "results": [
                {
                    "citation": block.citation.label,
                    "score": block.score,
                    "source_namespace": block.citation.source_namespace,
                    "source_key": block.citation.source_key,
                    "headings": block.citation.headings,
                    "page_numbers": block.citation.page_numbers,
                    "chunk_id": str(block.citation.chunk_id),
                    "text": block.text,
                }
                for block in context.blocks
            ],
        }
    )
    return 0


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def _score(value: str) -> float:
    parsed = float(value)
    if not -1.0 <= parsed <= 1.0:
        raise argparse.ArgumentTypeError("must be between -1 and 1")
    return parsed


def _print_json(value: object) -> None:
    print(json.dumps(value, ensure_ascii=False, indent=2))
