import json
from pathlib import Path

import pytest

from atlasrag.answering.context import RetrievedContext
from atlasrag.cli import main
from atlasrag.ingestion.embedder import EmbeddingConfig


class CliTestEmbedder:
    def __init__(self) -> None:
        self.config = EmbeddingConfig(
            provider="test",
            model="cli-test-model",
            model_revision="revision-1",
            dimensions=3,
            normalize=True,
        )
        self.provider_version = "1.0.0"

    def embed_documents(
        self,
        texts: tuple[str, ...],
    ) -> tuple[tuple[float, ...], ...]:
        vectors = (
            (0.0, 1.0, 0.0),
            (1.0, 0.0, 0.0),
            (0.0, 0.0, 1.0),
            (0.1, 0.9, 0.0),
        )
        return tuple(vectors[index] for index, _ in enumerate(texts))

    def embed_query(self, text: str) -> tuple[float, ...]:
        return (1.0, 0.0, 0.0)


class CliTestGenerator:
    def __init__(self, *, model: str, max_output_tokens: int) -> None:
        self.model = model
        self.max_output_tokens = max_output_tokens

    def generate(self, *, query: str, context: RetrievedContext) -> str:
        return "Report the incident immediately [1]."


@pytest.fixture
def fake_local_embedder(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "atlasrag.ingestion.sentence_transformer_embedder.SentenceTransformerEmbedder",
        CliTestEmbedder,
    )
    monkeypatch.setattr(
        "atlasrag.answering.openai_generator.OpenAIAnswerGenerator",
        CliTestGenerator,
    )


def test_cli_ingests_then_searches_persisted_document(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    fake_local_embedder: None,
) -> None:
    artifact_root = tmp_path / "artifacts"
    ingest_status = main(
        [
            "ingest",
            "tests/fixtures/documents/security_policy.md",
            "--artifacts",
            str(artifact_root),
            "--namespace",
            "cli-test",
            "--source-key",
            "security/policy",
        ]
    )
    ingest_output = json.loads(capsys.readouterr().out)

    assert ingest_status == 0
    assert ingest_output["chunks"] == 4
    assert ingest_output["embedding_model"] == "cli-test-model"
    assert all(Path(path).is_file() for path in ingest_output["artifacts"].values())

    search_status = main(
        [
            "search",
            "How do I report an incident?",
            "--artifacts",
            str(artifact_root),
            "--top-k",
            "2",
            "--minimum-score",
            "0.5",
        ]
    )
    search_output = json.loads(capsys.readouterr().out)

    assert search_status == 0
    assert search_output["corpus"] == {"documents": 1, "chunks": 4}
    assert len(search_output["results"]) == 1
    assert search_output["results"][0]["citation"] == "[1]"
    assert search_output["results"][0]["headings"] == [
        "Security Incident Policy",
        "Reporting procedure",
    ]
    assert "report" in search_output["results"][0]["text"].casefold()

    ask_status = main(
        [
            "ask",
            "What should I do?",
            "--artifacts",
            str(artifact_root),
            "--model",
            "requested-model",
            "--top-k",
            "1",
            "--max-output-tokens",
            "123",
        ]
    )
    ask_output = json.loads(capsys.readouterr().out)

    assert ask_status == 0
    assert ask_output["answer"] == "Report the incident immediately [1]."
    assert ask_output["model"] == "requested-model"
    assert ask_output["citations"][0]["label"] == "[1]"


def test_cli_reports_missing_artifact_root(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    fake_local_embedder: None,
) -> None:
    status = main(
        [
            "search",
            "query",
            "--artifacts",
            str(tmp_path / "missing"),
        ]
    )
    captured = capsys.readouterr()

    assert status == 1
    assert captured.out == ""
    assert "Artifact root does not exist" in captured.err


@pytest.mark.parametrize(
    "arguments",
    [
        ["search", "query", "--top-k", "0"],
        ["search", "query", "--minimum-score", "1.1"],
    ],
)
def test_cli_rejects_invalid_search_limits(arguments: list[str]) -> None:
    with pytest.raises(SystemExit) as raised:
        main(arguments)

    assert raised.value.code == 2
