from typing import Any

import pytest

from atlasrag.ingestion.sentence_transformer_embedder import (
    SentenceTransformerEmbedder,
)


class FakeMatrix:
    def __init__(self, rows: list[list[float]]) -> None:
        self.rows = rows

    def tolist(self) -> list[list[float]]:
        return self.rows


class FakeSentenceTransformer:
    def __init__(self, dimensions: int | None = 3) -> None:
        self.dimensions = dimensions
        self.calls: list[dict[str, Any]] = []

    def get_embedding_dimension(self) -> int | None:
        return self.dimensions

    def encode_document(
        self,
        sentences: list[str],
        *,
        batch_size: int,
        show_progress_bar: bool,
        convert_to_numpy: bool,
        normalize_embeddings: bool,
    ) -> FakeMatrix:
        self.calls.append(
            {
                "sentences": sentences,
                "batch_size": batch_size,
                "show_progress_bar": show_progress_bar,
                "convert_to_numpy": convert_to_numpy,
                "normalize_embeddings": normalize_embeddings,
            }
        )
        return FakeMatrix(
            [
                [float(index + dimension) for dimension in range(3)]
                for index, _ in enumerate(sentences)
            ]
        )


def test_sentence_transformer_embedder_exposes_reproducible_configuration() -> None:
    embedder = SentenceTransformerEmbedder(
        model_name="organization/model",
        model_revision="abc123",
        normalize=False,
        model=FakeSentenceTransformer(),
    )

    assert embedder.config.provider == "sentence-transformers"
    assert embedder.config.model == "organization/model"
    assert embedder.config.model_revision == "abc123"
    assert embedder.config.dimensions == 3
    assert not embedder.config.normalize
    assert embedder.provider_version


def test_sentence_transformer_embedder_uses_document_encoding_options() -> None:
    model = FakeSentenceTransformer()
    embedder = SentenceTransformerEmbedder(
        model_name="organization/model",
        model_revision="abc123",
        batch_size=2,
        normalize=True,
        model=model,
    )

    result = embedder.embed_documents(("first", "second"))

    assert result == ((0.0, 1.0, 2.0), (1.0, 2.0, 3.0))
    assert model.calls == [
        {
            "sentences": ["first", "second"],
            "batch_size": 2,
            "show_progress_bar": False,
            "convert_to_numpy": True,
            "normalize_embeddings": True,
        }
    ]


def test_sentence_transformer_embedder_skips_empty_batches() -> None:
    model = FakeSentenceTransformer()
    embedder = SentenceTransformerEmbedder(
        model_name="organization/model",
        model_revision="abc123",
        model=model,
    )

    assert embedder.embed_documents(()) == ()
    assert model.calls == []


@pytest.mark.parametrize(
    ("argument", "value", "message"),
    [
        ("model_name", " ", "model_name must not be blank"),
        ("model_revision", "", "model_revision must not be blank"),
        ("batch_size", 0, "batch_size must be positive"),
    ],
)
def test_sentence_transformer_embedder_rejects_invalid_configuration(
    argument: str,
    value: str | int,
    message: str,
) -> None:
    arguments: dict[str, Any] = {
        "model_name": "organization/model",
        "model_revision": "abc123",
        "batch_size": 32,
        "model": FakeSentenceTransformer(),
    }
    arguments[argument] = value

    with pytest.raises(ValueError, match=message):
        SentenceTransformerEmbedder(**arguments)


@pytest.mark.parametrize("dimensions", [None, 0, -1])
def test_sentence_transformer_embedder_requires_known_positive_dimensions(
    dimensions: int | None,
) -> None:
    with pytest.raises(ValueError, match="positive embedding dimension"):
        SentenceTransformerEmbedder(
            model_name="organization/model",
            model_revision="abc123",
            model=FakeSentenceTransformer(dimensions),
        )
