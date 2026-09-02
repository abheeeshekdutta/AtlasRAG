from importlib.metadata import version as distribution_version
from typing import Protocol, cast

from sentence_transformers import SentenceTransformer

from atlasrag.ingestion.embedder import EmbeddingConfig

DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_MODEL_REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"


class _EmbeddingMatrix(Protocol):
    def tolist(self) -> list[list[float]]: ...


class _SentenceTransformerModel(Protocol):
    def get_embedding_dimension(self) -> int | None: ...

    def encode_document(
        self,
        sentences: list[str],
        *,
        batch_size: int,
        show_progress_bar: bool,
        convert_to_numpy: bool,
        normalize_embeddings: bool,
    ) -> _EmbeddingMatrix: ...


class SentenceTransformerEmbedder:
    """Generate local document embeddings with Sentence Transformers."""

    def __init__(
        self,
        *,
        model_name: str = DEFAULT_MODEL,
        model_revision: str = DEFAULT_MODEL_REVISION,
        normalize: bool = True,
        batch_size: int = 32,
        device: str | None = None,
        model: _SentenceTransformerModel | None = None,
    ) -> None:
        if not model_name.strip():
            raise ValueError("model_name must not be blank")
        if not model_revision.strip():
            raise ValueError("model_revision must not be blank")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")

        if model is None:
            model = cast(
                _SentenceTransformerModel,
                SentenceTransformer(
                    model_name,
                    revision=model_revision,
                    device=device,
                ),
            )

        dimensions = model.get_embedding_dimension()
        if dimensions is None or dimensions <= 0:
            raise ValueError("model must expose a positive embedding dimension")

        self.config = EmbeddingConfig(
            provider="sentence-transformers",
            model=model_name,
            model_revision=model_revision,
            dimensions=dimensions,
            normalize=normalize,
        )
        self.provider_version = distribution_version("sentence-transformers")
        self._batch_size = batch_size
        self._model = model

    def embed_documents(
        self,
        texts: tuple[str, ...],
    ) -> tuple[tuple[float, ...], ...]:
        if not texts:
            return ()

        matrix = self._model.encode_document(
            list(texts),
            batch_size=self._batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=self.config.normalize,
        )
        return tuple(tuple(float(value) for value in row) for row in matrix.tolist())
