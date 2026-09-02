from collections.abc import Sequence


def main(argv: Sequence[str] | None = None) -> int:
    """Run the AtlasRAG command-line application."""
    from atlasrag.cli import main as cli_main

    return cli_main(argv)
