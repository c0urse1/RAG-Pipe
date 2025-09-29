from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import click

from ...config.composition import build_application
from ...application.dto.ingest_dto import IngestDocumentDTO
from ...application.dto.query_dto import QueryDTO


@click.group()
def cli() -> None:
    """BU Superagent CLI"""


@cli.command()
@click.option("--file", "file_path", type=click.Path(path_type=Path), required=True, help="Path to a text file to ingest")
@click.option("--title", type=str, default=None, help="Optional title")
def ingest(file_path: Path, title: Optional[str]) -> None:
    app = build_application()
    text = file_path.read_text(encoding="utf-8")
    doc = IngestDocumentDTO(id=file_path.stem, title=title or file_path.name, content=text)
    app["ingest"].execute([doc])
    click.echo("Ingested 1 document.")


@cli.command()
@click.option("--text", type=str, required=True, help="Query text")
@click.option("--top-k", type=int, default=5)
def query(text: str, top_k: int) -> None:
    app = build_application()
    results = app["query"].execute(QueryDTO(text=text, top_k=top_k))
    click.echo(json.dumps(results, indent=2))
