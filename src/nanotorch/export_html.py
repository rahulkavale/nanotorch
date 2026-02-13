from __future__ import annotations

# Export the interactive notebook to a shareable HTML file.
# This keeps a single source of truth (the .ipynb) while making
# a static artifact you can send or publish.

from pathlib import Path

import nbformat
from nbconvert import HTMLExporter


def main() -> None:
    nb_path = Path("notebooks/step_through_training.ipynb")
    out_path = Path("artifacts/plots/step_through_training.html")

    nb = nbformat.read(nb_path, as_version=4)
    exporter = HTMLExporter()
    body, _ = exporter.from_notebook_node(nb)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(body, encoding="utf-8")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
