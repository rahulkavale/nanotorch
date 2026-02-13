from __future__ import annotations

# Export the interactive notebook to a shareable HTML file.
# This keeps a single source of truth (the .ipynb) while making
# a static artifact you can send or publish.

from nanotorch.export_html import main


if __name__ == "__main__":
    main()
