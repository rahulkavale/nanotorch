Common uv vs Poetry commands (quick comparison)

Project init
- uv: `uv init` (creates pyproject + basic files)
- poetry: `poetry init` (interactive pyproject creation)

Add dependency
- uv: `uv add <pkg>`
- poetry: `poetry add <pkg>`

Add dev dependency
- uv: `uv add --dev <pkg>` (goes into dev group)
- poetry: `poetry add <pkg>` with dev/group options

Remove dependency
- uv: `uv remove <pkg>`
- poetry: `poetry remove <pkg>`

Install/sync env
- uv: `uv sync` (sync env from lock)
- poetry: `poetry install` (installs from pyproject/lock)

Run commands in env
- uv: `uv run <cmd>`
- poetry: `poetry run <cmd>`

Lock dependencies
- uv: `uv lock`
- poetry: `poetry lock`

Build / publish
- uv: `uv build`
- poetry: `poetry build`, `poetry publish`

Notes
- uv uses a cross‑platform `uv.lock` lockfile.
- Poetry uses `poetry.lock`.
