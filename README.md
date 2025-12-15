# PySpam 

![Tests Passing](https://github.com/sanchitram1/pyspam/actions/workflows/ci.yml/badge.svg)


A tool for identifying spam packages on Python's registry.

## Installation

1. Install uv from [astral.sh/uv](https://astral.sh/uv)
2. Clone the repository
3. Run `uv sync` to install dependencies
  - Alternatively, for all dependencies, including API ones,
run `uv sync --all-extras`

## Training

All the training code is in [feature_engineering](./feature_engineering). To
run it, just:

```sh
uv run pipeline.py 
```

To change the name of the input file, see
[settings.py](./feature_engineering/settings.py) .

## API

Need to run the below snippet for [bq.py](api/bq.py) to work

```
gcloud auth application-default login
```

To run the API, you need to run:

```bash
uv run uvicorn api.main:app --reload
```

## Dashboard

> [!note]
> To run the dashboard, you need to have two terminals, one that runs the API,
> and one that runs the dashboard. See API to determine how to run API

To run the dashboard, execute the following in your terminal

```bash
uv run streamlit run dashboard.py
```

## Testing

To see how many lines of code are successfully tested

```bash
coverage run --source=. -m pytest 
coverage report
```

Tests are written in pytest.

## Deploy

Requires: `gcloud`

```bash
gcloud auth login
gcloud config set project $PROJECT_ID
gcloud services enable cloudbuild.googleapis.com run.googleapis.com
gcloud run deploy pyspam-api --source .
```

## Model Context Protocol (MCP)

This repository includes a standalone MCP server (mcp_server/) that allows AI agents 
(like Claude Desktop or Cursor) to natively "consult" the PySpam API before suggesting 
packages.

### Quick Start (Requires [`pkgx`](https://pkgx.sh))

The server script is self-bootstrapping. It uses a shebang to automatically pull the 
correct Python version and dependencies (mcp, httpx) via pkgx + uv.

**1. Install pkgx (if not installed):**

```bash
curl -Ssf https://pkgx.sh | sh
```

**2. Make executable:**

```bash
chmod +x mcp_server/server.py
```

**3. Verify it runs:**

```bash
./mcp_server/server.py
```
If successful, it will hang silently (listening on stdio). Press Ctrl+C to exit.


### Client Configuration

To use this with your AI editor, add the configuration below to your MCP Settings file.

- Cursor: Cmd+Shift+P > MCP: Open Settings File
- Claude Desktop: ~/Library/Application Support/Claude/claude_desktop_config.json

**Option 1:** The pkgx Method (Recommended) Since the script is executable, you can 
point the client directly to the file. Note: You must use the absolute path to the repo.

```json
{
  "mcpServers": {
    "pyspam": {
      "command": "/ABSOLUTE/PATH/TO/pyspam/mcp_server/server.py",
      "args": []
    }
  }
}
```

**Option 2:** The Standard uv Method If you do not use pkgx, you can invoke the server 
using standard uv.

```json
{
  "mcpServers": {
    "pyspam": {
      "command": "uv",
      "args": [
        "run",
        "--with", 
        "mcp", 
        "--with", 
        "httpx",
        "/ABSOLUTE/PATH/TO/pyspam/mcp_server/server.py"
      ]
    }
  }
}
```

## Tasks

### install

Installs the packages exposed by this repo

```bash
uv pip install -e .
```

### sync-all

Gets all the requirements you need for developing everything

Requires: install

```bash
uv sync all-extras
```

### test 

Runs all the tests

```bash
pytest .
```

### lint

```bash
ruff format . 
ruff check . --fix
```