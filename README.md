# pyspam

![Tests Passing](https://github.com/sanchitram1/pyspam/actions/workflows/ci.yml/badge.svg)
[![Coverage Status](https://coveralls.io/repos/github/sanchitram1/pyspam/badge.svg?branch=main)
](https://coveralls.io/github/sanchitram1/pyspam?branch=main)

A tool for identifying spam packages on Python's registry.

## Pre-requisites

1. [`uv`](https://astral.sh/uv)
2. [`gcloud`](https://docs.cloud.google.com/sdk/docs/install-sdk)
3. Not required, but [`pkgx`](https://pkgx.sh) is useful as well.
4. Not required, but [`xc`](https://xcfile.dev) is useful as well.

> [!TIP]
> If you have pkgx, all you need to do is prefix everything with `pkgx ...` and it works
> like magic

## Installation

1. Install uv from [astral.sh/uv](https://astral.sh/uv)
2. Clone the repository
3. Run `uv sync` to install dependencies
  - Alternatively, for all dependencies, including API ones,
run `uv sync --all-extras`

## Training

Training can be broken down into three steps:

### 1. Raw Data

Our data source is from Google BigQuery – `pypi.distribution_metadata`, which contains
all metadata information for every single package published to PyPI. The
[training.sql](sql/training.sql) query will generate a labeled dataset of spam vs. non-
spam python packages

> [!WARNING]
> In the training.sql file, we reference `project.ground_truth`, which is a labeled
> dataset that we authored to training the data. We haven't published the dataset to BQ
> yet, but for now, you would need need to create a table called `ground_truth` which
> contains two columns: `package_name` and `is_spam`, which is your source for the
> labels for spam python packages.

### 2. Feature Engineering

The code to translate the raw data into a set of features for an ML model is in
[feature_engineering](./feature_engineering) . To run it, just:

```sh
uv run feature_engineering/pipeline.py \
  --input /path/to/input/file.json \
  --output /path/to/output/file.json
```

You can use defaults set in [settings.py](./feature_engineering/settings.py) as well.
We're gonna author a README in that folder to explain how the pipeline works.

### 3. Model training

Currently the notebook [models.ipynb](training/models.ipynb) is the source to generate
all the joblib models that we use for our analysis.
[#10](https://github.com/sanchitram1/pyspam/issues/10) tracks the changes we need to
make to this process

The output is a set of joblib files that are written to [models](models/) .

## API

We implemented a secured API to demonstrate how an external service (like an MCP server
or LLM) could use this model to evaluate PyPI packages. The code lives in
[api](api/main.py) .

### Local Setup

To run it locally, you need two things:
1. **Google Credentials:** Authenticate so [bq.py](api/bq.py) can query BigQuery.
2. **Local Secret:** Set a dummy secret key for JWT generation.

```bash
# 1. Authenticate with Google Cloud
gcloud auth application-default login

# 2. Set a temporary secret for local testing
export API_TOKEN_SECRET="local-dev-secret"

# 3. Start the server
uv run uvicorn api.main:app --reload
```

### Usage

**Step 1: Generate an API Key**

The API is protected by JWT authentication. You must first generate a temporary access
token, simulating how a user on the portfolio website would gain access.

```bash
curl -X POST [http://127.0.0.1:8000/generate-key](http://127.0.0.1:8000/generate-key)
```

*Copy the `token` string from the JSON response.*

**Step 2: Scan a package**: Replace `<YOUR_TOKEN>` with the token from Step 1:

```bash
curl -H "Authorization: Bearer <YOUR_TOKEN>" http://127.0.0.1:8000/scan/requests
```

## Dashboard

> [!NOTE]
> To run the dashboard locally, you need to have two terminals, one that runs the API,
> and one that runs the dashboard. See [API](#api) for instructions on how to run the
> API

To run the dashboard, execute the following in your terminal

```bash
uv run streamlit run dashboard.py
```

## Testing

We use pytest:

```bash
pytest tests/
```

## Deploy

Requires: `gcloud`

```bash
gcloud auth login
gcloud config set project $PROJECT_ID
gcloud services enable cloudbuild.googleapis.com run.googleapis.com
gcloud run deploy pyspam-api --source .
```

### Troubleshooting

**`Error: "BigQuery execution failed... Project [old-project-id] has been deleted"`**

Force a refresh of the local credentials for your current project:

```bash
gcloud auth application-default login
```

Make sure to sign in with the Google account associated with the active project.

## Model Context Protocol (MCP)

This repository includes a standalone MCP server (mcp_server/) that allows AI agents
(like Claude Desktop or Cursor) to natively "consult" the PySpam API before suggesting
packages.

### Quick Start (Requires [`pkgx`](https://pkgx.sh))

The server script is self-bootstrapping. It uses a shebang to automatically pull the
correct Python version and dependencies (mcp, httpx) via pkgx + uv. You will need to
make mcp_server/server.py executable:

```bash
chmod +x mcp_server/server.py
```

### Client Configuration

To use this with your AI editor, add the configuration below to your MCP Settings file.

- Cursor: Cmd+Shift+P > MCP: Open Settings File
- Claude Desktop: ~/Library/Application
Support/Claude/claude_desktop_config.json

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

Collection of repeatable tasks runnable via `xc`

### install

Installs the packages exposed by this repo

```bash
uv pip install -e .
```

### sync

Gets all the requirements you need for developing everything

Requires: install

```bash
uv sync --all-extras
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

### deploy

```bash
gcloud run deploy pyspam-api \
  --source . \
  --region us-west1 \
  --max-instances 1 \
  --allow-unauthenticated \
  --set-secrets="API_TOKEN_SECRET=pyspam-jwt-secret:latest" 
```

### api

```bash
uv run uvicorn api.main:app --reload
```