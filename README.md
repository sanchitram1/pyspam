# PySpam

[![Tests Passing](https://img.shields.io/badge/tests-54%20passing-green)
](https://github.com/sanchitram1/pyspam/actions)
[![Coverage](https://img.shields.io/badge/coverage-81%25-brightgreen)
](https://github.com/sanchitram1/pyspam)

A tool for identifying spam packages on Python's registry.

## Installation

1. Install uv from [astral.sh/uv](https://astral.sh/uv)
2. Clone the repository
3. Run `uv sync` to install dependencies
  - Alternatively, for all dependencies, including API ones,
run `uv sync --all-extras`

## Training

All the training code is in [feature_engineering](./feature_engineering) . To
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