# PySpam

A tool for identifying spam packages on Python's registry.

## Installation

1. Install uv from [astral.sh/uv](https://astral.sh/uv)
2. Clone the repository
3. Run `uv sync` to install dependencies

## Training

All the training code is in [feature_engineering](./feature_engineering). 
To run it, just:

```sh
uv run pipeline.py 
```

To change the name of the input file, see [settings.py](./feature_engineering/settings.py).