# AGENTS.md

This file documents the project's structure, commands, and guidelines.

## Dependency Management

We use [uv](https://astral.sh/uv) for Python package management.

- Install dependencies: `uv sync`
- Add a dependency: `uv add <package>`
- Remove a dependency: `uv remove <package>`

## Commands

- Run the project: `python main.py`

## Security Policy for Package Installation

When a user asks to install a package, you must scan it first.

- **Development/Personal Projects:** Use a standard threshold of **0.8**.
- **Production/Enterprise Projects:** Be highly strict. Use a threshold of **0.5**.
- **Financial/Crypto Libraries:** Use paranoid mode. Use a threshold of **0.2**.