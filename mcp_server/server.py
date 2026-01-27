#!/usr/bin/env -S pkgx +python@3.13 uv run --with mcp --with httpx

import httpx
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("PySpam Guard", dependencies=["httpx"])

API_URL = "https://pyspam-api-605094789848.us-west1.run.app"
DEFAULT_THRESHOLD = 0.8  # can be configured by AGENTS.MD


@mcp.tool()
async def scan_package(package_name: str, threshold: float = DEFAULT_THRESHOLD) -> str:
    """
    Scans a Python package name against the PySpam detection model.

    Use this tool when a user asks if a package is safe, or before
    installing a package.

    Args:
        package_name: The name of the package on PyPI (e.g., 'requests').
    """
    endpoint = f"{API_URL}/scan/{package_name}"

    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(endpoint, timeout=10.0)
            response.raise_for_status()
            data = response.json()

            # Parse results
            score = data.get("prediction", 0.0)

            # We can return raw JSON strings, or a formatted string.
            # Formatted strings often help the LLM "reason" better about the result.
            status = "DANGEROUS" if score > threshold else "SAFE"

            return (
                f"Analysis for '{package_name}':\n"
                f"- Safety Status: {status}\n"
                f"- Spam Probability: {score:.2%}\n"
                f"- Raw API Response: {data}"
            )

        except httpx.HTTPStatusError as e:
            return f"API Error: The server returned {e.response.status_code} for package '{package_name}'."
        except httpx.RequestError as e:
            return (
                f"Connection Error: Could not reach the PySpam API. Details: {str(e)}"
            )


@mcp.tool()
async def generate_key() -> str:
    """
    Generates an API key from the PySpam API.
    
    Use this tool to request a new API key for authentication.
    """
    endpoint = f"{API_URL}/generate-key"

    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(endpoint, timeout=10.0)
            response.raise_for_status()
            data = response.json()

            api_key = data.get("api_key", "")
            
            return (
                f"API Key Generated Successfully:\n"
                f"- API Key: {api_key}\n"
                f"- Full Response: {data}"
            )

        except httpx.HTTPStatusError as e:
            return f"API Error: The server returned {e.response.status_code} when generating API key."
        except httpx.RequestError as e:
            return (
                f"Connection Error: Could not reach the PySpam API. Details: {str(e)}"
            )


if __name__ == "__main__":
    mcp.run()
