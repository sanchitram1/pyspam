#!/usr/bin/env uv run
"""
Fetch and extract spam package names from multiple sources.

This script combines package names from various malware/spam datasets:
- lxyeternal's pypi_malregistry: A research project cataloging spam packages
- DataDog's malicious-software-packages-dataset: Known malicious packages
"""

import json
import re
import sys
import tarfile
from io import BytesIO
from urllib.request import urlopen

SOURCES = {
    "malregistry": {
        "url": "https://github.com/lxyeternal/pypi_malregistry/archive/refs/heads/main.tar.gz",
        "type": "tar.gz",
        "description": "lxyeternal's pypi_malregistry - research project on PyPI spam packages",
    },
    "datadog": {
        "url": "https://raw.githubusercontent.com/DataDog/malicious-software-packages-dataset/main/samples/pypi/manifest.json",
        "type": "json",
        "description": "DataDog malicious-software-packages-dataset - known malicious PyPI packages",
    },
}


def extract_package_from_tarname(filename):
    """
    Extract package name from tar.gz filename like 'pkg-name-version.tar.gz'.

    Uses regex to match everything before the last hyphen followed by a version number.
    Version format: digits with optional dots and more digits.
    """
    match = re.match(r"^(.+)-\d+(\.\d+)*\.tar\.gz$", filename)
    return match.group(1) if match else None


def fetch_malregistry():
    """Fetch package names from lxyeternal's malregistry tar.gz."""
    packages: set[str] = set()
    url = SOURCES["malregistry"]["url"]

    with urlopen(url) as response:
        tar = tarfile.open(fileobj=BytesIO(response.read()), mode="r:gz")

        for member in tar.getmembers():
            # Skip directories and the root directory
            if member.isdir() or member.name.endswith("/"):
                continue

            # Skip the metadata directory itself
            if "pypi_malregistry" in member.name.split("/")[0]:
                filename = member.name.split("/")[-1]
                if filename and not filename.startswith("pypi_malregistry"):
                    pkg_name = extract_package_from_tarname(filename)
                    if pkg_name:
                        packages.add(pkg_name)

    return packages


def fetch_datadog():
    """Fetch package names from DataDog's manifest.json."""
    packages = set()
    url = SOURCES["datadog"]["url"]

    with urlopen(url) as response:
        manifest = json.loads(response.read().decode("utf-8"))
        # manifest is a dict with package names as keys
        packages.update(manifest.keys())

    return packages


def main():
    """Fetch from all sources and print deduplicated package names."""
    all_packages = set()

    try:
        print("Fetching from malregistry...", file=sys.stderr)
        all_packages.update(fetch_malregistry())

        print("Fetching from DataDog...", file=sys.stderr)
        all_packages.update(fetch_datadog())

        print("Total unique packages:", len(all_packages), file=sys.stderr)

        # Print to stdout
        for package in sorted(all_packages):
            print(package)

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
