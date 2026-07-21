#!/usr/bin/env python3
"""Open a URL with the system default browser in a cross-platform way."""

from __future__ import annotations

import sys
import webbrowser


def main() -> int:
    if len(sys.argv) != 2:
        print("Usage: python scripts/open_url.py <URL>")
        return 1

    url = sys.argv[1]
    opened = webbrowser.open(url)
    if not opened:
        print(f"Could not open browser automatically. Please open: {url}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
