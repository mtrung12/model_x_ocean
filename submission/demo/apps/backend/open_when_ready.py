"""Open the demo URL once the local backend is ready."""

from __future__ import annotations

import sys
import time
import urllib.request
import webbrowser


def main() -> int:
    url = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8010"
    timeout_s = float(sys.argv[2]) if len(sys.argv) > 2 else 90.0
    deadline = time.monotonic() + timeout_s

    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=2) as response:
                if 200 <= response.status < 500:
                    webbrowser.open(url)
                    return 0
        except OSError:
            pass
        time.sleep(0.75)

    webbrowser.open(url)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
