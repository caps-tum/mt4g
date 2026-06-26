#!/usr/bin/env python3
"""Embed scripts/plot_bandwidth.py into src/const/bandwidthChartScript.hpp.

The C++ binary ships the bandwidth plotting script as an embedded string (the
same idiom as src/const/chartScript.hpp) so that `mt4g -g` can run it from a
temporary file without depending on an external script path at runtime.

scripts/plot_bandwidth.py is the single source of truth; run this script after
editing it to refresh the generated header:

    python3 scripts/embed_plot_script.py
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "scripts" / "plot_bandwidth.py"
HEADER = ROOT / "src" / "const" / "bandwidthChartScript.hpp"


def main() -> int:
    script = SCRIPT.read_text()
    if ")PY" in script:
        raise SystemExit("plot_bandwidth.py contains the raw-string delimiter ')PY'")
    header = (
        "#pragma once\n"
        "// =====================================================================\n"
        "//  AUTO-GENERATED -- do not edit by hand.\n"
        "//  Mirror of scripts/plot_bandwidth.py (the canonical plotting script).\n"
        "//  Regenerate after changing that script with:\n"
        "//      python3 scripts/embed_plot_script.py\n"
        "//  Embedding the script (like const/chartScript.hpp) lets the binary run\n"
        "//  it from a temp file with no external file-path dependency at runtime.\n"
        "// =====================================================================\n"
        'inline const char* bandwidthChartScript = R"PY(\n' + script + ')PY";\n'
    )
    HEADER.write_text(header)
    print(f"wrote {HEADER} ({len(header)} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
