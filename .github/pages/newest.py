# Copyright (c) 2022-2026, Yongchao Wu
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""Whether a minor version, such as 2.0, is at least as new as every release on
the branch gh-pages: exit status 0 if so, 1 if a newer one is there.

    python .github/pages/newest.py 2.0
"""

import json
import subprocess
import sys


def released(names):
    """The minor versions among the names of versions.json, as tuples."""
    versions = []
    for name in names:
        parts = name.split(".")
        if len(parts) == 2 and all(p.isdigit() for p in parts):
            versions.append(tuple(int(p) for p in parts))
    return versions


def main(version):
    mine = tuple(int(p) for p in version.split("."))
    shown = subprocess.run(
        ["git", "show", "gh-pages:versions.json"], capture_output=True, text=True
    )
    names = (
        [v["version"] for v in json.loads(shown.stdout)]
        if shown.returncode == 0
        else []
    )
    return 0 if all(mine >= other for other in released(names)) else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
