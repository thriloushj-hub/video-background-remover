"""Rebuild bench_payload/vbgr_code.zip from the working tree.

Why this exists: the zip used to be rebuilt by refreshing the *contents* of a
frozen 44-entry list, and that list was verified entry-by-entry by hash.  That
verification can only catch a file whose content drifted -- it is structurally
incapable of catching a file that should be in the zip and is not.  On
2026-08-31 the VM ran `pytest` and reported **118 passed** against 126 locally,
because `tests/test_composite_outputs.py` had been written after the list was
frozen and therefore never shipped.  No `vbgr/` source file was affected, so no
result was invalidated -- but only by luck.

This enumerates the tree instead, so a new file ships by existing.
"""
from __future__ import annotations

import hashlib
import os
import sys
import zipfile

TREES = ("vbgr", "bench", "tests")
ROOT_FILES_SUFFIX = (".py",)
EXTRA = {"v1_window_baselines.json": "bench/results/v1_window_baselines.json"}
SKIP_DIRS = {"__pycache__", "results", ".git"}
SKIP_NAMES = {"build_code_zip.py"}
DEST = "bench_payload/vbgr_code.zip"


def members(repo: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for tree in TREES:
        for root, dirs, files in os.walk(os.path.join(repo, tree)):
            dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
            for f in files:
                if not f.endswith((".py", ".txt")) or f in SKIP_NAMES:
                    continue
                full = os.path.join(root, f)
                out[os.path.relpath(full, repo).replace("\\", "/")] = full
    for f in os.listdir(repo):
        if f.endswith(ROOT_FILES_SUFFIX) and os.path.isfile(os.path.join(repo, f)):
            out[f] = os.path.join(repo, f)
    for name, rel in EXTRA.items():
        p = os.path.join(repo, rel)
        if os.path.exists(p):
            out[name] = p
    return out


def main(repo: str = ".") -> None:
    m = members(repo)
    dest = os.path.join(repo, DEST)
    before = set()
    if os.path.exists(dest):
        before = set(zipfile.ZipFile(dest).namelist())

    tmp = dest + ".tmp"
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as z:
        for name in sorted(m):
            z.write(m[name], name)

    # verify content AND membership
    z = zipfile.ZipFile(tmp)
    bad = [n for n in m
           if hashlib.sha256(z.read(n)).hexdigest()
           != hashlib.sha256(open(m[n], "rb").read()).hexdigest()]
    missing = sorted(set(m) - set(z.namelist()))
    if bad or missing:
        print("REFUSING TO REPLACE: mismatched", bad, "missing", missing)
        sys.exit(1)

    added = sorted(set(m) - before)
    dropped = sorted(before - set(m))
    os.replace(tmp, dest)
    print(f"{len(m)} entries, {os.path.getsize(dest)/1e6:.2f} MB")
    if added:
        print("  added:  " + ", ".join(added))
    if dropped:
        print("  dropped:" + ", ".join(dropped))
    if not added and not dropped:
        print("  membership unchanged")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else ".")
