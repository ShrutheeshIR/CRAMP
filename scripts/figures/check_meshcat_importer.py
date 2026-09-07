#!/usr/bin/env python3
"""Verify the meshcat_html_importer Blender add-on this pipeline depends on.

    python3 scripts/figures/check_meshcat_importer.py            # static check
    python3 scripts/figures/check_meshcat_importer.py --probe    # also launch Blender

Unlike the sibling repos (iiwa-bimanual-augmented-jacobian-test,
rby1-constrained-planning), this project does NOT vendor the add-on under
`third_party/` and does not install it -- the add-on is already present at
`~/.config/blender/5.0/extensions/user_default/meshcat_html_importer`, so this
script verifies rather than installs. If it is ever missing on a new machine,
copy it from a sibling repo's `third_party/` or install upstream v0.1.3 from
https://github.com/nepfaff/drake-blender-recorder.

Why this exists at all: a missing or broken add-on does not fail loudly. The
render scripts import `build_scene_from_file` inside Blender; if that import
fails the scene is simply empty, Blender renders a blank transparent PNG, and
the process exits 0. That is indistinguishable from success at a glance, so
every driver runs `check()` before launching Blender.

--check is static (files on disk). --probe is the check that actually proves
something: it imports the add-on inside Blender's own interpreter, which is the
only place version skew or a broken dependency shows up.

Plain `python3` -- runs outside Blender and imports nothing but stdlib.
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from blender_paths import BLENDER, EXTENSIONS  # noqa: E402

ADDON_ID = "meshcat_html_importer"
EXPECTED_VERSION = "0.1.3"

TARGET = Path(EXTENSIONS) / ADDON_ID
MANIFEST = TARGET / "blender_manifest.toml"
# The entry point every render script imports. Checked by name rather than by
# importing it here, because this interpreter has no `bpy` and the module would
# fail to import outside Blender.
ENTRY_MODULE = TARGET / "blender_impl" / "scene_builder.py"
ENTRY_FUNCTION = "def build_scene_from_file"

# Run inside Blender by --probe. Mirrors exactly what blender_common.import_scene
# does, so a passing probe means the real import path works.
PROBE_SNIPPET = """
import sys
sys.path.insert(0, {extensions!r})
from meshcat_html_importer.blender_impl.scene_builder import build_scene_from_file
assert callable(build_scene_from_file)
print("ADDON_OK")
"""


def _manifest_field(name):
    """Read a top-level `name = "value"` string out of the add-on manifest.

    Hand-parsed rather than via tomllib so the failure mode on a malformed or
    truncated manifest is a clean None instead of an exception.
    """
    if not MANIFEST.is_file():
        return None
    for line in MANIFEST.read_text().splitlines():
        key, sep, value = line.partition("=")
        if sep and key.strip() == name:
            return value.strip().strip('"').strip("'")
    return None


def check(verbose=True, expect_version=EXPECTED_VERSION):
    """Static verification: the add-on is installed, is the expected version, and
    still exposes the entry point. Returns True/False, never raises."""
    problems = []

    if not TARGET.is_dir():
        problems.append(f"not installed: {TARGET}")
    else:
        addon_id = _manifest_field("id")
        version = _manifest_field("version")
        if addon_id != ADDON_ID:
            problems.append(f"manifest id is {addon_id!r}, expected {ADDON_ID!r}")
        if expect_version and version != expect_version:
            problems.append(f"version is {version!r}, expected {expect_version!r}")
        if not ENTRY_MODULE.is_file():
            problems.append(f"missing entry module: {ENTRY_MODULE}")
        elif ENTRY_FUNCTION not in ENTRY_MODULE.read_text():
            problems.append(f"{ENTRY_MODULE} no longer defines {ENTRY_FUNCTION!r}")

    if problems:
        if verbose:
            for problem in problems:
                print(f"[addon] {problem}")
            print("[addon] the render scripts will produce an EMPTY scene without it.")
            print(f"[addon] expected at: {TARGET}")
        return False

    if verbose:
        print(f"[addon] ok: {ADDON_ID} {_manifest_field('version')} at {TARGET}")
    return True


def probe(blender=BLENDER, timeout=180, verbose=True):
    """Import the add-on inside Blender itself and require ADDON_OK on stdout."""
    if not os.path.isfile(blender):
        if verbose:
            print(f"[addon] Blender not found at {blender}")
        return False

    snippet = PROBE_SNIPPET.format(extensions=EXTENSIONS)
    try:
        result = subprocess.run(
            [blender, "--background", "--python-expr", snippet],
            capture_output=True, text=True, timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        if verbose:
            print(f"[addon] probe timed out after {timeout}s")
        return False

    if "ADDON_OK" not in result.stdout:
        if verbose:
            # stderr first: Blender writes a wall of startup chatter to stdout
            # before anything diagnostic, so the useful line is usually here.
            print(f"[addon] probe FAILED (exit {result.returncode})")
            if result.stderr.strip():
                print(result.stderr.strip()[-4000:])
            if result.stdout.strip():
                print(result.stdout.strip()[-2000:])
        return False

    if verbose:
        print(f"[addon] probe ok: imported inside {blender}")
    return True


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true",
                    help="static check only (the default if no flag is given)")
    ap.add_argument("--probe", action="store_true",
                    help="also import the add-on inside Blender; the check that "
                         "actually catches version skew")
    ap.add_argument("--expect-version", default=EXPECTED_VERSION,
                    help=f"required add-on version (default: {EXPECTED_VERSION}); "
                         "pass an empty string to skip the version check")
    ap.add_argument("--blender", default=BLENDER, help="Blender executable for --probe")
    ap.add_argument("--timeout", type=int, default=180, help="--probe timeout in seconds")
    args = ap.parse_args()

    ok = check(expect_version=args.expect_version)
    if ok and args.probe:
        ok = probe(blender=args.blender, timeout=args.timeout)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
