#!/usr/bin/env python3
"""Convert the FR3's COLLADA visual meshes to Wavefront OBJ, and patch the URDF.

    python3 scripts/figures/convert_visual_meshes.py            # convert + patch
    python3 scripts/figures/convert_visual_meshes.py --check    # verify only
    python3 scripts/figures/convert_visual_meshes.py --dry-run

Why this exists
---------------
Nine of the FR3's twelve visual meshes (link0 through link7, and hand) are
COLLADA. Drake's Meshcat publishes .dae perfectly well -- it is one of the three
formats it names -- but the meshcat_html_importer Blender add-on does not: its
`_create_from_mesh_file` handles only `gltf`, `glb` and `obj`, and returns
`(None, None)` for anything else. Nothing logs a complaint. The result is that
the arm imports into Blender as a floating marker, marker holder and tip, with
no arm attached, and the render looks like a deliberate composition rather than
a bug.

OBJ is the format both halves accept without argument, and the one already
proven in this exact scene: the marker, marker holder and tip are .obj and have
always landed in the right place. Drake publishes it, and the add-on imports it
through `bpy.ops.wm.obj_import`.

The two rejected alternatives, so they are not retried:

* Binary glTF (.glb) -- the add-on reads it, but Drake refuses to publish .glb
  from a URDF, which trades a missing arm in Blender for a missing arm
  everywhere.
* Text glTF (.gltf) -- both sides accept it, but glTF is a Y-up format and these
  meshes are Z-up. Exported through trimesh the axis conversion is applied twice,
  once on write and once by Blender's importer, and every link imports with its y
  and z extents swapped: link0 measures 0.226 x 0.189 x 0.140 m on disk and
  0.226 x 0.140 x 0.190 m in Blender. OBJ carries no axis convention and no such
  round trip.

Each link is merged into a single mesh and its per-part base colours are baked to
vertex colours, written inline as `v x y z r g b`. That keeps the FR3's white
shells distinct from its dark grey joint housings without a sidecar .mtl or
texture atlas, so each mesh stays exactly one file on disk as the .dae was.

The .dae files are left in place, so the change is reversible and nothing is
lost; the pre-patch URDF is saved beside the patched one.

Implementation note: this uses trimesh + pycollada rather than Blender, because
Blender 5.0 removed its COLLADA importer entirely (`bpy.ops.wm.collada_import`
no longer exists). trimesh also preserves the per-part base colours these files
carry -- the FR3's white shells against its dark grey joint housings -- which is
what makes the rendered arm look like an FR3 rather than a uniform blob.

This edits fr3_trajopt/models/fr3_marker/, which its own README documents as a
deliberate local copy carrying patches for Drake compatibility. This is another
such patch, and the README records it.
"""

import argparse
import os
import re
import shutil
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODEL_DIR = os.path.join(_REPO_ROOT, "fr3_trajopt", "models", "fr3_marker")
VISUAL_DIR = os.path.join(MODEL_DIR, "meshes", "visual")
URDF = os.path.join(MODEL_DIR, "fr3_expo_spherized.urdf")

# The meshes the URDF actually references from a <visual>. finger.dae is on disk
# but unreferenced -- the marker assembly replaces the gripper fingers -- so
# converting it would be wasted work.
DAE_NAMES = ("link0", "link1", "link2", "link3", "link4", "link5", "link6", "link7",
             "hand")


def dae_path(name):
    return os.path.join(VISUAL_DIR, f"{name}.dae")


def obj_path(name):
    return os.path.join(VISUAL_DIR, f"{name}.obj")


DEFAULT_PART_COLOR = (200, 200, 200, 255)


def convert_one(name):
    """Convert one .dae to a single OBJ, baking per-part colours to vertices."""
    import numpy as np
    import trimesh

    src, dst = dae_path(name), obj_path(name)
    if not os.path.isfile(src):
        raise SystemExit(f"[convert] missing source mesh: {src}")

    scene = trimesh.load(src, force="scene")
    # dump() applies each geometry's scene-graph transform; concatenating the raw
    # geometry dict instead would drop any part that is positioned by its node.
    parts = scene.dump()
    if not parts:
        raise SystemExit(f"[convert] {name}.dae contained no geometry")

    coloured = []
    for part in parts:
        part = part.copy()
        try:
            rgba = np.asarray(part.visual.material.baseColorFactor, dtype=np.uint8)
        except Exception:
            rgba = np.asarray(DEFAULT_PART_COLOR, dtype=np.uint8)
        if rgba.shape[0] == 3:
            rgba = np.append(rgba, 255)
        part.visual = trimesh.visual.ColorVisuals(
            part, vertex_colors=np.tile(rgba, (len(part.vertices), 1)))
        coloured.append(part)

    merged = trimesh.util.concatenate(coloured)
    if len(merged.faces) == 0:
        raise SystemExit(f"[convert] {name}.dae produced 0 faces")

    merged.export(dst)
    if not os.path.isfile(dst) or os.path.getsize(dst) == 0:
        raise SystemExit(f"[convert] failed to write {dst}")

    # The OBJ must be self-contained. trimesh emits a sidecar .mtl (and a texture
    # atlas) when a mesh carries material visuals rather than vertex colours;
    # catching that here stops a half-written mesh reaching Drake, which would
    # warn about an unopenable material library and then render it untextured.
    for stray in ("material.mtl", "material_0.png"):
        path = os.path.join(VISUAL_DIR, stray)
        if os.path.exists(path):
            os.remove(path)

    lo, hi = merged.bounds
    n_colors = len(np.unique(merged.visual.vertex_colors, axis=0))
    print(f"[convert] {name}.dae -> {name}.obj  "
          f"({os.path.getsize(dst) / 1e6:.2f} MB, {len(parts)} parts merged, "
          f"{len(merged.faces)} faces, {n_colors} colours, extent "
          f"{hi[0] - lo[0]:.3f}x{hi[1] - lo[1]:.3f}x{hi[2] - lo[2]:.3f} m)")


def patch_urdf(dry_run=False):
    """Point every <visual> mesh reference at the .obj beside its .dae.

    Only visual meshes are touched. Collision geometry is left exactly as it is:
    it is never rendered, and changing it would change what the planner and the
    collision checker see, which is not this script's business.
    """
    with open(URDF, "r") as f:
        text = f.read()

    changed = []
    for name in DAE_NAMES:
        new = f"meshes/visual/{name}.obj"
        for old in (f"meshes/visual/{name}.dae", f"meshes/visual/{name}.glb",
                    f"meshes/visual/{name}.gltf"):
            if old in text:
                text = text.replace(old, new)
                if name not in changed:
                    changed.append(name)

    if not changed:
        print("[patch] URDF already references .obj for every visual mesh")
        return False
    if dry_run:
        print(f"[patch] would rewrite {len(changed)} references: {', '.join(changed)}")
        return True

    backup = URDF + ".dae.bak"
    if not os.path.exists(backup):
        shutil.copy2(URDF, backup)
        print(f"[patch] saved the pre-patch URDF as {os.path.basename(backup)}")
    with open(URDF, "w") as f:
        f.write(text)
    print(f"[patch] rewrote {len(changed)} visual mesh references to .obj: "
          + ", ".join(changed))
    return True


def check(verbose=True):
    """Every referenced visual mesh exists and is in a format Meshcat publishes."""
    with open(URDF, "r") as f:
        text = f.read()

    refs = re.findall(r'filename="(meshes/visual/[^"]+)"', text)
    problems = []
    for ref in refs:
        path = os.path.join(MODEL_DIR, ref)
        if not os.path.isfile(path):
            problems.append(f"missing file: {ref}")
        elif ref.lower().endswith(".dae"):
            problems.append(f"{ref} is COLLADA. Drake publishes it, but the "
                            "meshcat_html_importer Blender add-on silently drops it, so "
                            "it will be missing from every render")
        elif ref.lower().endswith(".glb"):
            problems.append(f"{ref} is binary glTF, which Drake's Meshcat refuses to "
                            "publish from a URDF -- use .obj instead")
        elif ref.lower().endswith(".gltf"):
            problems.append(f"{ref} is text glTF, whose Y-up convention makes these Z-up "
                            "meshes import with y and z swapped -- use .obj instead")

    if verbose:
        if problems:
            for problem in problems:
                print(f"[check] {problem}")
            print("[check] run: python3 scripts/figures/convert_visual_meshes.py")
        else:
            print(f"[check] all {len(refs)} visual meshes present and publishable")
    return not problems


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true",
                    help="verify only; exit non-zero if any visual mesh is unusable")
    ap.add_argument("--dry-run", action="store_true", help="say what would change")
    ap.add_argument("--force", action="store_true", help="re-convert existing .obj files")
    args = ap.parse_args()

    if args.check:
        sys.exit(0 if check() else 1)

    todo = [n for n in DAE_NAMES if args.force or not os.path.isfile(obj_path(n))]
    if not todo:
        print(f"[convert] all {len(DAE_NAMES)} meshes already converted "
              "(pass --force to redo them)")
    elif args.dry_run:
        print(f"[convert] would convert: {', '.join(todo)}")
    else:
        print(f"[convert] converting {len(todo)} mesh(es) with trimesh")
        for name in todo:
            convert_one(name)

    patch_urdf(dry_run=args.dry_run)
    if not args.dry_run:
        sys.exit(0 if check() else 1)


if __name__ == "__main__":
    main()
