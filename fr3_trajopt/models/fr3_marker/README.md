# fr3_marker (local copy)

`fr3_expo_spherized.urdf` is a copy of
`cricket/resources/fr3_marker/fr3_expo_spherized.urdf` with two bugs patched
so Drake's `Parser` can load it:

1. The upstream file references `meshes/visual/hand.obj`, which does not
   exist on disk (only `meshes/visual/hand.dae` does). Patched to reference
   the `.dae` file.
2. The `fr3_marker` and `fr3_tip` links use `<material name="marker_black">`
   without ever defining that material's color (unlike `panda_white` /
   `panda_red`, which are defined inline at first use elsewhere in the
   file). Drake requires every material to carry color/texture info, so
   `marker_black` is patched to `<color rgba="0.05 0.05 0.05 1.0">`.

`meshes/` here is a real copy of `cricket/resources/fr3_marker/meshes/`
(~12 MB), not a symlink -- this directory must be self-contained since it's
run inside containers that only mount `fr3_trajopt/`, not `cricket/`.

If the upstream URDF or meshes change in `cricket/`, re-copy them here and
re-apply the patches.

## Visual meshes converted to OBJ

The nine COLLADA visual meshes (`link0`--`link7`, `hand`) were converted to
Wavefront OBJ and the URDF's `<visual>` references repointed at them, by
`scripts/figures/convert_visual_meshes.py`. The `.dae` originals are kept, and
the pre-conversion URDF is saved as `fr3_expo_spherized.urdf.dae.bak`.

The reason is not Drake: Drake's Meshcat publishes `.dae` without complaint. It
is the `meshcat_html_importer` Blender add-on used by the figure pipeline, whose
`_create_from_mesh_file` handles only `gltf`, `glb` and `obj` and silently
returns nothing for anything else. With `.dae` the arm imports into Blender as a
floating marker, marker holder and tip with no arm attached, and nothing logs a
warning.

OBJ rather than glTF: `.glb` is refused by Drake's Meshcat when referenced from a
URDF, and text `.gltf` is a Y-up format, so these Z-up meshes round-trip through
two axis conversions and import with their y and z extents swapped (`link0`
measures 0.226 x 0.189 x 0.140 m on disk but 0.226 x 0.140 x 0.190 m in Blender).
OBJ carries no axis convention, and the marker/holder/tip were always OBJ and
have always landed correctly.

Each link's parts are merged into one mesh with its per-part base colours baked
to vertex colours, written inline as `v x y z r g b`. That keeps the white shells
distinct from the dark grey joint housings with no sidecar `.mtl` or texture, so
each mesh stays a single file as the `.dae` was.

Re-run `python3 scripts/figures/convert_visual_meshes.py --check` after any
upstream re-copy; it exits non-zero if a referenced mesh is missing or is in a
format that will silently disappear.
