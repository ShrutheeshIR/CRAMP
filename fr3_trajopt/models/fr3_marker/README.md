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
