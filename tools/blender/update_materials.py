"""Bring simulations already imported into an open .blend up to date, without re-importing:
rebuild their volume materials with the current import_explosion.py shader, reload the
grids from disk (e.g. after re-running the simulations into the same folders), restore the
velocity settings for motion blur and re-key the fireball light from the new manifest.

Everything else in the scene (camera, sun, world, render settings, other objects) is left
alone. Each old material is kept as "<name>_backup" (with a fake user, so it survives saving),
and numbers changed by hand in the old material carry over to the new one: unlinked inputs of
nodes with the same label (e.g. "Density", "Dust above", "Glow strength") and of the
Principled Volume node (anisotropy, absorption colour...). Pass --fresh to skip that.

Inside Blender: Scripting tab -> Text Editor -> Open this file -> Run Script, then save.
From a terminal:
    blender explosion.blend --python tools/blender/update_materials.py -- [--save] [--fresh]
            [any import_explosion.py option, e.g. --spots 0.8 --glow 1.5]

Objects: every volume object whose .vdb files sit next to a NukeSim manifest.json, whether it
was made by import_explosion.py / import_shots.py ("<prefix>_Explosion", with its light
"<prefix>_FireballLight") or imported by hand (File > Import > OpenVDB). The shot is told by
the folder: fireball_closeup -> Fireball, mushroom_20kt -> Mushroom, sand_20kt -> Sand.
"""

import os
import sys
import traceback

import bpy


def _script_dir():
    """Folder of this script: __file__, the path after --python, or the Text Editor's file."""
    path = globals().get("__file__")
    if not path or not os.path.isfile(path):
        path = None
        if "--python" in sys.argv:
            path = sys.argv[sys.argv.index("--python") + 1]
        else:
            for text in bpy.data.texts:
                if text.filepath and os.path.basename(bpy.path.abspath(text.filepath)) == "update_materials.py":
                    path = bpy.path.abspath(text.filepath)
                    break
    return os.path.dirname(os.path.abspath(path)) if path else os.getcwd()


sys.path.insert(0, _script_dir())
import import_explosion as ie  # noqa: E402

# Options given on the command line win over the hand-set values of the nodes they set.
OPTION_LABELS = {
    "--density": ["Density"],
    "--dust": ["Dust"],
    "--condensate": ["Cloud water"],
    "--emission": ["Emission"],
    "--glow": ["Glow strength"],
    "--dust-above": ["Dust above"],
    "--dust-below": ["Dust below"],
    "--detail-scale": ["Detail noise", "Spot noise"],
}

# Output folder -> shot prefix, as in import_shots.py.
FOLDER_PREFIX = {"fireball_closeup": "Fireball", "mushroom_20kt": "Mushroom", "sand_20kt": "Sand"}

# Same per-shot options as import_shots.py.
SHOT_OPTIONS = {
    "Mushroom": ["--dust-above", "250"],
    "Sand": ["--dust-below", "250", "--no-light"],
}


def unlinked_values(node):
    values = {}
    for i, sock in enumerate(node.inputs):
        if not sock.is_linked and hasattr(sock, "default_value"):
            try:
                values[i] = tuple(sock.default_value)
            except TypeError:
                values[i] = sock.default_value
    return values


def carry_over(old_nt, new_nt, skip_labels=()):
    """Copy hand-set numbers from the old node tree: labelled nodes and the Principled Volume."""
    def key(n):
        if n.label in skip_labels:
            return None
        if n.bl_idname == "ShaderNodeVolumePrincipled":
            return ("principled",)
        if n.bl_idname == "ShaderNodeValue" and n.name in ("StretchTolerance",):
            return ("value", n.name)
        return (n.bl_idname, n.label) if n.label else None

    old = {}
    for n in old_nt.nodes:
        k = key(n)
        if k is not None and k not in old:
            if n.bl_idname == "ShaderNodeValue":
                old[k] = {"out": n.outputs[0].default_value}
            else:
                old[k] = unlinked_values(n)
    copied = 0
    for n in new_nt.nodes:
        k = key(n)
        if k is None or k not in old:
            continue
        if n.bl_idname == "ShaderNodeValue":
            n.outputs[0].default_value = old[k]["out"]
            copied += 1
            continue
        for i, value in old[k].items():
            if i < len(n.inputs) and not n.inputs[i].is_linked and hasattr(n.inputs[i], "default_value"):
                try:
                    n.inputs[i].default_value = value
                    copied += 1
                except (TypeError, ValueError):
                    pass
    return copied


def rekey_light(light_obj, args, manifest):
    """Re-key position and power from the manifest, keeping the light's current peak power."""
    light = light_obj.data
    peak_energy = args.light_power
    for idblock in (light,):
        ad = idblock.animation_data
        if ad and ad.action:
            for fc in getattr(ad.action, "fcurves", []):
                if fc.data_path == "energy" and len(fc.keyframe_points):
                    peak_energy = max(k.co[1] for k in fc.keyframe_points) or peak_energy
    light_obj.animation_data_clear()
    light.animation_data_clear()
    frames = manifest["frames"]
    peak = max(f.get("light_power", 0.0) for f in frames) or 1.0
    for f in frames:
        n = f["number"]
        light_obj.location = f.get("light_position", (0.0, 0.0, 0.0))
        light_obj.keyframe_insert("location", frame=n)
        light.energy = peak_energy * f.get("light_power", 0.0) / peak
        light.keyframe_insert("energy", frame=n)


def sim_folder(obj):
    """Folder of the object's .vdb files when it holds a NukeSim manifest.json, else None."""
    if obj.type != "VOLUME" or not obj.data.filepath:
        return None
    folder = os.path.dirname(os.path.abspath(bpy.path.abspath(obj.data.filepath)))
    return folder if os.path.isfile(os.path.join(folder, "manifest.json")) else None


def shot_prefix(obj, folder):
    if obj.name.endswith("_Explosion"):
        return obj.name[: -len("_Explosion")]
    return FOLDER_PREFIX.get(os.path.basename(folder), os.path.basename(folder))


def update_object(obj, folder, extra, fresh):
    prefix = shot_prefix(obj, folder)
    vol = obj.data
    manifest = ie.load_manifest(folder)
    frames = manifest["frames"]
    args = ie.parse_args(["--dir", folder, "--name", prefix] + SHOT_OPTIONS.get(prefix, []) + extra)
    ie.set_names(prefix)

    # Grids: the sequence length may have changed, and files rewritten on disk must be re-read.
    # Where the sequence starts is left as set in the scene.
    vol.is_sequence = len(frames) > 1
    vol.frame_duration = len(frames)
    scene = bpy.context.scene
    scene.frame_set(scene.frame_current)
    vol.grids.unload()
    vol.grids.load()
    if vol.grids.error_message:
        print(f"update_materials.py: WARNING ({obj.name}): {vol.grids.error_message}")
    grids = [g.name for g in vol.grids]
    if hasattr(vol, "velocity_grid") and "velocity" in grids:
        vol.velocity_grid = "velocity"
        vol.velocity_unit = "SECOND"
        vol.velocity_scale = 1.0

    old = vol.materials[0] if len(vol.materials) and vol.materials[0] else None
    backup = None
    if old is not None:
        ie.MATERIAL_NAME = old.name
        backup = bpy.data.materials.get(old.name + "_backup")
        if backup is not None:
            bpy.data.materials.remove(backup)
        backup = old.copy()
        backup.name = old.name + "_backup"
        backup.use_fake_user = True
        if old.node_tree is not None:
            old.node_tree.animation_data_clear()  # stale RestWeight keys; the backup keeps its own copy
    mat = ie.build_material(args, float(manifest.get("world_scale", 1.0)), manifest.get("fields", []), manifest, grids)
    copied = 0
    if backup is not None and backup.node_tree is not None and not fresh:
        skip = [label for opt, labels in OPTION_LABELS.items() if opt in extra for label in labels]
        copied = carry_over(backup.node_tree, mat.node_tree, skip)
    if old is None:
        vol.materials.append(mat)

    light = bpy.data.objects.get(f"{prefix}_FireballLight")
    if light is not None and light.type == "LIGHT":
        rekey_light(light, args, manifest)
    print(f"update_materials.py: {obj.name}: {len(frames)} frames from {folder}; grids {grids}; "
          f"material rebuilt ({copied} hand-set values kept, old one saved as "
          f"{backup.name if backup else '-'})")
    return len(frames)


def main():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    save = "--save" in argv
    fresh = "--fresh" in argv
    extra = [a for a in argv if a not in ("--save", "--fresh")]
    objects = [(o, sim_folder(o)) for o in bpy.data.objects]
    objects = [(o, f) for o, f in objects if f]
    if not objects:
        print("update_materials.py: no volume objects reading NukeSim output (a folder with manifest.json) in this file")
        return
    frames = 0
    for obj, folder in objects:
        try:
            frames = max(frames, update_object(obj, folder, extra, fresh))
        except BaseException:  # keep going with the other objects, but say exactly what failed
            print(f"update_materials.py: FAILED to update {obj.name}:")
            traceback.print_exc()
    if frames:
        bpy.context.scene.frame_end = max(bpy.context.scene.frame_end, frames)
    if save and bpy.data.filepath:
        bpy.ops.wm.save_mainfile()
        print(f"update_materials.py: saved {bpy.data.filepath}")


main()
