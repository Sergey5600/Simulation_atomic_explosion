"""Import the three cinematic simulations into one Blender scene, side by side:

    out/fireball_closeup -> "Fireball" (frames 1-240, the slow-motion close-up)
    out/mushroom_20kt    -> "Mushroom" (its ground dust fades out below 250 m)
    out/sand_20kt        -> "Sand"     (dust only, fades out above 250 m, no light)

They share world coordinates and frame times. For the close-up shot hide "Mushroom" (and
for the wide shots hide "Fireball"); "Sand" belongs in every shot. Missing folders are skipped.

    blender --python tools/blender/import_shots.py -- [--root out] [--preview-scene] [any
            import_explosion.py option, e.g. --glow 2 --detail 0.3]
"""

import os
import sys
import traceback

import bpy



def _script_dir():
    """Folder of this script. Blender does not always define __file__ for --python scripts,
    so fall back to the path given after --python on the command line."""
    path = globals().get("__file__")
    if not path and "--python" in sys.argv:
        path = sys.argv[sys.argv.index("--python") + 1]
    return os.path.dirname(os.path.abspath(path)) if path else os.getcwd()


sys.path.insert(0, _script_dir())
import import_explosion as ie  # noqa: E402

SHOTS = [
    ("fireball_closeup", "Fireball", []),
    ("mushroom_20kt", "Mushroom", ["--dust-above", "250"]),
    ("sand_20kt", "Sand", ["--dust-below", "250", "--no-light"]),
]


def main():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    root = "out"
    if "--root" in argv:
        i = argv.index("--root")
        root = argv[i + 1]
        argv = argv[:i] + argv[i + 2:]
    preview = "--preview-scene" in argv
    argv = [a for a in argv if a != "--preview-scene"]
    root = os.path.abspath(bpy.path.abspath(root))

    frames = 0
    imported, failed = [], []
    for folder, name, extra in SHOTS:
        path = os.path.join(root, folder)
        if not os.path.isfile(os.path.join(path, "manifest.json")):
            print(f"import_shots.py: skipping {path} (no manifest.json)")
            continue
        opts = ["--dir", path, "--name", name] + extra + argv
        if preview and name == "Mushroom":
            opts.append("--preview-scene")
        print(f"import_shots.py: importing {name} from {path}")
        try:
            frames = max(frames, ie.run(ie.parse_args(opts)))
            imported.append(name)
        except BaseException:  # keep going with the other shots, but say exactly what failed
            print(f"import_shots.py: FAILED to import {name}:")
            traceback.print_exc()
            failed.append(name)
    if failed:
        print(f"import_shots.py: could not import {', '.join(failed)} (see the errors above)")
    if not imported:
        print(f"import_shots.py: nothing imported from {root}")
        return
    bpy.context.scene.frame_end = frames
    print(f"import_shots.py: imported {', '.join(imported)}; frames 1-{frames}")


main()
