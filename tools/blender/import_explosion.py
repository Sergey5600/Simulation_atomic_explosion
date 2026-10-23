"""Import a NukeSim OpenVDB sequence into Blender (tested with 5.2) and set up
the volume material and a light that follows the fireball.

From a terminal:
    blender --python tools/blender/import_explosion.py -- --dir out/test_fireball
    blender --python tools/blender/import_explosion.py -- --dir out/test_fireball --preview-scene

Inside Blender: open this file in the Text Editor, set DEFAULT_DIR below and
press "Run Script".

Options (after the "--"):
    --dir DIR            folder with manifest.json and the .vdb files
    --density X          smoke density multiplier (default 0.2)
    --dust X             dust density multiplier (default 0.2)
    --condensate X       cloud-water density multiplier (default 0.1)
    --emission X         fireball emission multiplier (default 0.05)
    --light-power W      peak power of the fireball point light in watts (default 5e9)
    --no-light           do not create the fireball light
    --preview-scene      also add a camera, sun, sky and desert ground plane
    --render-test PNG    render one frame to PNG and quit (for checking the setup)
    --frame N            frame used by --render-test (default: middle frame)
    --samples N          Cycles samples for --render-test (default 32)
"""

import argparse
import json
import os
import sys

import bpy

DEFAULT_DIR = ""  # used when the script is run from Blender's Text Editor

OBJECT_NAME = "NukeSim_Explosion"
MATERIAL_NAME = "NukeSim_Volume"
LIGHT_NAME = "NukeSim_FireballLight"


def parse_args():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    p = argparse.ArgumentParser(prog="import_explosion.py")
    p.add_argument("--dir", default=DEFAULT_DIR)
    p.add_argument("--density", type=float, default=0.2)
    p.add_argument("--dust", type=float, default=0.2)
    p.add_argument("--condensate", type=float, default=0.1)
    p.add_argument("--emission", type=float, default=0.05)
    p.add_argument("--light-power", type=float, default=5e9)
    p.add_argument("--no-light", action="store_true")
    p.add_argument("--preview-scene", action="store_true")
    p.add_argument("--render-test", default="")
    p.add_argument("--frame", type=int, default=0)
    p.add_argument("--samples", type=int, default=32)
    args = p.parse_args(argv)
    if not args.dir:
        raise SystemExit("import_explosion.py: pass --dir <output folder> (or set DEFAULT_DIR)")
    args.dir = os.path.abspath(bpy.path.abspath(args.dir))
    return args


def load_manifest(folder):
    path = os.path.join(folder, "manifest.json")
    if not os.path.isfile(path):
        raise SystemExit(f"import_explosion.py: {path} not found - run nukesim first")
    with open(path, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    if not manifest.get("frames"):
        raise SystemExit("import_explosion.py: manifest has no frames")
    return manifest


def ensure_node_tree(idblock):
    """Blender 5 always uses nodes and deprecates use_nodes; older versions need it switched on."""
    if getattr(idblock, "node_tree", None) is None:
        idblock.use_nodes = True
    return idblock.node_tree


def remove_existing(name):
    obj = bpy.data.objects.get(name)
    if obj is None:
        return
    data = obj.data
    bpy.data.objects.remove(obj, do_unlink=True)
    if data is not None and data.users == 0:
        if isinstance(data, bpy.types.Volume):
            bpy.data.volumes.remove(data)
        elif isinstance(data, bpy.types.Light):
            bpy.data.lights.remove(data)


def set_input(node, names, value):
    """Set the first existing input among `names` (socket names change between Blender versions)."""
    for name in names:
        sock = node.inputs.get(name)
        if sock is not None:
            sock.default_value = value
            return sock
    return None


def find_input(node, names):
    for name in names:
        sock = node.inputs.get(name)
        if sock is not None:
            return sock
    raise KeyError(f"{node.bl_idname} has none of the inputs {names}")


def import_volume(folder, manifest):
    frames = manifest["frames"]
    remove_existing(OBJECT_NAME)

    volume = bpy.data.volumes.new(OBJECT_NAME)
    volume.filepath = os.path.join(folder, frames[0]["file"])
    volume.is_sequence = len(frames) > 1
    volume.frame_start = 1
    volume.frame_duration = len(frames)
    volume.sequence_mode = "CLIP"

    obj = bpy.data.objects.new(OBJECT_NAME, volume)
    bpy.context.scene.collection.objects.link(obj)
    bpy.context.scene.frame_set(1)
    volume.grids.load()
    if volume.grids.error_message:
        print(f"import_explosion.py: WARNING: {volume.grids.error_message}")

    # Velocity is exported in world units per video second, already matching the slow motion.
    if hasattr(volume, "velocity_grid") and "velocity" in manifest.get("fields", []):
        volume.velocity_grid = "velocity"
        volume.velocity_unit = "SECOND"
        volume.velocity_scale = 1.0
    return obj


# Fireball gas is shock-heated air full of nitrogen dioxide: the pinkish-brown of real caps.
SMOKE_COLOR = (0.62, 0.36, 0.30, 1.0)
DUST_COLOR = (0.78, 0.66, 0.50, 1.0)  # desert sand
WATER_COLOR = (0.96, 0.97, 0.98, 1.0)  # cloud droplets scatter almost all light


def build_material(args, world_scale, fields):
    mat = bpy.data.materials.get(MATERIAL_NAME) or bpy.data.materials.new(MATERIAL_NAME)
    nt = ensure_node_tree(mat)
    nt.nodes.clear()

    out = nt.nodes.new("ShaderNodeOutputMaterial")
    out.location = (700, 0)
    pv = nt.nodes.new("ShaderNodeVolumePrincipled")
    pv.location = (400, 0)
    nt.links.new(pv.outputs[0], out.inputs["Volume"])

    def attribute(name, y):
        node = nt.nodes.new("ShaderNodeAttribute")
        node.attribute_type = "GEOMETRY"
        node.attribute_name = name
        node.location = (-400, y)
        return node

    def multiply(socket, factor, y, label):
        node = nt.nodes.new("ShaderNodeMath")
        node.operation = "MULTIPLY"
        node.label = label
        node.location = (-150, y)
        node.inputs[1].default_value = factor
        nt.links.new(socket, node.inputs[0])
        return node.outputs[0]

    # Grids are wired explicitly, so disable Principled Volume's own attribute lookups.
    set_input(pv, ["Density Attribute"], "")
    set_input(pv, ["Temperature Attribute"], "")
    set_input(pv, ["Color Attribute"], "")
    set_input(pv, ["Blackbody Intensity"], 0.0)

    def math(op, a, b, y, label=""):
        node = nt.nodes.new("ShaderNodeMath")
        node.operation = op
        node.label = label
        node.location = (50, y)
        for i, v in enumerate((a, b)):
            if isinstance(v, float):
                node.inputs[i].default_value = v
            else:
                nt.links.new(v, node.inputs[i])
        return node.outputs[0]

    # Exported values are per metre; convert to per Blender unit.
    dens = attribute("density", 300)
    smoke = multiply(dens.outputs["Fac"], args.density / world_scale, 300, "Density")

    def by_id(sockets, identifier):
        return next(sk for sk in sockets if sk.identifier == identifier)

    def mix_color(fac, a, b, y):
        mix = nt.nodes.new("ShaderNodeMix")
        mix.data_type = "RGBA"
        mix.location = (250, y)
        nt.links.new(fac, by_id(mix.inputs, "Factor_Float"))
        for ident, value in (("A_Color", a), ("B_Color", b)):
            if isinstance(value, tuple):
                by_id(mix.inputs, ident).default_value = value
            else:
                nt.links.new(value, by_id(mix.inputs, ident))
        return by_id(mix.outputs, "Result_Color")

    # Smoke, sand and cloud water share one volume: densities add, the colour is the
    # density-weighted mix of the three.
    total = smoke
    color = None
    if "dust" in fields:
        dust_attr = attribute("dust", 100)
        dust = multiply(dust_attr.outputs["Fac"], args.dust / world_scale, 100, "Dust")
        total = math("ADD", smoke, dust, 250, "Smoke + dust")
        frac = math("DIVIDE", dust, math("ADD", total, 1e-6, 150), 50, "Dust fraction")
        color = mix_color(frac, SMOKE_COLOR, DUST_COLOR, 150)
    if "condensate" in fields:
        cond_attr = attribute("condensate", -50)
        cond = multiply(cond_attr.outputs["Fac"], args.condensate / world_scale, -50, "Cloud water")
        dry = total
        total = math("ADD", dry, cond, 0, "Total density")
        frac = math("DIVIDE", cond, math("ADD", total, 1e-6, -100), -50, "Water fraction")
        color = mix_color(frac, color if color is not None else SMOKE_COLOR, WATER_COLOR, 0)
    nt.links.new(total, find_input(pv, ["Density"]))
    if color is not None:
        nt.links.new(color, find_input(pv, ["Color"]))
    else:
        set_input(pv, ["Color"], SMOKE_COLOR)
    temp = attribute("temperature", -100)
    bb = nt.nodes.new("ShaderNodeBlackbody")
    bb.location = (-150, -100)
    nt.links.new(temp.outputs["Fac"], bb.inputs["Temperature"])
    nt.links.new(bb.outputs[0], find_input(pv, ["Emission Color"]))

    flame = attribute("flame", -300)
    nt.links.new(multiply(flame.outputs["Fac"], args.emission / world_scale, -300, "Emission"),
                 find_input(pv, ["Emission Strength"]))
    return mat


def build_light(args, manifest):
    remove_existing(LIGHT_NAME)
    frames = manifest["frames"]
    peak = max(f.get("light_power", 0.0) for f in frames) or 1.0
    light = bpy.data.lights.new(LIGHT_NAME, type="POINT")
    light.color = (1.0, 0.78, 0.55)
    light.shadow_soft_size = 50.0 * manifest.get("world_scale", 1.0)
    obj = bpy.data.objects.new(LIGHT_NAME, light)
    bpy.context.scene.collection.objects.link(obj)
    for f in frames:
        n = f["number"]
        obj.location = f.get("light_position", (0.0, 0.0, 0.0))
        obj.keyframe_insert("location", frame=n)
        light.energy = args.light_power * f.get("light_power", 0.0) / peak
        light.keyframe_insert("energy", frame=n)
    return obj


def build_preview_scene(manifest):
    """Camera, sun, sky and a desert ground plane framing the cloud (its largest extent)."""
    scene = bpy.context.scene
    last = manifest["frames"][-1]
    lo = last.get("domain_min", [-500.0, -500.0, 0.0])
    hi = last.get("domain_max", [500.0, 500.0, 1000.0])
    clouds = [f for f in manifest["frames"] if "cloud_min" in f and f["cloud_max"][2] > f["cloud_min"][2]]
    if clouds:
        lo = [min(f["cloud_min"][a] for f in clouds) for a in range(3)]
        hi = [max(f["cloud_max"][a] for f in clouds) for a in range(3)]
    size = max(hi[0] - lo[0], hi[2] - lo[2])
    center = ((lo[0] + hi[0]) * 0.5, (lo[1] + hi[1]) * 0.5, (lo[2] + hi[2]) * 0.4)

    for name in ("NukeSim_Camera", "NukeSim_Sun", "NukeSim_Ground"):
        remove_existing(name)

    cam_data = bpy.data.cameras.new("NukeSim_Camera")
    cam_data.lens = 35.0
    cam_data.clip_end = size * 50.0
    cam = bpy.data.objects.new("NukeSim_Camera", cam_data)
    scene.collection.objects.link(cam)
    cam.location = (center[0], center[1] - 1.6 * size, size * 0.12)
    direction = (center[0] - cam.location[0], center[1] - cam.location[1], center[2] - cam.location[2])
    from mathutils import Vector
    cam.rotation_euler = Vector(direction).to_track_quat("-Z", "Y").to_euler()
    scene.camera = cam

    sun_data = bpy.data.lights.new("NukeSim_Sun", type="SUN")
    sun_data.energy = 4.0
    sun_data.angle = 0.01
    sun = bpy.data.objects.new("NukeSim_Sun", sun_data)
    sun.rotation_euler = (0.9, 0.0, 0.6)
    scene.collection.objects.link(sun)

    mesh = bpy.data.meshes.new("NukeSim_Ground")
    r = size * 20.0
    mesh.from_pydata([(-r, -r, 0), (r, -r, 0), (r, r, 0), (-r, r, 0)], [], [(0, 1, 2, 3)])
    ground = bpy.data.objects.new("NukeSim_Ground", mesh)
    scene.collection.objects.link(ground)
    gmat = bpy.data.materials.get("NukeSim_Desert") or bpy.data.materials.new("NukeSim_Desert")
    bsdf = ensure_node_tree(gmat).nodes.get("Principled BSDF")
    if bsdf is not None:
        bsdf.inputs["Base Color"].default_value = (0.55, 0.45, 0.33, 1.0)
        bsdf.inputs["Roughness"].default_value = 0.95
    mesh.materials.append(gmat)

    world = scene.world or bpy.data.worlds.new("NukeSim_World")
    scene.world = world
    bg = ensure_node_tree(world).nodes.get("Background")
    if bg is not None:
        bg.inputs["Color"].default_value = (0.45, 0.62, 0.8, 1.0)
        bg.inputs["Strength"].default_value = 0.8


def setup_scene_timing(manifest):
    scene = bpy.context.scene
    fps = float(manifest.get("fps", 24.0))
    scene.render.fps = max(1, round(fps))
    scene.render.fps_base = scene.render.fps / fps
    scene.frame_start = 1
    scene.frame_end = len(manifest["frames"])


def render_test(args, manifest):
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.cycles.device = "CPU"
    scene.cycles.samples = args.samples
    scene.render.resolution_x = 480
    scene.render.resolution_y = 360
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    frame = args.frame or (len(manifest["frames"]) + 1) // 2
    scene.frame_set(frame)
    scene.render.filepath = os.path.abspath(args.render_test)
    bpy.ops.render.render(write_still=True)
    print(f"import_explosion.py: rendered frame {frame} to {scene.render.filepath}")


def main():
    args = parse_args()
    manifest = load_manifest(args.dir)
    world_scale = float(manifest.get("world_scale", 1.0))

    setup_scene_timing(manifest)
    obj = import_volume(args.dir, manifest)
    obj.data.materials.clear()
    obj.data.materials.append(build_material(args, world_scale, manifest.get("fields", [])))
    if not args.no_light:
        build_light(args, manifest)
    if args.preview_scene or args.render_test:
        build_preview_scene(manifest)

    grids = [g.name for g in obj.data.grids]
    print(f"import_explosion.py: imported {len(manifest['frames'])} frames from {args.dir}; "
          f"grids in frame 1: {grids}")

    if args.render_test:
        render_test(args, manifest)


main()
