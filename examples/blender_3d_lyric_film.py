"""Example: a 3D lyric film built from timed.json, rendered in Blender.

Design (matched to the song's own measurements: 136 BPM, G major, bright 3kHz mix):
  - luminous cold glass, not dark-and-grim — the weight comes from space, not black
  - THE SEAM: a vertical line of light. Words straddle it early; as the song runs
    they are pushed further to either side. "Now we toe the line" made structural.
  - depth of field is driven by the alignment: focus distance is keyframed to the
    word being sung, so the rack focus resolves on the syllable
  - one warm beat: the light turns amber only for "I hope you know I'm sorry"

What this demonstrates: the timing data is the input to something real. Word
onsets drive the camera's focus distance, so the rack focus resolves on the
syllable — a shot that only works if the milliseconds are right.

Run:
  blender -b -P blender_3d_lyric_film.py -- timed.json out_dir 0 167 video 167
  ffmpeg -i out_dir/render.mp4 -i song.wav -c:v copy -c:a aac -shortest film.mp4

Roughly 2 s/frame at 1080p on a laptop CPU — a 3 minute song is ~100 minutes.
"""
import bpy, json, sys, math, random
from mathutils import Vector

argv = sys.argv[sys.argv.index("--") + 1:]
TIMED, OUTDIR = argv[0], argv[1]
T0, T1 = float(argv[2]), float(argv[3])
FPS = 30
SONG_LEN = float(argv[5]) if len(argv) > 5 else 180.0   # song length, for the seam's progress
random.seed(11)

words = [w for w in json.load(open(TIMED))["words"] if w["end"] > T0 and w["start"] < T1]
print(f"[scene] {len(words)} words in {T0}-{T1}s")

# ---------------------------------------------------------------- scene reset
bpy.ops.wm.read_factory_settings(use_empty=True)
try:      # Blender 5 stores fcurves in slotted actions; set interpolation up front
    bpy.context.preferences.edit.keyframe_new_interpolation_type = 'BEZIER'
    bpy.context.preferences.edit.keyframe_new_handle_type = 'AUTO_CLAMPED'
except Exception: pass
sc = bpy.context.scene
eng = {e.identifier for e in bpy.types.RenderSettings.bl_rna.properties['engine'].enum_items}
sc.render.engine = "BLENDER_EEVEE_NEXT" if "BLENDER_EEVEE_NEXT" in eng else "BLENDER_EEVEE"
sc.render.resolution_x, sc.render.resolution_y = 1920, 1080
sc.render.fps = FPS
sc.frame_start = 1
sc.frame_end = int((T1 - T0) * FPS)
ev = sc.eevee
for attr, val in [("taa_render_samples", 64), ("use_raytracing", True),
                  ("use_bloom", True), ("use_motion_blur", True)]:
    if hasattr(ev, attr):
        setattr(ev, attr, val)
if hasattr(sc.render, "use_motion_blur"):
    sc.render.use_motion_blur = True
sc.render.film_transparent = False
try:
    sc.view_settings.view_transform = "AgX"
    sc.view_settings.look = "AgX - Medium High Contrast"
except Exception:
    try: sc.view_settings.look = "Medium High Contrast"
    except Exception: pass

def f(t):                      # song-time -> frame in this clip
    return (t - T0) * FPS + 1

# ---------------------------------------------------------------- world: cold, high key
world = bpy.data.worlds.new("W"); sc.world = world
world.use_nodes = True
nt = world.node_tree
bg = nt.nodes["Background"]
grad = nt.nodes.new("ShaderNodeTexGradient"); grad.gradient_type = "LINEAR"
ramp = nt.nodes.new("ShaderNodeValToRGB")
ramp.color_ramp.elements[0].position = 0.28
ramp.color_ramp.elements[0].color = (0.004, 0.006, 0.012, 1)   # near black
ramp.color_ramp.elements[1].position = 0.92
ramp.color_ramp.elements[1].color = (0.055, 0.075, 0.115, 1)   # cold haze
mapg = nt.nodes.new("ShaderNodeMapping")
texc = nt.nodes.new("ShaderNodeTexCoord")
mapg.inputs["Rotation"].default_value = (0, 1.5708, 0)
nt.links.new(texc.outputs["Window"], mapg.inputs["Vector"])
nt.links.new(mapg.outputs["Vector"], grad.inputs["Vector"])
nt.links.new(grad.outputs["Color"], ramp.inputs["Fac"])
nt.links.new(ramp.outputs["Color"], bg.inputs[0])
bg.inputs[1].default_value = 1.0
sc.eevee.use_volumetric_shadows = True if hasattr(sc.eevee,"use_volumetric_shadows") else None
# warm beat: the world briefly turns amber for the apology
apology = next((w for w in words if w["text"].lower().startswith("sorry")), None)
if apology:
    a0, a1 = apology["start"] - 2.2, apology["end"] + 1.2
    warm = ramp.color_ramp.elements[1]
    for t, col in [(a0 - 1.0, (0.72, 0.80, 0.92, 1)),
                   (a0 + 0.6, (1.00, 0.66, 0.34, 1)),
                   (a1, (1.00, 0.66, 0.34, 1)),
                   (a1 + 1.6, (0.72, 0.80, 0.92, 1))]:
        warm.color = col
        warm.keyframe_insert("color", frame=f(t))
    print(f"[scene] warm beat at {a0:.1f}-{a1:.1f}s")

# ---------------------------------------------------------------- glass material
def glass_mat():
    m = bpy.data.materials.new("Glass"); m.use_nodes = True
    b = m.node_tree.nodes["Principled BSDF"]
    b.inputs["Base Color"].default_value = (0.78, 0.86, 0.96, 1)
    b.inputs["Roughness"].default_value = 0.02
    b.inputs["IOR"].default_value = 1.52
    hit = []
    for names, val in [(("Transmission Weight", "Transmission"), 1.0),
                       (("Metallic",), 0.0),
                       (("Coat Weight", "Clearcoat"), 1.0),
                       (("Coat Roughness", "Clearcoat Roughness"), 0.03),
                       (("Specular IOR Level", "Specular"), 0.9)]:
        for n in names:
            if n in b.inputs:
                b.inputs[n].default_value = val; hit.append(n); break
    print("[scene] glass inputs set:", ", ".join(hit))
    m.use_backface_culling = False
    for flag in ("use_screen_refraction", "use_raytrace_refraction"):
        if hasattr(m, flag):
            setattr(m, flag, True)
    if hasattr(m, "use_transparent_shadow"): m.use_transparent_shadow = True
    m.blend_method = "BLEND" if hasattr(m, "blend_method") else m.blend_method
    return m

GLASS = glass_mat()

def emit_mat(col, strength):
    m = bpy.data.materials.new("Emit"); m.use_nodes = True
    nt = m.node_tree; nt.nodes.clear()
    e = nt.nodes.new("ShaderNodeEmission"); o = nt.nodes.new("ShaderNodeOutputMaterial")
    e.inputs[0].default_value = col; e.inputs[1].default_value = strength
    nt.links.new(e.outputs[0], o.inputs[0])
    return m

# ---------------------------------------------------------------- the seam
seam = bpy.data.objects.new("seam", bpy.data.meshes.new("seam"))
bpy.ops.mesh.primitive_plane_add(size=1)
seam = bpy.context.object; seam.name = "seam"
seam.scale = (0.012, 1, 26)
seam.rotation_euler = (math.pi / 2, 0, 0)
seam.location = (0, 6.5, 0)
seam.data.materials.append(emit_mat((1.0, 0.98, 0.94, 1), 18.0))
seam.scale = (0.010, 1, 26)

# ---- atmosphere: light actually travels through this scene
bpy.ops.mesh.primitive_cube_add(size=1, location=(0, 3.0, 0))
vol = bpy.context.object; vol.name = "atmosphere"
vol.scale = (26, 16, 16)
vm = bpy.data.materials.new("Vol"); vm.use_nodes = True
vnt = vm.node_tree; vnt.nodes.clear()
vs = vnt.nodes.new("ShaderNodeVolumeScatter")
vs.inputs["Color"].default_value = (0.62, 0.74, 0.95, 1)
vs.inputs["Density"].default_value = 0.045
vs.inputs["Anisotropy"].default_value = 0.55
vout = vnt.nodes.new("ShaderNodeOutputMaterial")
vnt.links.new(vs.outputs[0], vout.inputs["Volume"])
vol.data.materials.append(vm)
vol.visible_shadow = False

# a hard light behind the seam -> shafts through the haze
bpy.ops.object.light_add(type="SPOT", location=(0, 9.5, 0.4))
shaft = bpy.context.object
shaft.data.energy = 7000; shaft.data.spot_size = math.radians(58)
shaft.data.spot_blend = 0.35; shaft.data.shadow_soft_size = 0.25
shaft.rotation_euler = (math.radians(-90), 0, 0)
shaft.data.color = (0.86, 0.92, 1.0)

# dust: small emissive motes, mostly out of focus -> bokeh
dust_mat = emit_mat((0.95, 0.97, 1.0, 1), 14.0)
bpy.ops.mesh.primitive_ico_sphere_add(subdivisions=1, radius=1)
proto = bpy.context.object; proto.name = "mote"
proto.data.materials.append(dust_mat)
proto.hide_render = True
for i in range(150):
    m = proto.copy(); m.data = proto.data
    r = random.uniform(0.012, 0.045)
    m.scale = (r, r, r)
    m.location = (random.uniform(-7.5, 7.5), random.uniform(-3.0, 9.0), random.uniform(-4.0, 4.0))
    m.hide_render = False
    bpy.context.collection.objects.link(m)

# ---------------------------------------------------------------- key light
bpy.ops.object.light_add(type="AREA", location=(-6, -7, 5))
key = bpy.context.object; key.data.energy = 1100; key.data.size = 5
key.rotation_euler = (math.radians(58), 0, math.radians(-38))
bpy.ops.object.light_add(type="AREA", location=(7, -5, -2))
rim = bpy.context.object; rim.data.energy = 900; rim.data.size = 4
rim.rotation_euler = (math.radians(75), 0, math.radians(52))

# ---------------------------------------------------------------- font
FONT = None
for p in ["/System/Library/Fonts/Supplemental/Futura.ttc",
          "/System/Library/Fonts/Supplemental/Helvetica.ttc",
          "/System/Library/Fonts/Supplemental/Georgia.ttf"]:
    try:
        FONT = bpy.data.fonts.load(p); print("[scene] font:", p); break
    except Exception:
        continue

# ---------------------------------------------------------------- words in space
def progress(t):
    return min(1.0, max(0.0, (t - 4.0) / (SONG_LEN - 8.0)))

objs = []
for i, w in enumerate(words):
    bpy.ops.object.text_add(location=(0, 0, 0))
    ob = bpy.context.object
    ob.data.body = w["text"]
    if FONT: ob.data.font = FONT
    ob.data.align_x = "CENTER"; ob.data.align_y = "CENTER"
    ob.data.extrude = 0.20
    ob.data.bevel_depth = 0.022
    ob.data.bevel_resolution = 4
    ob.data.size = 1.05
    ob.data.materials.append(GLASS)

    # the seam pushes words apart as the song runs
    pr = progress(w["start"])
    side = -1 if (i % 2 == 0) else 1
    x = side * (0.06 + pr * 0.80) + random.uniform(-0.10, 0.10)
    y = random.uniform(-1.6, 2.6)              # depth spread feeds the DOF
    z = random.uniform(-0.42, 0.42)
    ob.location = (x, y, z)
    ob.rotation_euler = (math.radians(90), 0, math.radians(random.uniform(-3.5, 3.5)))
    objs.append((ob, w))

    # appear on the sung onset, hold, then recede
    st, en = w["start"], max(w["end"], w["start"] + 0.30)
    ob.scale = (0.001, 0.001, 0.001)
    ob.keyframe_insert("scale", frame=max(1, f(st) - 3))
    ob.scale = (1, 1, 1)
    ob.keyframe_insert("scale", frame=f(st) + 1)
    ob.keyframe_insert("scale", frame=f(en))
    ob.scale = (0.001, 0.001, 0.001)
    ob.keyframe_insert("scale", frame=f(en) + 9)

# ---------------------------------------------------------------- camera + focus
bpy.ops.object.camera_add(location=(0, -7.4, 0.10), rotation=(math.radians(90), 0, 0))
cam = bpy.context.object; sc.camera = cam
cam.data.lens = 55
cam.data.dof.use_dof = True
cam.data.dof.aperture_fstop = 1.2          # shallow: only the sung word is sharp

# focus distance keyed to the word being sung — the alignment drives the lens
for ob, w in objs:
    d = (Vector(ob.location) - Vector(cam.location)).length
    cam.data.dof.keyframe_insert("focus_distance", frame=max(1, f(w["start"]) - 6))
    cam.data.dof.focus_distance = d
    cam.data.dof.keyframe_insert("focus_distance", frame=f(w["start"]) + 2)

# slow drift, beat-aware (136 BPM ≈ 0.441 s per beat)
BEAT = 60.0 / 136.0
t = T0
while t < T1:
    k = f(t)
    cam.location = (math.sin(t * 0.11) * 0.30, -7.4 + math.sin(t * 0.07) * 0.28,
                    0.10 + math.cos(t * 0.09) * 0.14)
    cam.rotation_euler = (math.radians(90) + math.sin(t * 0.05) * 0.008, 0,
                          math.sin(t * 0.06) * 0.010)
    cam.keyframe_insert("location", frame=k)
    cam.keyframe_insert("rotation_euler", frame=k)
    t += BEAT * 8

bpy.ops.wm.save_as_mainfile(filepath=f"{OUTDIR}/scene.blend")
if len(argv) > 4 and argv[4] == "video":
    sc.render.image_settings.file_format = "FFMPEG"
    sc.render.ffmpeg.format = "MPEG4"
    sc.render.ffmpeg.codec = "H264"
    sc.render.ffmpeg.constant_rate_factor = "HIGH"
    sc.render.ffmpeg.ffmpeg_preset = "GOOD"
    sc.render.filepath = f"{OUTDIR}/render.mp4"
else:
    sc.render.image_settings.file_format = "PNG"
    sc.render.filepath = f"{OUTDIR}/frames/"
print(f"[scene] saved · {sc.frame_end} frames to render")
