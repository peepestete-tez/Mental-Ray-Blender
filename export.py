"""Blender depsgraph -> mentalray.core.Scene."""
import math

import bpy
import numpy as np

from .core import (Scene, Camera, Light, Material, Texture, Settings, Sky, sun_direction,
                   CONSTANT, LAMBERT, PHONG, BLINN)

_SHADERS = {'CONSTANT': CONSTANT, 'LAMBERT': LAMBERT, 'PHONG': PHONG, 'BLINN': BLINN}
_FALLOFF = {'NONE': 0, 'LINEAR': 1, 'QUADRATIC': 2}
_CONVERTIBLE = {'MESH', 'CURVE', 'SURFACE', 'META', 'FONT'}


class ExportError(Exception):
    pass


def lin_to_srgb(c):
    c = np.clip(np.asarray(c, dtype=np.float64), 0.0, None)
    return np.where(c <= 0.0031308, c * 12.92, 1.055 * np.power(c, 1.0 / 2.4) - 0.055)


def srgb_to_lin(c):
    c = np.clip(np.asarray(c, dtype=np.float64), 0.0, None)
    return np.where(c <= 0.04045, c / 12.92, np.power((c + 0.055) / 1.055, 2.4))


class _Ctx:
    def __init__(self, ms):
        self.ms = ms
        self.dr = ms.display_referred

    def color(self, rgb):
        """Blender (scene linear) colour -> value used for shading."""
        rgb = tuple(float(x) for x in rgb[:3])
        return tuple(lin_to_srgb(rgb)) if self.dr else rgb

    def gamma_color(self, rgb):
        """Colour from a COLOR_GAMMA property (already a display value)."""
        rgb = tuple(float(x) for x in rgb[:3])
        return rgb if self.dr else tuple(srgb_to_lin(rgb))


def _image_array(ctx, img):
    w, h = img.size
    ch = img.channels
    if w == 0 or h == 0:
        return None
    buf = np.empty(w * h * ch, dtype=np.float32)
    img.pixels.foreach_get(buf)
    buf = buf.reshape(h, w, ch)
    rgb = buf[..., :3] if ch >= 3 else np.repeat(buf[..., :1], 3, axis=2)
    rgb = rgb.astype(np.float64)
    # Image.pixels is scene-linear for sRGB images; convert back to the values
    # that were painted when running in display-referred mode.
    if ctx.dr and img.colorspace_settings.name != 'Non-Color':
        rgb = lin_to_srgb(rgb)
    return rgb


def _export_texture(ctx, t, image_cache):
    """MentalRayTexture property group -> core Texture (or None)."""
    if t.kind == 'NONE':
        return None
    g = ctx.gamma_color
    if t.kind == 'IMAGE':
        if t.image is None:
            return None
        key = t.image.name
        if key not in image_cache:
            image_cache[key] = _image_array(ctx, t.image)
        if image_cache[key] is None:
            return None
        return Texture("image", image=image_cache[key], mapping="UV", repeat_u=t.repeat_u, repeat_v=t.repeat_v)
    return Texture(
        kind="noise" if t.kind == 'NOISE' else "checker", mapping=t.mapping,
        color_a=g(t.color_a), color_b=g(t.color_b), repeat_u=t.repeat_u, repeat_v=t.repeat_v,
        scale=t.scale, noise_type=t.noise_type, octaves=t.octaves, lacunarity=t.lacunarity,
        gain=t.gain, distortion=t.distortion, seed=t.seed)


def _export_material(ctx, mat, image_cache):
    if mat is None:
        return Material()
    p = mat.mentalray
    g = ctx.gamma_color
    diffuse = ctx.color(mat.diffuse_color) if p.use_viewport_color else g(p.diffuse)
    bump = _export_texture(ctx, p.bump_tex, image_cache)
    return Material(
        name=mat.name, shader=_SHADERS[p.shader], diffuse=diffuse, ambient=g(p.ambient),
        specular=g(p.specular), shininess=p.shininess, reflectivity=p.reflectivity,
        transparency=p.transparency, ior=p.ior, emission=g(p.emission),
        texture=_export_texture(ctx, p.color_tex, image_cache), texture_mix=p.texture_mix,
        bump=bump, bump_strength=p.bump_strength if bump is not None else 0.0)


def _export_meshes(depsgraph, ctx, scene_out):
    mats = [Material()]            # index 0 = default
    mat_index = {}
    image_cache = {}
    P, N, UV, M, XF, TXF = [], [], [], [], [], []

    for inst in depsgraph.object_instances:
        ob = inst.object
        if ob.type not in _CONVERTIBLE:
            continue
        try:
            mesh = ob.to_mesh()
        except RuntimeError:
            mesh = None
        if mesh is None:
            continue
        try:
            mesh.calc_loop_triangles()
            nt = len(mesh.loop_triangles)
            if nt == 0:
                continue
            nv = len(mesh.vertices)
            co = np.empty(nv * 3, dtype=np.float32)
            mesh.vertices.foreach_get('co', co)
            co = co.reshape(nv, 3).astype(np.float64)
            vn = np.empty(nv * 3, dtype=np.float32)
            mesh.vertex_normals.foreach_get('vector', vn)
            vn = vn.reshape(nv, 3).astype(np.float64)

            tv = np.empty(nt * 3, dtype=np.int32)
            tl = np.empty(nt * 3, dtype=np.int32)
            tp = np.empty(nt, dtype=np.int32)
            tn = np.empty(nt * 3, dtype=np.float32)
            mesh.loop_triangles.foreach_get('vertices', tv)
            mesh.loop_triangles.foreach_get('loops', tl)
            mesh.loop_triangles.foreach_get('polygon_index', tp)
            mesh.loop_triangles.foreach_get('normal', tn)
            tv = tv.reshape(nt, 3); tl = tl.reshape(nt, 3); tn = tn.reshape(nt, 3).astype(np.float64)

            npoly = len(mesh.polygons)
            smooth = np.empty(npoly, dtype=bool)
            mesh.polygons.foreach_get('use_smooth', smooth)
            pmat = np.empty(npoly, dtype=np.int32)
            mesh.polygons.foreach_get('material_index', pmat)

            mw = np.array(inst.matrix_world, dtype=np.float64)
            R = mw[:3, :3]
            wco = co @ R.T + mw[:3, 3]
            nm = np.linalg.inv(R).T
            wvn = vn @ nm.T
            wtn = tn @ nm.T
            flip = np.linalg.det(R) < 0

            pos = wco[tv]                                       # (nt, 3, 3)
            tri_smooth = smooth[tp]
            nrm = np.where(tri_smooth[:, None, None], wvn[tv], np.repeat(wtn[:, None, :], 3, axis=1))
            nrm /= np.maximum(np.linalg.norm(nrm, axis=2, keepdims=True), 1e-20)

            uv = np.zeros((nt, 3, 2))
            if mesh.uv_layers.active is not None:
                nl = len(mesh.loops)
                uvd = np.empty(nl * 2, dtype=np.float32)
                mesh.uv_layers.active.data.foreach_get('uv', uvd)
                uv = uvd.reshape(nl, 2).astype(np.float64)[tl]

            if flip:                                            # mirrored transform reverses winding
                pos = pos[:, [0, 2, 1]]; nrm = nrm[:, [0, 2, 1]]; uv = uv[:, [0, 2, 1]]

            # materials
            slots = [s.material for s in ob.material_slots]
            slot_to_global = []
            for m in slots:
                key = m.name if m is not None else None
                if key is None:
                    slot_to_global.append(0)
                    continue
                if key not in mat_index:
                    mat_index[key] = len(mats)
                    mats.append(_export_material(ctx, m, image_cache))
                slot_to_global.append(mat_index[key])
            if slot_to_global:
                lut = np.array(slot_to_global, dtype=np.int64)
                tmat = lut[np.clip(pmat[tp], 0, len(lut) - 1)]
            else:
                tmat = np.zeros(nt, dtype=np.int64)

            try:
                w2o = np.linalg.inv(mw)[:3, :]
            except np.linalg.LinAlgError:
                w2o = np.hstack([np.eye(3), np.zeros((3, 1))])
            XF.append(w2o)
            TXF.append(np.full(nt, len(XF) - 1, dtype=np.int64))
            P.append(pos); N.append(nrm); UV.append(uv); M.append(tmat)
        finally:
            ob.to_mesh_clear()

    scene_out.materials = mats
    if P:
        scene_out.tri_pos = np.concatenate(P)
        scene_out.tri_nrm = np.concatenate(N)
        scene_out.tri_uv = np.concatenate(UV)
        scene_out.tri_mat = np.concatenate(M)
        scene_out.xforms = np.array(XF)
        scene_out.tri_xf = np.concatenate(TXF)


def _export_lights(depsgraph, ctx):
    out = []
    ms = ctx.ms
    for inst in depsgraph.object_instances:
        ob = inst.object
        if ob.type != 'LIGHT':
            continue
        li = ob.data
        mr = li.mentalray
        mw = np.array(inst.matrix_world, dtype=np.float64)
        pos = tuple(mw[:3, 3])
        axis_z = mw[:3, 2] / max(np.linalg.norm(mw[:3, 2]), 1e-20)
        direction = tuple(-axis_z)
        if li.type == 'SUN':
            scale = li.energy
        else:
            scale = li.energy / 1000.0 * ms.light_scale
        base = ctx.color(li.color)
        # In display-referred mode the colour is already a display value; the
        # intensity multiplies it directly, the way a period renderer would.
        color = tuple(c * scale for c in base)
        kw = dict(position=pos, direction=direction, color=color, shadows=mr.shadows,
                  falloff=_FALLOFF[mr.falloff], falloff_ref=mr.falloff_ref)
        if li.type == 'POINT':
            out.append(Light("point", **kw))
        elif li.type == 'SUN':
            out.append(Light("directional", **kw))
        elif li.type == 'SPOT':
            out.append(Light("spot", cone=li.spot_size * 0.5, blend=li.spot_blend, **kw))
        elif li.type == 'AREA':
            sx = li.size * 0.5
            sy = (li.size_y if li.shape in {'RECTANGLE', 'ELLIPSE'} else li.size) * 0.5
            ax = mw[:3, 0] * sx          # matrix columns already carry the object scale
            ay = mw[:3, 1] * sy
            out.append(Light("area", area_u=tuple(ax), area_v=tuple(ay),
                             area_samples=(mr.area_u, mr.area_v), **kw))
    return out


def _export_camera(depsgraph, width, height):
    scene = depsgraph.scene
    cam_ob = scene.camera
    if cam_ob is None:
        raise ExportError("Scene has no active camera")
    cam_ob = cam_ob.evaluated_get(depsgraph)
    cam = cam_ob.data
    mw = np.array(cam_ob.matrix_world, dtype=np.float64)
    right = mw[:3, 0] / np.linalg.norm(mw[:3, 0])
    up = mw[:3, 1] / np.linalg.norm(mw[:3, 1])
    fwd = -mw[:3, 2] / np.linalg.norm(mw[:3, 2])
    aspect = width / float(height)
    fit = cam.sensor_fit
    sensor = cam.sensor_height if fit == 'VERTICAL' else cam.sensor_width
    if fit == 'AUTO':
        fit = 'HORIZONTAL' if width >= height else 'VERTICAL'
    c = Camera(origin=tuple(mw[:3, 3]), right=tuple(right), up=tuple(up), forward=tuple(fwd))
    if cam.type == 'ORTHO':
        half = cam.ortho_scale * 0.5
        if fit == 'HORIZONTAL':
            c.ortho_x, c.ortho_y = half, half / aspect
        else:
            c.ortho_x, c.ortho_y = half * aspect, half
        c.ortho = True
    else:
        t = (sensor * 0.5) / cam.lens
        if fit == 'HORIZONTAL':
            c.tan_x, c.tan_y = t, t / aspect
        else:
            c.tan_x, c.tan_y = t * aspect, t
    # Blender's lens shift is relative to the fit dimension
    if fit == 'HORIZONTAL':
        c.shift_x = 2.0 * cam.shift_x
        c.shift_y = 2.0 * cam.shift_y * aspect
    else:
        c.shift_x = 2.0 * cam.shift_x / aspect
        c.shift_y = 2.0 * cam.shift_y
    return c


def _world_color(ctx, scene):
    w = scene.world
    if w is None:
        return (0.0, 0.0, 0.0)
    col = tuple(w.color)
    tree = getattr(w, "node_tree", None)      # use_nodes is deprecated in newer Blender; the tree is enough
    if tree is not None:
        for node in tree.nodes:
            if node.type == 'BACKGROUND':
                col = tuple(node.inputs['Color'].default_value)[:3]
                break
    return ctx.color(col)


def _export_sky(ctx, scene):
    w = scene.world
    if w is None:
        return None
    p = getattr(w, "mentalray", None)
    if p is None or not p.use_sky:
        return None
    if p.sun_object is not None:
        mw = np.array(p.sun_object.matrix_world, dtype=np.float64)
        axis_z = mw[:3, 2] / max(np.linalg.norm(mw[:3, 2]), 1e-20)
        sun = tuple(axis_z)                    # light points along -Z, so the sun is towards +Z
    else:
        sun = sun_direction(p.sun_azimuth, p.sun_elevation)
    g = ctx.gamma_color
    return Sky(
        zenith=g(p.zenith), horizon=g(p.horizon), ground=g(p.ground), gradient_power=p.gradient_power,
        sun_enable=p.sun_enable, sun_dir=sun, sun_color=g(p.sun_color), sun_size=math.radians(p.sun_size),
        sun_glow=p.sun_glow, clouds=p.clouds, cloud_coverage=p.cloud_coverage, cloud_scale=p.cloud_scale,
        cloud_softness=p.cloud_softness, cloud_color=g(p.cloud_color), cloud_shade=g(p.cloud_shade),
        cloud_seed=p.cloud_seed, stars=p.stars, star_density=p.star_density,
        star_brightness=p.star_brightness, visible_to_camera=p.visible_to_camera)


def export_scene(depsgraph, width, height):
    scene = depsgraph.scene
    ms = scene.mentalray
    ctx = _Ctx(ms)
    out = Scene()
    out.camera = _export_camera(depsgraph, width, height)
    out.background = _world_color(ctx, scene)
    out.sky = _export_sky(ctx, scene)
    out.ambient = ctx.color(ms.ambient)
    out.lights = _export_lights(depsgraph, ctx)
    _export_meshes(depsgraph, ctx, out)
    if len(out.tri_pos) == 0:
        raise ExportError("Scene contains no renderable geometry")
    st = Settings(
        width=width, height=height, min_samples=ms.min_samples, max_samples=ms.max_samples,
        contrast=tuple(ms.contrast), filter=ms.filter, filter_size=ms.filter_size,
        bucket_size=ms.bucket_size, trace_reflect=ms.trace_reflect, trace_refract=ms.trace_refract,
        trace_sum=ms.trace_sum, shadow_depth=ms.shadow_depth, clamp=ms.clamp,
        transparent_film=scene.render.film_transparent, fog=ms.fog_enable,
        fog_color=ctx.color(ms.fog_color), fog_density=ms.fog_density)
    return out, st
