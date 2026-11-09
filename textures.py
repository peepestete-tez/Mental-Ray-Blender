"""Procedural and image textures for colour and bump mapping.

Noise textures are *solid* (3D) like the classic 90s ones: they are evaluated at the surface point,
so no UV unwrap is needed and they stay attached to the object ('OBJECT' mapping) or sit fixed in
space ('WORLD'). 'UV' mapping evaluates them on the UV coordinates instead.
"""
import math

import numpy as np

_MASK = 0xFFFFFFFF
_GRADS = np.array([[1, 1, 0], [-1, 1, 0], [1, -1, 0], [-1, -1, 0],
                   [1, 0, 1], [-1, 0, 1], [1, 0, -1], [-1, 0, -1],
                   [0, 1, 1], [0, -1, 1], [0, 1, -1], [0, -1, -1]], dtype=np.float64)


# ------------------------------------------------------------------ noise primitives
def _hash3(ix, iy, iz, seed):
    h = (ix.astype(np.int64) * 374761393 + iy.astype(np.int64) * 668265263
         + iz.astype(np.int64) * 2147483629 + int(seed) * 362437) & _MASK
    h = ((h ^ (h >> 13)) * 1274126177) & _MASK
    return h ^ (h >> 16)


def _fade(t):
    return t * t * t * (t * (t * 6.0 - 15.0) + 10.0)


def perlin3(q, seed=0):
    """Gradient noise, roughly in [-1, 1]. q: (n, 3)."""
    x, y, z = q[:, 0], q[:, 1], q[:, 2]
    xi, yi, zi = np.floor(x), np.floor(y), np.floor(z)
    fx, fy, fz = x - xi, y - yi, z - zi
    u, v, w = _fade(fx), _fade(fy), _fade(fz)

    def corner(dx, dy, dz):
        g = _GRADS[_hash3(xi + dx, yi + dy, zi + dz, seed) % 12]
        return g[:, 0] * (fx - dx) + g[:, 1] * (fy - dy) + g[:, 2] * (fz - dz)

    c000, c100 = corner(0, 0, 0), corner(1, 0, 0)
    c010, c110 = corner(0, 1, 0), corner(1, 1, 0)
    c001, c101 = corner(0, 0, 1), corner(1, 0, 1)
    c011, c111 = corner(0, 1, 1), corner(1, 1, 1)
    x00 = c000 + (c100 - c000) * u
    x10 = c010 + (c110 - c010) * u
    x01 = c001 + (c101 - c001) * u
    x11 = c011 + (c111 - c011) * u
    y0 = x00 + (x10 - x00) * v
    y1 = x01 + (x11 - x01) * v
    return np.clip(y0 + (y1 - y0) * w, -1.0, 1.0)


def worley3(q, seed=0):
    """Distance to the nearest feature point (F1), roughly 0..1. q: (n, 3)."""
    xi, yi, zi = np.floor(q[:, 0]), np.floor(q[:, 1]), np.floor(q[:, 2])
    fx, fy, fz = q[:, 0] - xi, q[:, 1] - yi, q[:, 2] - zi
    best = np.full(len(q), 1e9)
    for dz in (-1, 0, 1):
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                h = _hash3(xi + dx, yi + dy, zi + dz, seed + 101)
                jx = (h & 0x3FF) / 1023.0
                jy = ((h >> 10) & 0x3FF) / 1023.0
                jz = ((h >> 20) & 0x3FF) / 1023.0
                d = (dx + jx - fx) ** 2 + (dy + jy - fy) ** 2 + (dz + jz - fz) ** 2
                best = np.minimum(best, d)
    return np.sqrt(best)


def _fbm(q, tex):
    """Signed fractal sum, about [-1, 1]."""
    total = np.zeros(len(q))
    amp, norm, freq = 1.0, 0.0, 1.0
    for o in range(max(1, int(tex.octaves))):
        total += amp * perlin3(q * freq, tex.seed + o * 31)
        norm += amp
        amp *= tex.gain
        freq *= tex.lacunarity
    return np.clip(total / norm * 1.5, -1.0, 1.0)


def _turb(q, tex):
    """Sum of |noise|, 0..1."""
    total = np.zeros(len(q))
    amp, norm, freq = 1.0, 0.0, 1.0
    for o in range(max(1, int(tex.octaves))):
        total += amp * np.abs(perlin3(q * freq, tex.seed + o * 31))
        norm += amp
        amp *= tex.gain
        freq *= tex.lacunarity
    return np.clip(total / norm * 2.2, 0.0, 1.0)


def noise_height(tex, q):
    """Scalar 0..1 for a noise texture at texture-space points q (n, 3)."""
    t = tex.noise_type
    if t == "TURBULENCE":
        return _turb(q, tex)
    if t == "MARBLE":
        return 0.5 + 0.5 * np.sin((q[:, 0] + tex.distortion * 0.25 * _turb(q, tex)) * math.pi)
    if t == "WOOD":
        r = np.sqrt(q[:, 0] ** 2 + q[:, 1] ** 2) + tex.distortion * 0.1 * _turb(q, tex)
        f = r - np.floor(r)
        return f * f * (3.0 - 2.0 * f)
    if t == "CELLS":
        return 1.0 - np.clip(worley3(q, tex.seed), 0.0, 1.0)
    return np.clip(0.5 + 0.5 * _fbm(q, tex), 0.0, 1.0)          # FRACTAL


# ------------------------------------------------------------------ coordinates and sampling
def _is_uv(tex):
    return tex.kind == "image" or tex.mapping == "UV"


def tex_coords(tex, Pw, W2O, uv):
    """Texture-space coordinates (n, 3) for noise / solid checker."""
    if tex.mapping == "UV":
        q = np.stack([uv[:, 0], uv[:, 1], np.zeros(len(uv))], axis=1)
    elif tex.mapping == "OBJECT" and W2O is not None:
        q = np.einsum('nij,nj->ni', W2O[:, :, :3], Pw) + W2O[:, :, 3]
    else:
        q = Pw
    return q * tex.scale


def _sample_image(img, u, v):
    H, W = img.shape[:2]
    x = (u % 1.0) * W - 0.5
    y = (v % 1.0) * H - 0.5
    x0 = np.floor(x).astype(np.int64); y0 = np.floor(y).astype(np.int64)
    fx = (x - x0)[:, None]; fy = (y - y0)[:, None]
    x0w, x1w = x0 % W, (x0 + 1) % W
    y0w, y1w = y0 % H, (y0 + 1) % H
    c00 = img[y0w, x0w, :3]; c10 = img[y0w, x1w, :3]
    c01 = img[y1w, x0w, :3]; c11 = img[y1w, x1w, :3]
    return (c00 * (1 - fx) + c10 * fx) * (1 - fy) + (c01 * (1 - fx) + c11 * fx) * fy


def _checker_parity(tex, Pw, W2O, uv):
    if tex.mapping == "UV":
        s = np.floor(uv[:, 0] * tex.repeat_u) + np.floor(uv[:, 1] * tex.repeat_v)
    else:
        q = tex_coords(tex, Pw, W2O, uv)
        s = np.floor(q[:, 0]) + np.floor(q[:, 1]) + np.floor(q[:, 2])
    return (s.astype(np.int64) & 1) == 0


def tex_height(tex, Pw, W2O, uv):
    """Scalar height 0..1 used for bump mapping."""
    if tex.kind == "noise":
        return noise_height(tex, tex_coords(tex, Pw, W2O, uv))
    if tex.kind == "checker":
        return _checker_parity(tex, Pw, W2O, uv).astype(np.float64)
    rgb = _sample_image(tex.image, uv[:, 0] * tex.repeat_u, uv[:, 1] * tex.repeat_v)
    return rgb @ np.array([0.299, 0.587, 0.114])


def tex_color(tex, Pw, W2O, uv):
    """(n, 3) colour for diffuse mapping."""
    a = np.asarray(tex.color_a, dtype=np.float64)
    b = np.asarray(tex.color_b, dtype=np.float64)
    if tex.kind == "image":
        return _sample_image(tex.image, uv[:, 0] * tex.repeat_u, uv[:, 1] * tex.repeat_v)
    if tex.kind == "checker":
        return np.where(_checker_parity(tex, Pw, W2O, uv)[:, None], a[None, :], b[None, :])
    h = noise_height(tex, tex_coords(tex, Pw, W2O, uv))
    return a[None, :] + (b - a)[None, :] * h[:, None]


def bump_step(tex):
    """Finite-difference step for bump mapping: (in_uv_units, step).

    UV-mapped textures step in UV space; solid textures step in world units. The step is a small
    fraction of the finest feature so the slope is local."""
    if tex.kind == "image":
        n = max(tex.image.shape[:2]) if tex.image is not None else 256
        return True, 0.5 / (n * max(tex.repeat_u, tex.repeat_v, 1e-6))
    if tex.mapping == "UV":
        f = max(tex.repeat_u, tex.repeat_v, 1e-6) if tex.kind == "checker" else max(tex.scale, 1e-6)
        return True, 0.02 / f
    return False, 0.02 / max(tex.scale, 1e-6)


def bump_norm(tex):
    """(norm, per_uv) for bump strength.

    Height is measured in units of the texture's feature size so a given Bump Strength looks the same at any
    texture scale. `per_uv` is True when the feature is defined in UV space (UV-mapped textures); the caller then
    also multiplies by the world length of one UV unit, so the look does not depend on how large the mesh is
    in UV terms. Plain images have no natural feature size: raw height in world units (norm 1, per_uv False)."""
    if tex.kind == "image":
        if getattr(tex, "feature_uv", 0.0) > 0.0:
            return tex.feature_uv, True
        return 1.0, False
    if tex.mapping == "UV":
        if tex.kind == "checker":
            return 1.0 / max(tex.repeat_u, tex.repeat_v, 1e-6), True
        return 1.0 / max(tex.scale, 1e-6), True
    return 1.0 / max(tex.scale, 1e-6), False
