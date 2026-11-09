"""Bucket renderer with mental ray style adaptive supersampling.

Sample levels follow mental ray's convention: level n >= 0 is 4**n samples per
pixel (2**n x 2**n), negative levels are one sample per 2**|n| x 2**|n| block of
pixels. Rendering starts at the minimum level and subdivides any cell whose
colour differs from a neighbouring cell by more than the per-channel contrast
threshold, up to the maximum level. Samples are then reconstructed with the
selected filter (box / triangle / gaussian / mitchell / lanczos).
"""
import math
import numpy as np

from .tracer import Tracer


# ---------------------------------------------------------------- filters
# Each takes u in [-1, 1] (offset divided by the filter radius).
def _box(u):
    return (np.abs(u) < 1.0).astype(np.float64)


def _triangle(u):
    return np.clip(1.0 - np.abs(u), 0.0, None)


def _gauss(u):
    return np.where(np.abs(u) < 1.0, np.exp(-4.5 * u * u), 0.0)


def _mitchell(u, B=1.0 / 3.0, C=1.0 / 3.0):
    x = np.abs(u) * 2.0
    x2, x3 = x * x, x * x * x
    near = ((12 - 9 * B - 6 * C) * x3 + (-18 + 12 * B + 6 * C) * x2 + (6 - 2 * B)) / 6.0
    far = ((-B - 6 * C) * x3 + (6 * B + 30 * C) * x2 + (-12 * B - 48 * C) * x + (8 * B + 24 * C)) / 6.0
    return np.where(x < 1, near, np.where(x < 2, far, 0.0))


def _lanczos(u):
    x = u * 2.0
    return np.where(np.abs(x) < 2.0, np.sinc(x) * np.sinc(x / 2.0), 0.0)


FILTERS = {"box": _box, "triangle": _triangle, "gaussian": _gauss,
           "mitchell": _mitchell, "lanczos": _lanczos}
DEFAULT_FILTER_SIZE = {"box": 1.0, "triangle": 2.0, "gaussian": 3.0, "mitchell": 4.0, "lanczos": 4.0}


# ---------------------------------------------------------------- ordering
def _hilbert_d2xy(n, d):
    x = y = 0
    t = d
    s = 1
    while s < n:
        rx = 1 & (t // 2)
        ry = 1 & (t ^ rx)
        if ry == 0:
            if rx == 1:
                x = s - 1 - x
                y = s - 1 - y
            x, y = y, x
        x += s * rx
        y += s * ry
        t //= 4
        s *= 2
    return x, y


def hilbert_order(nx, ny):
    """Bucket coordinates in Hilbert-curve order (mental ray's default bucket order)."""
    n = 1
    while n < max(nx, ny):
        n *= 2
    out = []
    for d in range(n * n):
        x, y = _hilbert_d2xy(n, d)
        if x < nx and y < ny:
            out.append((x, y))
    return out


# ---------------------------------------------------------------- bucket
def _splat(acc, cx, cy, area, col, x_org, y_org, W, H, kern, rx, ry, mx, my):
    bx = np.floor(cx).astype(np.int64)
    by = np.floor(cy).astype(np.int64)
    for oy in range(-my, my + 1):
        for ox in range(-mx, mx + 1):
            ix = bx + ox
            iy = by + oy
            ux = (cx - (ix + 0.5)) / rx
            uy = (cy - (iy + 0.5)) / ry
            inside = ((np.abs(ux) < 1.0) & (np.abs(uy) < 1.0)
                      & (ix >= x_org) & (ix < x_org + W) & (iy >= y_org) & (iy < y_org + H))
            if not inside.any():
                continue
            w = kern(ux[inside]) * kern(uy[inside]) * area[inside]
            yy = iy[inside] - y_org
            xx = ix[inside] - x_org
            for ch in range(4):
                np.add.at(acc[..., ch], (yy, xx), w * col[inside][:, ch])
            np.add.at(acc[..., 4], (yy, xx), w)


def render_bucket(tr, st, x0, y0, x1, y1):
    kern = FILTERS.get(st.filter, _box)
    rx = ry = max(st.filter_size, 0.5) / 2.0
    mx = my = int(math.ceil(rx + 0.5)) - 1
    thr = np.asarray(st.contrast, dtype=np.float64)

    lo = min(st.min_samples, st.max_samples)
    hi = max(st.min_samples, st.max_samples)
    B = 2 ** max(0, -lo)                       # coarse block size in pixels

    # Region covered by samples (bucket + filter margin), aligned to the coarse block
    rx0 = int(math.floor((x0 - mx) / B)) * B
    ry0 = int(math.floor((y0 - my) / B)) * B
    rx1 = int(math.ceil((x1 + mx) / B)) * B
    ry1 = int(math.ceil((y1 + my) / B)) * B
    W, H = rx1 - rx0, ry1 - ry0
    acc = np.zeros((H, W, 5))

    L = lo
    s = 2.0 ** (-L)                            # cell size in pixels
    gw, gh = int(round(W / s)), int(round(H / s))
    gx, gy = np.meshgrid(np.arange(gw), np.arange(gh))
    grid = np.zeros((gh, gw, 4))
    active = np.ones((gh, gw), dtype=bool)

    def evaluate(mask):
        ys, xs = np.nonzero(mask)
        px = rx0 + (xs + 0.5) * s
        py = ry0 + (ys + 0.5) * s
        grid[ys, xs] = tr.sample(px, py)

    def finalize(mask):
        ys, xs = np.nonzero(mask)
        if ys.size == 0:
            return
        cols = grid[ys, xs]
        if L < 0:
            # coarse cell: every pixel it covers gets this colour as a level-0 sample
            b = int(round(s))
            for oy in range(b):
                for ox in range(b):
                    cx = rx0 + xs * b + ox + 0.5
                    cy = ry0 + ys * b + oy + 0.5
                    _splat(acc, cx, cy, np.ones(ys.size), cols, rx0, ry0, W, H, kern, rx, ry, mx, my)
        else:
            cx = rx0 + (xs + 0.5) * s
            cy = ry0 + (ys + 0.5) * s
            _splat(acc, cx, cy, np.full(ys.size, s * s), cols, rx0, ry0, W, H, kern, rx, ry, mx, my)

    evaluate(active)
    while True:
        if L >= hi:
            finalize(active)
            break
        diff = np.zeros((gh, gw), dtype=bool)
        if gw > 1:
            hm = np.any(np.abs(grid[:, 1:] - grid[:, :-1]) > thr, axis=2)
            diff[:, 1:] |= hm
            diff[:, :-1] |= hm
        if gh > 1:
            vm = np.any(np.abs(grid[1:] - grid[:-1]) > thr, axis=2)
            diff[1:] |= vm
            diff[:-1] |= vm
        refine = active & diff
        finalize(active & ~refine)
        if not refine.any():
            break
        L += 1
        s /= 2.0
        grid = np.repeat(np.repeat(grid, 2, axis=0), 2, axis=1)
        active = np.repeat(np.repeat(refine, 2, axis=0), 2, axis=1)
        gh, gw = grid.shape[:2]
        evaluate(active)

    iy0, ix0 = y0 - ry0, x0 - rx0
    sub = acc[iy0:iy0 + (y1 - y0), ix0:ix0 + (x1 - x0)]
    w = sub[..., 4:5]
    out = np.where(w > 0, sub[..., :4] / np.maximum(w, 1e-12), 0.0)
    return out.astype(np.float32)


def render_buckets(scene, st):
    """Generator yielding (index, total, x0, y0, x1, y1, rgba) per finished bucket."""
    tr = Tracer(scene, st)
    bs = max(8, int(st.bucket_size))
    nbx = (st.width + bs - 1) // bs
    nby = (st.height + bs - 1) // bs
    order = hilbert_order(nbx, nby)
    total = len(order)
    for i, (bx, by) in enumerate(order):
        x0, y0 = bx * bs, by * bs
        x1, y1 = min(x0 + bs, st.width), min(y0 + bs, st.height)
        yield i, total, x0, y0, x1, y1, render_bucket(tr, st, x0, y0, x1, y1)


def render_image(scene, st, progress=None):
    img = np.zeros((st.height, st.width, 4), dtype=np.float32)
    for i, total, x0, y0, x1, y1, px in render_buckets(scene, st):
        img[y0:y1, x0:x1] = px
        if progress:
            progress(i + 1, total)
    return img
