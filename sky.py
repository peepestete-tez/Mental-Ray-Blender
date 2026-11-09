"""Procedural 90s-CGI sky: horizon/zenith gradient, sun disc + glow, fractal clouds, stars.

Evaluated per ray direction, so the same sky is what the camera sees as background
and what reflective / refractive surfaces reflect."""
from dataclasses import dataclass
import math

import numpy as np


def _hash(ix, iy, seed):
    h = (ix.astype(np.int64) * 374761393 + iy.astype(np.int64) * 668265263 + int(seed) * 1442695041) & 0xFFFFFFFF
    h = ((h ^ (h >> 13)) * 1274126177) & 0xFFFFFFFF
    h = h ^ (h >> 16)
    return (h & 0xFFFF) / 65535.0


def _vnoise(x, y, seed):
    xi = np.floor(x); yi = np.floor(y)
    fx = x - xi; fy = y - yi
    fx = fx * fx * (3.0 - 2.0 * fx)
    fy = fy * fy * (3.0 - 2.0 * fy)
    a = _hash(xi, yi, seed); b = _hash(xi + 1, yi, seed)
    c = _hash(xi, yi + 1, seed); d = _hash(xi + 1, yi + 1, seed)
    return (a + (b - a) * fx) * (1.0 - fy) + (c + (d - c) * fx) * fy


def _fbm(x, y, seed, octaves=5):
    total = np.zeros_like(x)
    amp, norm = 0.5, 0.0
    for o in range(octaves):
        total += amp * _vnoise(x, y, seed + o * 17)
        norm += amp
        x = x * 2.03 + 11.7
        y = y * 2.03 + 3.1
        amp *= 0.5
    return total / norm


def _smooth(e0, e1, x):
    t = np.clip((x - e0) / max(e1 - e0, 1e-6), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


@dataclass
class Sky:
    zenith: tuple = (0.10, 0.30, 0.75)
    horizon: tuple = (0.65, 0.80, 0.95)
    ground: tuple = (0.25, 0.22, 0.20)
    gradient_power: float = 0.6          # < 1 pushes the horizon colour higher up the sky
    sun_enable: bool = True
    sun_dir: tuple = (0.4, -0.5, 0.6)    # direction TOWARDS the sun
    sun_color: tuple = (1.0, 0.95, 0.8)
    sun_size: float = 0.04               # angular radius, radians
    sun_glow: float = 0.5
    clouds: bool = True
    cloud_coverage: float = 0.5
    cloud_scale: float = 2.0
    cloud_softness: float = 0.35
    cloud_color: tuple = (1.0, 1.0, 1.0)
    cloud_shade: tuple = (0.55, 0.60, 0.72)
    cloud_seed: int = 1
    stars: bool = False
    star_density: float = 0.05
    star_brightness: float = 1.0
    visible_to_camera: bool = True

    def eval(self, d):
        """d: (n, 3) unit directions -> (n, 3) colour."""
        d = np.asarray(d, dtype=np.float64)
        d = d / np.maximum(np.linalg.norm(d, axis=1, keepdims=True), 1e-20)
        z = d[:, 2]
        zen = np.asarray(self.zenith); hor = np.asarray(self.horizon); gro = np.asarray(self.ground)

        up = np.clip(z, 0.0, 1.0) ** max(self.gradient_power, 1e-3)
        col = hor + (zen - hor) * up[:, None]
        below = np.clip(-z * 6.0, 0.0, 1.0)[:, None]
        col = np.where((z < 0)[:, None], hor + (gro - hor) * below, col)

        if self.stars:
            res = 110.0
            g = d * res
            gi = np.floor(g)
            f = g - gi - 0.5
            dist = np.linalg.norm(f, axis=1)
            key = _hash(gi[:, 0], gi[:, 1] + gi[:, 2] * 7919.0, self.cloud_seed + 11)
            br = _hash(gi[:, 0] + 13.0, gi[:, 1] + gi[:, 2] * 7919.0, self.cloud_seed + 5) ** 2
            star = (key > 1.0 - self.star_density) & (dist < 0.3) & (z > 0)
            amt = np.where(star, (1.0 - dist / 0.3) * (0.4 + 0.6 * br) * self.star_brightness, 0.0)
            col = col + amt[:, None]

        sun = np.asarray(self.sun_dir, dtype=np.float64)
        sun = sun / max(np.linalg.norm(sun), 1e-20)
        s = None
        if self.sun_enable:
            s = np.clip(d @ sun, -1.0, 1.0)
            ang = np.arccos(s)
            glow = self.sun_glow * np.exp(-ang * 7.0)
            col = col + np.asarray(self.sun_color) * glow[:, None]

        cloud_a = np.zeros(len(d))
        if self.clouds:
            hz = z > 0.0
            if hz.any():
                zz = z[hz]
                denom = zz + 0.12
                u = d[hz, 0] / denom * self.cloud_scale
                v = d[hz, 1] / denom * self.cloud_scale
                n = _fbm(u, v, self.cloud_seed)
                th = 1.0 - self.cloud_coverage
                a = _smooth(th, th + max(self.cloud_softness, 1e-3), n)
                a *= np.clip(zz * 7.0, 0.0, 1.0)             # thin out into the horizon haze
                # cheap lighting: sample shifted towards the sun, brighter where density falls off that way
                n2 = _fbm(u + sun[0] * 0.12, v + sun[1] * 0.12, self.cloud_seed)
                lit = np.clip(0.55 + (n - n2) * 5.0, 0.0, 1.0)
                cc = np.asarray(self.cloud_shade) + (np.asarray(self.cloud_color) - np.asarray(self.cloud_shade)) * lit[:, None]
                col_h = col[hz]
                col[hz] = col_h * (1.0 - a[:, None]) + cc * a[:, None]
                cloud_a[hz] = a

        if self.sun_enable:
            disc = np.clip((self.sun_size - ang) / max(self.sun_size * 0.15, 1e-4), 0.0, 1.0)
            disc *= (1.0 - cloud_a)
            col = col * (1.0 - disc[:, None]) + np.asarray(self.sun_color) * disc[:, None]

        return col


PRESETS = {
    'DAY': dict(zenith=(0.10, 0.30, 0.75), horizon=(0.65, 0.80, 0.95), ground=(0.25, 0.22, 0.20),
                gradient_power=0.6, cloud_color=(1.0, 1.0, 1.0), cloud_shade=(0.55, 0.60, 0.72),
                sun_color=(1.0, 0.95, 0.8), sun_elevation=45.0, stars=False, clouds=True, cloud_coverage=0.5),
    'SUNSET': dict(zenith=(0.12, 0.12, 0.42), horizon=(1.0, 0.52, 0.18), ground=(0.18, 0.08, 0.06),
                   gradient_power=0.45, cloud_color=(1.0, 0.65, 0.4), cloud_shade=(0.35, 0.18, 0.30),
                   sun_color=(1.0, 0.75, 0.35), sun_elevation=6.0, stars=False, clouds=True, cloud_coverage=0.45),
    'NIGHT': dict(zenith=(0.0, 0.0, 0.05), horizon=(0.05, 0.06, 0.20), ground=(0.0, 0.0, 0.02),
                  gradient_power=0.5, cloud_color=(0.25, 0.28, 0.45), cloud_shade=(0.05, 0.05, 0.12),
                  sun_color=(0.85, 0.9, 1.0), sun_elevation=35.0, stars=True, clouds=True, cloud_coverage=0.25),
    'VAPOR': dict(zenith=(0.12, 0.0, 0.35), horizon=(1.0, 0.30, 0.70), ground=(0.10, 0.0, 0.20),
                  gradient_power=0.55, cloud_color=(1.0, 0.7, 0.95), cloud_shade=(0.45, 0.15, 0.55),
                  sun_color=(1.0, 0.85, 0.4), sun_elevation=10.0, stars=True, clouds=True, cloud_coverage=0.4),
    'CLEAR': dict(zenith=(0.05, 0.25, 0.85), horizon=(0.55, 0.78, 1.0), ground=(0.20, 0.20, 0.22),
                  gradient_power=0.7, cloud_color=(1.0, 1.0, 1.0), cloud_shade=(0.6, 0.65, 0.8),
                  sun_color=(1.0, 0.97, 0.85), sun_elevation=55.0, stars=False, clouds=False, cloud_coverage=0.5),
}


def sun_direction(azimuth_deg, elevation_deg):
    az, el = math.radians(azimuth_deg), math.radians(elevation_deg)
    return (math.cos(el) * math.sin(az), math.cos(el) * math.cos(az), math.sin(el))
