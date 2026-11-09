# SPDX-License-Identifier: GPL-3.0-or-later
"""Env_Atmosphere sky and Water texture shader, as numpy code with no bpy dependency.

Most of this module is taken unchanged from `mrfx.py` of the supplied "Fogle3D" Blender add-on
(copyright its authors, GPL-3.0-or-later; see NOTICE). Only the two adapters at the bottom
(`AtmosphereSky`, `make_water_image`) are new, and they only convert between Blender's Z-up world and
the Y-up space the original code works in.

Read this before trusting it for accuracy: the original author states that these are *approximations*
of mental ray shaders, not the shaders themselves. Their constants were "set by eye against the shader's
own preset icons". Treat the look as mental-ray-flavoured, not mental-ray-exact.
"""
import math

import numpy as np

WATER_SIZE = 512      # the original's default picture size for the Water shader

# ---- begin code taken from Fogle3D mrfx.py (unchanged) -------------------------------------------------------
def _f(p, k, d):
    v = p.get(k, d)
    try:
        return float(v)
    except (TypeError, ValueError):
        return float(d)


def _c(p, k, d):
    v = p.get(k)
    if isinstance(v, list) and len(v) >= 3:
        return np.array([float(x) for x in v[:3]], np.float64)
    return np.array(d, np.float64)


# ------------------------------------------------------------------------------------------------- noise
def _hash(ix, iy, iz, seed=0):
    h = (ix.astype(np.int64).astype(np.uint32) * np.uint32(0x8DA6B343)) \
        ^ (iy.astype(np.int64).astype(np.uint32) * np.uint32(0xD8163841)) \
        ^ (iz.astype(np.int64).astype(np.uint32) * np.uint32(0xCB1AB31F)) ^ np.uint32(seed * 0x9E3779B9 & 0xFFFFFFFF)
    h ^= h >> np.uint32(13)
    h *= np.uint32(0x5BD1E995)
    h ^= h >> np.uint32(15)
    return h


_GRAD = np.array([[1, 1, 0], [-1, 1, 0], [1, -1, 0], [-1, -1, 0], [1, 0, 1], [-1, 0, 1], [1, 0, -1], [-1, 0, -1],
                  [0, 1, 1], [0, -1, 1], [0, 1, -1], [0, -1, -1], [1, 1, 0], [-1, 1, 0], [0, -1, 1], [0, -1, -1]],
                 np.float64)


def noise3(x, y, z, seed=0):
    """Gradient noise, about -1..1, smooth (quintic fade)."""
    x0, y0, z0 = np.floor(x), np.floor(y), np.floor(z)
    fx, fy, fz = x - x0, y - y0, z - z0
    ix, iy, iz = x0.astype(np.int64), y0.astype(np.int64), z0.astype(np.int64)

    def fade(t):
        return t * t * t * (t * (t * 6 - 15) + 10)
    u, v, w = fade(fx), fade(fy), fade(fz)
    acc = None
    for dx in (0, 1):
        for dy in (0, 1):
            for dz in (0, 1):
                g = _GRAD[(_hash(ix + dx, iy + dy, iz + dz, seed) & np.uint32(15)).astype(np.intp)]
                d = g[..., 0] * (fx - dx) + g[..., 1] * (fy - dy) + g[..., 2] * (fz - dz)
                wgt = (u if dx else 1 - u) * (v if dy else 1 - v) * (w if dz else 1 - w)
                acc = d * wgt if acc is None else acc + d * wgt
    return acc


def _rand01(*ints, seed=0):
    a = [np.asarray(i) for i in ints] + [np.zeros_like(np.asarray(ints[0]))] * (3 - len(ints))
    return (_hash(a[0], a[1], a[2], seed) & np.uint32(0xFFFFFF)).astype(np.float64) / float(0x1000000)


# ------------------------------------------------------------------------------------------------- the sky
# The constants below are not the shader's (its code is not part of the port): they were set by eye against the
# shader's own preset icons, which are mental ray renders of the three sky scenes at their first and last key
# (re/tools/sky1/calib.sh).
BETA_R = 2.0 * np.array([5.8e-6, 13.5e-6, 33.1e-6])   # Rayleigh scattering at density 1, per metre (red, green, blue)
EXPOSURE = 1.0                                    # radiance scale of the in-scattered light
SUN_PATH = 0.03                                   # share of the optical depth towards the sun that dims the light
MOON_ALBEDO = 0.08                                # moon radiance per unit of sun intensity, fully lit
MOON_SIZE = 0.4                                   # the moon's diameter in degrees per unit of moon_size
GROUND_FADE = 0.05                                # radians below the horizon over which the ground darkens
CLOUD_AMBIENT = 0.004                             # cloud brightness per unit of sun intensity, unlit side
BETA_M = 1.5e-4                                   # Mie (dust) scattering per metre and unit of dust density
MIE_G = 0.8
NOISE_GAIN = 2.0                                  # gradient noise to about -1 .. 1


def _sphere_exit(r0, mu, R):
    """Distance along a ray from radius r0 (cosine mu to the vertical) to the sphere of radius R it is inside of."""
    disc = r0 * r0 * (mu * mu - 1.0) + R * R
    return -r0 * mu + np.sqrt(np.maximum(disc, 0.0))


def _ground_hit(r0, mu, R):
    """Distance to the planet (radius R) along the ray, or inf where it misses."""
    disc = r0 * r0 * (mu * mu - 1.0) + R * R
    d = -r0 * mu - np.sqrt(np.maximum(disc, 0.0))
    return np.where((disc >= 0) & (mu < 0) & (d > 0), d, np.inf)


class Sky:
    """Env_Atmosphere at one time: `radiance(dirs)` for unit directions in Softimage's world space (Y up)."""

    def __init__(self, p):
        self.p = p
        self.R = max(_f(p, "planet_radius", 6.0e6), 1.0e4)
        self.h = max(_f(p, "view_altitude", 1.8), 0.5)
        self.HR = max(_f(p, "atmosphere_thickness", 7400.0), 50.0)
        self.HM = max(_f(p, "dust_thickness", 1200.0), 10.0)
        self.bR = BETA_R * max(_f(p, "atmosphere_density", 1.0), 0.0)
        self.bM = BETA_M * max(_f(p, "dust_density", 0.01), 0.0)
        self.top = self.R + 8.0 * max(self.HR, self.HM)
        self.gain = _f(p, "view_gain", 1.0)
        self.sun_i = _f(p, "sun_intensity", 10.0) * _c(p, "sun_col", [1.0, 0.886, 0.662])
        self.sun = self.direction(_f(p, "sun_inclination", 30.0), _f(p, "sun_azimuth", -30.0))
        self.moon = self.direction(_f(p, "moon_inclination", 85.0), _f(p, "moon_azimuth", -100.0))
        self.samples = int(min(max(_f(p, "no_samples", 10), 12), 32))
        self._grid()

    @staticmethod
    def direction(inclination, azimuth):
        """Inclination: degrees from the vertical (+Y); azimuth: in the X-Z plane, 0 along +X, 90 along +Z."""
        i, a = math.radians(inclination), math.radians(azimuth)
        return np.array([math.sin(i) * math.cos(a), math.cos(i), math.sin(i) * math.sin(a)])

    # -- optical depth (Rayleigh, Mie) from points to the sun; inf where the planet is in the way
    def _sun_depth(self, r, mu_s, n=8):
        far = _sphere_exit(r, mu_s, self.top)
        blocked = np.isfinite(_ground_hit(r, mu_s, self.R))
        dR = np.zeros_like(r)
        dM = np.zeros_like(r)
        for k in range(n):
            s = far * (k + 0.5) / n
            rr = np.sqrt(r * r + s * s + 2 * r * s * mu_s)
            alt = np.maximum(rr - self.R, 0.0)
            dR += np.exp(-alt / self.HR) * far / n
            dM += np.exp(-alt / self.HM) * far / n
        dR = np.where(blocked, np.inf, dR)
        dM = np.where(blocked, np.inf, dM)
        return dR, dM

    def _grid(self):
        """The scattering integrals over view elevation x relative azimuth to the sun (the atmosphere is symmetric about
        the vertical plane through the sun): SR, SM (RGB, without the phase functions) and the transmittance T."""
        ne, na = 192, 48
        xs = np.linspace(-1.0, 1.0, ne)
        sin_e = np.sign(xs) * xs * xs                  # dense near the horizon
        cos_e = np.sqrt(np.maximum(1 - sin_e * sin_e, 0.0))
        dphi = np.linspace(0.0, math.pi, na)
        sun_az = math.atan2(self.sun[2], self.sun[0])
        az = sun_az + dphi
        E, A = np.meshgrid(np.arange(ne), np.arange(na), indexing="ij")
        d = np.stack([cos_e[E] * np.cos(az[A]), sin_e[E], cos_e[E] * np.sin(az[A])], -1).reshape(-1, 3)
        r0 = self.R + self.h
        mu = d[:, 1]
        L = np.minimum(_sphere_exit(r0, mu, self.top), _ground_hit(r0, mu, self.R))
        n = self.samples
        SR = np.zeros((len(d), 3))
        SM = np.zeros((len(d), 3))
        tR = np.zeros(len(d))
        tM = np.zeros(len(d))
        prev = np.zeros(len(d))
        for k in range(n):
            s1 = L * ((k + 1) / n) ** 2              # samples denser near the eye
            ds = s1 - prev
            s = prev + ds * 0.5
            prev = s1
            px = d * s[:, None]
            px[:, 1] += r0
            rr = np.linalg.norm(px, axis=1)
            alt = np.maximum(rr - self.R, 0.0)
            rhoR = np.exp(-alt / self.HR)
            rhoM = np.exp(-alt / self.HM)
            tR += rhoR * ds * 0.5
            tM += rhoM * ds * 0.5
            mu_s = (px @ self.sun) / rr
            sR, sM = self._sun_depth(rr, mu_s)
            with np.errstate(invalid="ignore"):
                tau = (self.bR[None, :] * (tR + SUN_PATH * sR)[:, None]
                       + 1.1 * self.bM * (tM + SUN_PATH * sM)[:, None])
            att = np.exp(-np.nan_to_num(tau, nan=np.inf, posinf=np.inf))
            SR += (rhoR * ds)[:, None] * att
            SM += (rhoM * ds)[:, None] * att
            tR += rhoR * ds * 0.5
            tM += rhoM * ds * 0.5
        T = np.exp(-(self.bR[None, :] * tR[:, None] + 1.1 * self.bM * tM[:, None]))
        self.ne, self.na = ne, na
        self.gSR = SR.reshape(ne, na, 3)
        self.gSM = SM.reshape(ne, na, 3)
        self.gT = T.reshape(ne, na, 3)
        self.sun_az = sun_az

    def _lookup(self, dirs, *grids):
        sin_e = np.clip(dirs[:, 1], -1.0, 1.0)
        x = np.sign(sin_e) * np.sqrt(np.abs(sin_e))
        fe = (x + 1.0) * 0.5 * (self.ne - 1)
        az = np.arctan2(dirs[:, 2], dirs[:, 0]) - self.sun_az
        dphi = np.abs((az + math.pi) % (2 * math.pi) - math.pi)
        fa = dphi / math.pi * (self.na - 1)
        e0 = np.clip(np.floor(fe).astype(np.intp), 0, self.ne - 2)
        a0 = np.clip(np.floor(fa).astype(np.intp), 0, self.na - 2)
        te = (fe - e0)[:, None]
        ta = (fa - a0)[:, None]
        out = []
        for g in grids:
            v = (g[e0, a0] * (1 - te) * (1 - ta) + g[e0 + 1, a0] * te * (1 - ta)
                 + g[e0, a0 + 1] * (1 - te) * ta + g[e0 + 1, a0 + 1] * te * ta)
            out.append(v)
        return out

    def transmittance(self, dirs):
        return self._lookup(dirs, self.gT)[0]

    def atmosphere(self, dirs):
        """In-scattered sunlight (linear, before the gain) and the transmittance to space along each direction."""
        SR, SM, T = self._lookup(dirs, self.gSR, self.gSM, self.gT)
        mu = np.clip(dirs @ self.sun, -1.0, 1.0)[:, None]
        pR = 3.0 / (16 * math.pi) * (1 + mu * mu) * EXPOSURE
        g = MIE_G
        pM = 3.0 / (8 * math.pi) * EXPOSURE * (1 - g * g) * (1 + mu * mu) / ((2 + g * g) * np.power(1 + g * g - 2 * g * mu, 1.5))
        L = self.sun_i[None, :] * (self.bR[None, :] * SR * pR + self.bM * SM * pM)
        return L, T

    # -- the bodies in front of the black of space
    def sun_disc(self, dirs, T, pix):
        p = self.p
        r = math.radians(max(_f(p, "sun_size", 5.0), 0.01) * 0.5)
        halo = max(_f(p, "sun_halo_size", 0.0), 0.0)
        ang = np.arccos(np.clip(dirs @ self.sun, -1.0, 1.0))
        soft = max(r * halo * 4.0, pix)
        disc = np.clip((r + soft * 0.5 - ang) / soft, 0.0, 1.0)
        glow = np.exp(-np.maximum(ang - r, 0.0) / max(r * halo * 3.0, 1e-6)) * (halo > 0) * 0.35
        k = (1.0 - min(max(_f(p, "sun_moderate", 0.0), 0.0), 1.0)) * (disc + glow)
        return self.sun_i[None, :] * k[:, None] * T

    def moon_disc(self, dirs, T, pix):
        p = self.p
        r = math.radians(max(_f(p, "moon_size", 5.0), 0.0) * MOON_SIZE * 0.5)
        col = _c(p, "moon_col", [0.8, 0.8, 0.8])
        if r <= 0 or not col.any():
            return 0.0
        halo = max(_f(p, "moon_halo_size", 0.1), 0.0)
        c = np.clip(dirs @ self.moon, -1.0, 1.0)
        ang = np.arccos(c)
        x = np.clip(np.sin(ang) / max(math.sin(r), 1e-9), 0.0, 1.0)
        tang = dirs - c[:, None] * self.moon[None, :]
        tl = np.linalg.norm(tang, axis=1)[:, None]
        tang = np.where(tl > 1e-12, tang / np.maximum(tl, 1e-12), 0.0)
        n = -np.sqrt(np.maximum(1 - x * x, 0.0))[:, None] * self.moon[None, :] + x[:, None] * tang
        lit = np.clip(n @ self.sun, 0.0, None)
        inside = np.clip((r + pix * 0.5 - ang) / max(pix, 1e-9), 0.0, 1.0) * (c > 0)
        base = self.sun_i[None, :] * col[None, :] * MOON_ALBEDO
        # the halo: scattered light of the lit part, falling off over `halo` moon radii
        phase = max(float(self.sun @ -self.moon) * 0.5 + 0.5, 0.0)
        glow = np.exp(-np.maximum(ang - r, 0.0) / max(r * halo * 3.0, 1e-6)) * (halo > 0) * 0.15 * (1 - phase) * (c > 0)
        return (base * (lit * inside + glow)[:, None]) * T

    def stars(self, dirs, T, pix, t):
        p = self.p
        dens = min(max(_f(p, "star_density", 0.6), 0.0), 1.0)
        bright = _f(p, "star_bright_scale", 1.0)
        if dens <= 0 or bright <= 0:
            return 0.0
        bias = _f(p, "star_bias", 0.1)
        vari = min(max(_f(p, "star_bright_vari", 0.5), 0.0), 1.0)
        sat = min(max(_f(p, "star_saturation", 0.2), 0.0), 1.0)
        rot = math.radians(_f(p, "star_rotation", 0.0))
        sx, sy = max(_f(p, "star_x_scale", 4.0), 0.1), max(_f(p, "star_y_scale", 4.0), 0.1)
        ox, oy = _f(p, "star_x_offset", 0.0), _f(p, "star_y_offset", 0.0)
        stime = _f(p, "star_time", 0.0)
        # cells on the sphere: azimuth x inclination, `cell` degrees at the scale 4 (a few hundred stars per steradian)
        az = np.arctan2(dirs[:, 2], dirs[:, 0]) + rot
        inc = np.arccos(np.clip(dirs[:, 1], -1.0, 1.0))
        cell = math.radians(0.9) * sx / 4.0, math.radians(0.9) * sy / 4.0
        u = az / cell[0] + ox
        v = inc / cell[1] + oy
        iu, iv = np.floor(u), np.floor(v)
        has = _rand01(iu, iv, seed=1) < dens * 0.5
        cu = iu + 0.2 + 0.6 * _rand01(iu, iv, seed=2)
        cv = iv + 0.2 + 0.6 * _rand01(iu, iv, seed=3)
        du = (u - cu) * cell[0] * np.sin(np.maximum(inc, 1e-3))
        dv = (v - cv) * cell[1]
        dist = np.sqrt(du * du + dv * dv)
        size = math.radians(0.04)
        sigma = max(size, pix * 0.6)
        prof = np.exp(-0.5 * (dist / sigma) ** 2) * (size / sigma) ** 2
        lum = bias + (1 - bias) * _rand01(iu, iv, seed=4) ** 3
        twinkle = 1.0 - vari * (0.5 + 0.5 * np.sin(stime * 6.0 + 40.0 * _rand01(iu, iv, seed=5)))
        hue = _rand01(iu, iv, seed=6)
        tint = np.stack([1 + sat * (hue - 0.5), np.ones_like(hue), 1 - sat * (hue - 0.5)], -1)
        s = (bright * lum * twinkle * prof * has)[:, None] * tint
        return s * T

    def clouds(self, dirs, pix, t):
        """Coverage (0..1) and distance (m) of the cloud layer along each direction (coverage 0 below the horizon)."""
        p = self.p
        amp = _f(p, "cloud_amplitude", 0.7)
        centre = _f(p, "cloud_centre", 0.3)
        if amp <= 0 and centre <= 0:
            return None, None
        ratio = _f(p, "cloud_ratio", 0.707)
        iters = max(_f(p, "cloud_iterations", 5.0), 0.0)
        alt = max(_f(p, "cloud_altitude", 500.0), 10.0)
        xs, ys = max(_f(p, "cloud_x_scale", 1.0), 1e-3), max(_f(p, "cloud_y_scale", 1.0), 1e-3)
        xo, yo = _f(p, "cloud_x_offset", 0.0), _f(p, "cloud_y_offset", 0.0)
        rot = math.radians(_f(p, "cloud_rotation", 0.0))
        ctime = _f(p, "cloud_time", 0.0) if int(_f(p, "cloud_animate", 0)) else 0.0
        turb = bool(int(_f(p, "cloud_absolute", 0)))
        r0 = self.R + self.h
        up = dirs[:, 1] > 1e-4
        dist = np.where(up, _sphere_exit(r0, dirs[:, 1], self.R + alt), np.inf)
        # the hit point in the lollipop map around the view point (at the pole): arc length (km) and direction
        hx = dirs[:, 0] * np.where(up, dist, 0.0)
        hz = dirs[:, 2] * np.where(up, dist, 0.0)
        hy = dirs[:, 1] * np.where(up, dist, 0.0) + r0
        horiz = np.sqrt(hx * hx + hz * hz)
        arc = np.arctan2(horiz, hy) * (self.R + alt) / 1000.0
        psi = np.arctan2(hz, hx)
        X, Z = arc * np.cos(psi), arc * np.sin(psi)
        X, Z = X * math.cos(rot) - Z * math.sin(rot), X * math.sin(rot) + Z * math.cos(rot)
        X, Z = X / xs + xo, Z / ys + yo
        # footprint of a pixel on the layer (in fractal units), to fade out octaves finer than a pixel
        foot = dist * pix / np.maximum(dirs[:, 1], 1e-3) / 1000.0 / min(xs, ys)
        # the fractal: octaves of noise, each `ratio` times the one before and twice its frequency, a fractional last
        # octave in part (so an iteration count under 1 gives faint, smooth clouds)
        total, a, f = 0.0, 1.0, 1.0
        for k in range(int(math.ceil(iters))):
            part = min(iters - k, 1.0)
            w = np.clip(1.0 - (f * foot - 0.25) / 0.25, 0.0, 1.0) * part * a
            nz = noise3(X * f, Z * f, np.full_like(X, ctime * 0.5 + 17.3 * k), seed=11 + k) * NOISE_GAIN
            if turb:
                nz = np.abs(nz) * 2 - 0.6
            total = total + w * nz
            a *= ratio
            f *= 2.0
        # the cloud where the fractal is over a half, thickening to full cover at 1
        F = centre + amp * total
        cov = np.where(up, np.clip(2.0 * F - 1.0, 0.0, 1.0), 0.0)
        return cov, dist

    def radiance(self, dirs, pix=0.002, t=0.0):
        """Linear radiance (RGB, before the gain) along unit directions (N, 3)."""
        L, T = self.atmosphere(dirs)
        # below the horizon: the planet hides what lies beyond it; the ground map is empty in every scene of the corpus,
        # so the ground is the haze of the horizon above it, darkening over GROUND_FADE radians of depression
        ground = ~np.isinf(_ground_hit(self.R + self.h, dirs[:, 1], self.R))
        if ground.any():
            dh = dirs[ground].copy()
            dh[:, 1] = 1e-3
            dh /= np.linalg.norm(dh, axis=1, keepdims=True)
            fade = np.exp(-np.maximum(-dirs[ground, 1], 0.0) / GROUND_FADE)[:, None]
            L[ground] = self.atmosphere(dh)[0] * fade + L[ground] * (1.0 - fade)
        T = T * (~ground)[:, None]
        # the stars fade out as the sky brightens (they are lost in the glare of the day)
        sky_lum = (L * self.gain).mean(axis=1, keepdims=True)
        fade = np.clip(1.0 - sky_lum * 6.0, 0.0, 1.0)
        L = L + self.stars(dirs, T, pix, t) * fade + self.moon_disc(dirs, T, pix)
        cov, dist = self.clouds(dirs, pix, t)
        if cov is not None:
            col = _c(self.p, "cloud_col", [1.0, 1.0, 1.0])
            alt = max(_f(self.p, "cloud_altitude", 500.0), 10.0)
            r = self.R + alt
            sR, sM = self._sun_depth(np.array([r]), np.array([float(self.sun[1])]), n=16)
            Ts = np.exp(-SUN_PATH * (self.bR * sR[0] + 1.1 * self.bM * sM[0])) if np.isfinite(sR[0]) else np.zeros(3)
            amb = self.atmosphere(np.array([[0.0, 1.0, 0.0], [0.0, 0.2, 0.98]]))[0].mean(axis=0)
            mu = np.clip(dirs @ self.sun, -1.0, 1.0)
            # lit from above by the sun (more towards the sun: forward scattering, thinner edges brighter)
            lit = 0.02 + 0.10 * np.power(np.clip(mu, 0.0, 1.0), 8) * (1.0 - cov) * 4.0
            C = col[None, :] * (self.sun_i[None, :] * Ts[None, :] * (lit[:, None] * 0.6 + CLOUD_AMBIENT)
                                + amb[None, :] * 1.5)
            ext = self.bR * np.exp(-alt * 0.5 / self.HR) + 1.1 * self.bM * np.exp(-alt * 0.5 / self.HM)
            Tv = np.exp(-ext[None, :] * np.minimum(dist, 1e9)[:, None])
            c = cov[:, None]
            L = L * (1.0 - c * Tv) + c * C * Tv
            T = T * (1.0 - c)
        L = L + self.sun_disc(dirs, T, pix)
        return L

    def display(self, dirs, pix=0.002, t=0.0):
        """Display values 0..1 (the shader's colour times the gain, clipped as a picture holds it)."""
        return np.clip(self.radiance(dirs, pix, t) * self.gain, 0.0, 1.0)


def water(p, t, size=WATER_SIZE):
    """The Water texture shader over the unit square at time `t` (seconds): RGBA floats (size, size, 4), row 0 at v = 1
    (the top of the picture), u along the columns."""
    n = int(min(max(_f(p, "no_waves", 10), 1), 64))
    vel = _f(p, "wave_vel", 0.3)
    f0 = max(_f(p, "wave_freq", 5.0), 1e-3)
    amp = _f(p, "wave_amp", 0.2)
    sub = max(_f(p, "wave_sub_freq", 0.2), 0.0)
    smooth = max(_f(p, "smoothness", 2.0), 0.0)
    wdir = _f(p, "wind_dir", 0.0)
    wdev = _f(p, "wind_dev", 0.0)
    tt = t + _f(p, "wave_offset", 0.0)
    ur, vr = _f(p, "urange", 1.0), _f(p, "vrange", 1.0)
    u = (np.arange(size) + 0.5) / size * ur
    v = (1.0 - (np.arange(size) + 0.5) / size) * vr
    U, V = np.meshgrid(u, v)
    h = np.zeros_like(U)
    k_ = np.arange(n)
    dev = (_rand01(k_, seed=21) - 0.5) * wdev
    ph = _rand01(k_, seed=22) * 2 * math.pi
    for k in range(n):
        f = f0 * (1.0 + sub * k)
        a = amp * (f0 / f) ** smooth
        ang = math.radians(wdir + float(dev[k]))
        c = vel * math.sqrt(f0 / f)             # deep water: the phase speed goes as the square root of the wavelength
        h += a * np.sin(2 * math.pi * f * (U * math.cos(ang) + V * math.sin(ang) - c * tt) + float(ph[k]))
    s = 0.5 + h
    bias = min(max(_f(p, "col_bias", 0.5), 1e-3), 1 - 1e-3)
    if int(_f(p, "col_clip", 1)):
        s = np.clip(s, 0.0, 1.0)
    s = np.where(s > 0, np.power(np.maximum(s, 0.0), math.log(bias) / math.log(0.5)), s)
    tr = p.get("col_trough") if isinstance(p.get("col_trough"), list) else [0.0, 0.0, 0.3, 0.0]
    cr = p.get("col_crest") if isinstance(p.get("col_crest"), list) else [1.0, 1.0, 1.0, 1.0]
    tr = np.array((list(tr) + [0.0] * 4)[:4], np.float64)
    cr = np.array((list(cr) + [1.0] * 4)[:4], np.float64)
    img = tr[None, None, :] + (cr - tr)[None, None, :] * s[..., None]
    return np.clip(img, 0.0, 1.0)

# ---- end of code taken from Fogle3D mrfx.py -----------------------------------------------------------------


# ------------------------------------------------------------------------------------------------- adapters (new)
def blender_to_si(d):
    """Blender world (Z up, -Y towards the viewer) -> the Y-up space of the code above. A proper rotation."""
    d = np.asarray(d, dtype=np.float64)
    return np.stack([d[..., 0], d[..., 2], -d[..., 1]], axis=-1)


def _angles(direction_blender):
    """(inclination from vertical, azimuth in the X-Z plane) in degrees, as the original's parameters want them."""
    v = blender_to_si(np.asarray(direction_blender, dtype=np.float64))
    v = v / max(float(np.linalg.norm(v)), 1e-20)
    return math.degrees(math.acos(max(-1.0, min(1.0, v[1])))), math.degrees(math.atan2(v[2], v[0]))


class AtmosphereSky:
    """Drop-in for core.sky.Sky: `eval(directions)` with directions in Blender world space."""

    def __init__(self, params, sun_dir, moon_dir=None, pixel_angle=0.002):
        p = dict(params)
        inc, az = _angles(sun_dir)
        p["sun_inclination"], p["sun_azimuth"] = inc, az
        if moon_dir is not None:
            p["moon_inclination"], p["moon_azimuth"] = _angles(moon_dir)
        self.visible_to_camera = bool(p.pop("visible_to_camera", True))
        self.pixel_angle = pixel_angle
        self._sky = Sky(p)

    def eval(self, d):
        d = np.asarray(d, dtype=np.float64)
        d = d / np.maximum(np.linalg.norm(d, axis=1, keepdims=True), 1e-20)
        return self._sky.display(blender_to_si(d), pix=self.pixel_angle)


def make_water_image(params, t, urange=1.0, vrange=1.0, size=WATER_SIZE, as_height=False):
    """The Water shader as an (size, size, 3) float image, row 0 = v 0 (this package's convention; the original
    puts v = 1 on row 0, so it is flipped). `as_height` gives a black-to-white ramp for bump mapping."""
    p = dict(params)
    p["urange"], p["vrange"] = urange, vrange
    if as_height:
        p["col_trough"], p["col_crest"] = [0.0, 0.0, 0.0, 0.0], [1.0, 1.0, 1.0, 1.0]
    img = water(p, t, size=size)[..., :3]
    return np.ascontiguousarray(img[::-1])
