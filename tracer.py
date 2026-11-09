"""Ray tracer / shader core: Constant, Lambert, Phong and Blinn shading with
reflection, refraction, transparent shadows, spot/area/directional lights and fog."""
import numpy as np

from .bvh import BVH
from .scene import CONSTANT, LAMBERT, PHONG, BLINN
from .textures import tex_color, tex_height, bump_step, bump_norm


def _norm(v):
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.maximum(n, 1e-20)


def _dot(a, b):
    return np.einsum('ij,ij->i', a, b)


class Tracer:
    def __init__(self, scene, settings, seed=1):
        self.scene = scene
        self.st = settings
        self.rng = np.random.default_rng(seed)
        pos = scene.tri_pos
        v0 = pos[:, 0]
        e1 = pos[:, 1] - pos[:, 0]
        e2 = pos[:, 2] - pos[:, 0]
        self.gn = _norm(np.cross(e1, e2)) if len(pos) else np.zeros((0, 3))
        self.tn = scene.tri_nrm
        self.tuv = scene.tri_uv
        self.tmat = scene.tri_mat
        if len(pos):
            diag = float(np.linalg.norm(pos.reshape(-1, 3).max(0) - pos.reshape(-1, 3).min(0)))
        else:
            diag = 1.0
        self.eps = max(1e-6, 1e-4 * diag)
        self.bvh = BVH(v0, e1, e2, leaf_size=8, tmin=self.eps * 0.01)
        mats = scene.materials
        self.mat_transp = np.array([m.transparency for m in mats])
        self.bg = np.asarray(scene.background, dtype=np.float64)
        self.amb = np.asarray(scene.ambient, dtype=np.float64)
        self.sky = getattr(scene, "sky", None)
        self.xforms = scene.xforms
        self.txf = scene.tri_xf
        self.has_xf = len(scene.xforms) > 0 and len(self.txf) == len(pos)
        self._tangents(pos, scene.tri_uv)

    def _tangents(self, pos, uv):
        """Per-triangle dP/du and dP/dv (for bump mapping). Triangles with degenerate UVs fall back to an
        arbitrary tangent frame, which is fine for solid textures."""
        T = len(pos)
        self.dpdu = np.zeros((T, 3)); self.dpdv = np.zeros((T, 3))
        if T == 0:
            return
        e1 = pos[:, 1] - pos[:, 0]; e2 = pos[:, 2] - pos[:, 0]
        d1 = uv[:, 1] - uv[:, 0]; d2 = uv[:, 2] - uv[:, 0]
        r = d1[:, 0] * d2[:, 1] - d2[:, 0] * d1[:, 1]
        ok = np.abs(r) > 1e-12
        inv = 1.0 / np.where(ok, r, 1.0)
        dpdu = (e1 * d2[:, 1:2] - e2 * d1[:, 1:2]) * inv[:, None]
        dpdv = (e2 * d1[:, 0:1] - e1 * d2[:, 0:1]) * inv[:, None]
        helper = np.where((np.abs(self.gn[:, 2]) < 0.9)[:, None], np.array([0.0, 0.0, 1.0]), np.array([1.0, 0.0, 0.0]))
        fb_u = _norm(np.cross(helper, self.gn))
        fb_v = np.cross(self.gn, fb_u)
        bad = ~ok | ~np.isfinite(dpdu).all(axis=1) | ~np.isfinite(dpdv).all(axis=1)
        self.dpdu = np.where(bad[:, None], fb_u, dpdu)
        self.dpdv = np.where(bad[:, None], fb_v, dpdv)

    def _bump(self, mat, P, N, uv, tri, W2O):
        """Perturb normals using the material's bump texture as a height field (Mikkelsen's surface-gradient form)."""
        tex = mat.bump
        dpdu = self.dpdu[tri]; dpdv = self.dpdv[tri]
        in_uv, h = bump_step(tex)
        if in_uv:
            hu = hv = np.full(len(P), h)
        else:
            hu = h / np.maximum(np.linalg.norm(dpdu, axis=1), 1e-12)
            hv = h / np.maximum(np.linalg.norm(dpdv, axis=1), 1e-12)
        zero = np.zeros(len(P))
        uvu = np.stack([hu, zero], axis=1); uvv = np.stack([zero, hv], axis=1)
        hp = tex_height(tex, P + dpdu * hu[:, None], W2O, uv + uvu)
        hm = tex_height(tex, P - dpdu * hu[:, None], W2O, uv - uvu)
        vp = tex_height(tex, P + dpdv * hv[:, None], W2O, uv + uvv)
        vm = tex_height(tex, P - dpdv * hv[:, None], W2O, uv - uvv)
        norm, per_uv = bump_norm(tex)
        k = mat.bump_strength * norm
        ku = k * np.linalg.norm(dpdu, axis=1) if per_uv else k
        kv = k * np.linalg.norm(dpdv, axis=1) if per_uv else k
        dbs = (hp - hm) / (2.0 * hu) * ku
        dbt = (vp - vm) / (2.0 * hv) * kv
        r1 = np.cross(dpdv, N)
        r2 = np.cross(N, dpdu)
        det = np.einsum('ij,ij->i', dpdu, r1)
        sg = np.where(det < 0, -1.0, 1.0)
        surf = sg[:, None] * (dbs[:, None] * r1 + dbt[:, None] * r2)
        out = _norm(np.abs(det)[:, None] * N - surf)
        good = np.isfinite(out).all(axis=1) & (np.abs(det) > 1e-18)
        return np.where(good[:, None], out, N)

    def environment(self, d, primary=False):
        """Colour seen along directions d that hit nothing (sky if present, else background)."""
        if self.sky is not None and (not primary or self.sky.visible_to_camera):
            return self.sky.eval(d)
        return np.tile(self.bg, (len(d), 1))

    # ------------------------------------------------------------------ camera
    def camera_rays(self, px, py):
        c = self.scene.camera
        W, H = self.st.width, self.st.height
        nx = px / W * 2.0 - 1.0 + c.shift_x
        ny = 1.0 - py / H * 2.0 + c.shift_y
        org = np.asarray(c.origin); right = np.asarray(c.right)
        up = np.asarray(c.up); fwd = np.asarray(c.forward)
        if c.ortho:
            o = org + right * (nx * c.ortho_x)[:, None] + up * (ny * c.ortho_y)[:, None]
            d = np.tile(fwd, (len(px), 1))
        else:
            o = np.tile(org, (len(px), 1))
            d = _norm(fwd + right * (nx * c.tan_x)[:, None] + up * (ny * c.tan_y)[:, None])
        return o, d

    def sample(self, px, py, chunk=16384):
        """Trace camera rays through pixel-space positions. Returns (n, 4) rgba."""
        out = np.empty((len(px), 4))
        st = self.st
        for s in range(0, len(px), chunk):
            sl = slice(s, s + chunk)
            o, d = self.camera_rays(px[sl], py[sl])
            rgb, a = self.trace(o, d, st.trace_reflect, st.trace_refract, st.trace_sum, primary=True)
            if st.clamp:
                rgb = np.clip(rgb, 0.0, 1.0)
            if st.transparent_film:
                rgb = rgb * a[:, None]
                alpha = a
            else:
                alpha = np.ones_like(a)
            out[sl, :3] = rgb
            out[sl, 3] = alpha
        return out

    # ------------------------------------------------------------------ tracing
    def trace(self, o, d, rl, fl, sl, primary=False):
        N = o.shape[0]
        t, tri, bu, bv = self.bvh.intersect(o, d, np.full(N, np.inf))
        hit = tri >= 0
        rgb = np.empty((N, 3))
        rgb[:] = self.bg
        miss = np.nonzero(~hit)[0]
        if miss.size:
            rgb[miss] = self.environment(d[miss], primary)
        alpha = hit.astype(np.float64)
        if hit.any():
            hi = np.nonzero(hit)[0]
            col = self.shade_hits(o[hi], d[hi], t[hi], tri[hi], bu[hi], bv[hi], rl, fl, sl)
            if self.st.fog:
                f = np.exp(-self.st.fog_density * t[hi])[:, None]
                fc = np.asarray(self.st.fog_color)
                col = fc + (col - fc) * f
            rgb[hi] = col
        return rgb, alpha

    def shade_hits(self, o, d, t, tri, bu, bv, rl, fl, sl):
        n = len(tri)
        P = o + d * t[:, None]
        w0 = (1.0 - bu - bv)[:, None]
        Ns = _norm(self.tn[tri, 0] * w0 + self.tn[tri, 1] * bu[:, None] + self.tn[tri, 2] * bv[:, None])
        Ng = self.gn[tri]
        back = _dot(d, Ng) > 0
        Ng = np.where(back[:, None], -Ng, Ng)
        Ns = np.where(back[:, None], -Ns, Ns)
        uv = self.tuv[tri, 0] * w0 + self.tuv[tri, 1] * bu[:, None] + self.tuv[tri, 2] * bv[:, None]
        mid = self.tmat[tri]
        eps = self.eps

        diff = np.zeros((n, 3)); spec = np.zeros((n, 3))
        refl = np.zeros(n); transp = np.zeros(n); ior = np.ones(n)
        V = -d
        Pb = P + Ng * eps

        for m in np.unique(mid):
            sel = np.nonzero(mid == m)[0]
            mat = self.scene.materials[m]
            k = len(sel)
            base = np.tile(np.asarray(mat.diffuse, dtype=np.float64), (k, 1))
            W2O = self.xforms[self.txf[tri[sel]]] if self.has_xf else None
            if mat.bump is not None and mat.bump_strength != 0.0:
                Ns[sel] = self._bump(mat, P[sel], Ns[sel], uv[sel], tri[sel], W2O)
            if mat.texture is not None:
                tc = tex_color(mat.texture, P[sel], W2O, uv[sel])
                base = base * (1.0 - mat.texture_mix) + tc * mat.texture_mix
            emis = np.asarray(mat.emission)
            if mat.shader == CONSTANT:
                diff[sel] = base + emis
            else:
                dsum = np.zeros((k, 3)); ssum = np.zeros((k, 3))
                for L in self.scene.lights:
                    self._light(L, mat, P[sel], Pb[sel], Ns[sel], V[sel], dsum, ssum)
                amb = self.amb * np.asarray(mat.ambient)
                diff[sel] = base * (amb + dsum) + emis
                spec[sel] = np.asarray(mat.specular) * ssum
            refl[sel] = mat.reflectivity
            transp[sel] = mat.transparency
            ior[sel] = max(mat.ior, 1e-3)

        out = diff

        # Refraction / transparency
        want = transp > 0
        refract_ok = fl > 0 and sl > 0
        ti = np.nonzero(want)[0] if refract_ok else np.array([], dtype=np.int64)
        if ti.size:
            di = d[ti]; Ni = Ns[ti]
            cosi = np.clip(-_dot(di, Ni), 0.0, 1.0)
            eta = np.where(back[ti], ior[ti], 1.0 / ior[ti])
            kk = 1.0 - eta ** 2 * (1.0 - cosi ** 2)
            tir = kk < 0
            Tdir = eta[:, None] * di + (eta * cosi - np.sqrt(np.clip(kk, 0, None)))[:, None] * Ni
            Rdir = di - 2.0 * _dot(di, Ni)[:, None] * Ni
            dirs = _norm(np.where(tir[:, None], Rdir, Tdir))
            orgs = P[ti] + np.where(tir[:, None], Ng[ti], -Ng[ti]) * eps
            rc, _ = self.trace(orgs, dirs, rl, fl - 1, sl - 1)
            tr = transp[ti][:, None]
            out[ti] = out[ti] * (1.0 - tr) + rc * tr
        ex = np.nonzero(want)[0] if not refract_ok else np.array([], dtype=np.int64)
        if ex.size:
            tr = transp[ex][:, None]
            out[ex] = out[ex] * (1.0 - tr) + self.environment(d[ex]) * tr

        # Reflection
        rwant = refl > 0
        if rl > 0 and sl > 0:
            ri = np.nonzero(rwant)[0]
            if ri.size:
                di = d[ri]; Ni = Ns[ri]
                rd = _norm(di - 2.0 * _dot(di, Ni)[:, None] * Ni)
                rc, _ = self.trace(Pb[ri], rd, rl - 1, fl, sl - 1)
                r = refl[ri][:, None]
                out[ri] = out[ri] * (1.0 - r) + rc * r
        else:
            ri = np.nonzero(rwant)[0]
            if ri.size:
                di = d[ri]; Ni = Ns[ri]
                rd = _norm(di - 2.0 * _dot(di, Ni)[:, None] * Ni)
                r = refl[ri][:, None]
                out[ri] = out[ri] * (1.0 - r) + self.environment(rd) * r

        return out + spec

    # ------------------------------------------------------------------ lights
    def _light_points(self, L, k):
        """Sample positions on the light for k shading points: list of (3,) or (k,3)."""
        if L.kind != "area":
            return [np.asarray(L.position, dtype=np.float64)]
        nu, nv = L.area_samples
        c = np.asarray(L.position); u = np.asarray(L.area_u); v = np.asarray(L.area_v)
        pts = []
        for i in range(nu):
            for j in range(nv):
                r = self.rng.random((k, 2))
                a = 2.0 * (i + r[:, 0]) / nu - 1.0
                b = 2.0 * (j + r[:, 1]) / nv - 1.0
                pts.append(c + u * a[:, None] + v * b[:, None])
        return pts

    def _light(self, L, mat, P, Pb, N, V, dsum, ssum):
        k = len(P)
        pts = self._light_points(L, k)
        ns = len(pts)
        col = np.asarray(L.color)
        ldir = np.asarray(L.direction)
        for sp in pts:
            if L.kind == "directional":
                Ld = np.tile(_norm(-ldir[None, :])[0], (k, 1))
                dist = np.full(k, np.inf)
            else:
                vec = sp - P
                dist = np.linalg.norm(vec, axis=1)
                Ld = vec / np.maximum(dist, 1e-20)[:, None]
            ndl = _dot(N, Ld)
            m = ndl > 0
            w = np.ones(k) / ns
            if L.kind == "spot":
                cosang = _dot(-Ld, np.tile(_norm(ldir[None, :])[0], (k, 1)))
                co = np.cos(L.cone)
                ci = np.cos(L.cone * (1.0 - L.blend))
                s = np.clip((cosang - co) / max(ci - co, 1e-6), 0.0, 1.0)
                w = w * s * s * (3.0 - 2.0 * s)
                m &= s > 0
            if L.falloff > 0 and L.kind != "directional":
                w = w * (L.falloff_ref / np.maximum(dist, 1e-6)) ** L.falloff
            idx = np.nonzero(m)[0]
            if idx.size == 0:
                continue
            if L.shadows:
                sh = self._shadow(Pb[idx], Ld[idx], dist[idx])
            else:
                sh = np.ones(idx.size)
            wi = (w[idx] * sh)
            dsum[idx] += (ndl[idx] * wi)[:, None] * col
            if mat.shader in (PHONG, BLINN) and mat.shininess >= 0:
                Ni = N[idx]; Vi = V[idx]; Li = Ld[idx]
                if mat.shader == PHONG:
                    R = 2.0 * ndl[idx][:, None] * Ni - Li
                    term = np.clip(_dot(R, Vi), 0.0, None) ** mat.shininess
                else:
                    H = _norm(Li + Vi)
                    term = np.clip(_dot(Ni, H), 0.0, None) ** (mat.shininess * 4.0)
                ssum[idx] += (term * wi)[:, None] * col

    def _shadow(self, P, Ld, dist):
        """Transmittance (0..1) along each shadow ray, honouring transparent occluders."""
        n = len(P)
        trans = np.ones(n)
        o = P.copy()
        remaining = dist.astype(np.float64).copy()
        active = np.arange(n)
        tri_tr = self.mat_transp[self.tmat]
        for _ in range(max(1, self.st.shadow_depth)):
            if active.size == 0:
                break
            t, tri, _, _ = self.bvh.intersect(o[active], Ld[active], remaining[active])
            h = tri >= 0
            if not h.any():
                break
            hid = active[h]
            tr = tri_tr[tri[h]]
            opaque = tr <= 0.0
            trans[hid[opaque]] = 0.0
            cont = ~opaque
            cid = hid[cont]
            trans[cid] *= tr[cont]
            o[cid] = o[cid] + Ld[cid] * (t[h][cont] + self.eps)[:, None]
            remaining[cid] = remaining[cid] - (t[h][cont] + self.eps)
            active = cid
        return trans
