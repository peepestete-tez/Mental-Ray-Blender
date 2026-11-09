"""Bounding volume hierarchy with numpy ray-packet traversal."""
import numpy as np


class BVH:
    def __init__(self, v0, e1, e2, leaf_size=8, tmin=1e-7):
        self.v0 = np.ascontiguousarray(v0, dtype=np.float64)
        self.e1 = np.ascontiguousarray(e1, dtype=np.float64)
        self.e2 = np.ascontiguousarray(e2, dtype=np.float64)
        self.tmin = tmin
        T = len(self.v0)
        self.nnodes = 0
        if T == 0:
            return
        p1 = self.v0 + self.e1
        p2 = self.v0 + self.e2
        lo = np.minimum(np.minimum(self.v0, p1), p2)
        hi = np.maximum(np.maximum(self.v0, p1), p2)
        cen = (lo + hi) * 0.5

        bmin, bmax, left, right, start, count, order = [], [], [], [], [], [], []

        def new_node():
            bmin.append(None); bmax.append(None)
            left.append(-1); right.append(-1)
            start.append(0); count.append(0)
            return len(left) - 1

        root = new_node()
        stack = [(root, np.arange(T))]
        while stack:
            n, idx = stack.pop()
            bmin[n] = lo[idx].min(axis=0)
            bmax[n] = hi[idx].max(axis=0)
            c = cen[idx]
            ext = c.max(axis=0) - c.min(axis=0)
            ax = int(np.argmax(ext))
            if len(idx) <= leaf_size or ext[ax] <= 0.0:
                start[n] = len(order)
                count[n] = len(idx)
                order.extend(idx.tolist())
                continue
            srt = idx[np.argsort(c[:, ax], kind="stable")]
            h = len(srt) // 2
            l, r = new_node(), new_node()
            left[n], right[n] = l, r
            stack.append((r, srt[h:]))
            stack.append((l, srt[:h]))

        self.nnodes = len(left)
        self.bmin = np.array(bmin)
        self.bmax = np.array(bmax)
        self.left = np.array(left)
        self.right = np.array(right)
        self.start = np.array(start)
        self.count = np.array(count)
        self.order = np.array(order, dtype=np.int64)

    def intersect(self, o, d, tmax):
        """Nearest hit for N rays. Returns (t, tri, u, v); tri == -1 on miss."""
        N = o.shape[0]
        best = np.array(tmax, dtype=np.float64, copy=True)
        tri = np.full(N, -1, dtype=np.int64)
        bu = np.zeros(N)
        bv = np.zeros(N)
        if self.nnodes == 0 or N == 0:
            return best, tri, bu, bv
        dd = np.where(np.abs(d) < 1e-12, 1e-12, d)
        inv = 1.0 / dd
        stack = [(0, np.arange(N))]
        while stack:
            n, idx = stack.pop()
            t0 = (self.bmin[n] - o[idx]) * inv[idx]
            t1 = (self.bmax[n] - o[idx]) * inv[idx]
            tn = np.minimum(t0, t1).max(axis=1)
            tf = np.maximum(t0, t1).min(axis=1)
            keep = (tf >= np.maximum(tn, 0.0)) & (tn < best[idx])
            idx = idx[keep]
            if idx.size == 0:
                continue
            if self.left[n] < 0:
                s, c = self.start[n], self.count[n]
                tids = self.order[s:s + c]
                v0 = self.v0[tids]; e1 = self.e1[tids]; e2 = self.e2[tids]
                od = o[idx]; dr = d[idx]
                pvec = np.cross(dr[:, None, :], e2[None, :, :])
                det = np.einsum('mj,kmj->km', e1, pvec)
                ok = np.abs(det) > 1e-14
                invdet = 1.0 / np.where(ok, det, 1.0)
                tvec = od[:, None, :] - v0[None, :, :]
                u = np.einsum('kmj,kmj->km', tvec, pvec) * invdet
                qvec = np.cross(tvec, e1[None, :, :])
                v = np.einsum('kj,kmj->km', dr, qvec) * invdet
                t = np.einsum('mj,kmj->km', e2, qvec) * invdet
                valid = (ok & (u >= 0) & (v >= 0) & (u + v <= 1)
                         & (t > self.tmin) & (t < best[idx][:, None]))
                t = np.where(valid, t, np.inf)
                j = t.argmin(axis=1)
                rows = np.arange(len(idx))
                tm = t[rows, j]
                upd = np.isfinite(tm)
                if upd.any():
                    sel = idx[upd]
                    best[sel] = tm[upd]
                    tri[sel] = tids[j[upd]]
                    bu[sel] = u[rows, j][upd]
                    bv[sel] = v[rows, j][upd]
            else:
                l, r = self.left[n], self.right[n]
                # visit the child on the near side of the first ray first
                ax = int(np.argmax(np.abs(self.bmax[l] + self.bmin[l] - self.bmax[r] - self.bmin[r])))
                near_left = (self.bmax[l][ax] + self.bmin[l][ax]) * d[idx[0], ax] <= \
                            (self.bmax[r][ax] + self.bmin[r][ax]) * d[idx[0], ax]
                if near_left:
                    stack.append((r, idx)); stack.append((l, idx))
                else:
                    stack.append((l, idx)); stack.append((r, idx))
        return best, tri, bu, bv
