import bpy
import numpy as np

from .core import render_buckets
from .export import export_scene, ExportError, srgb_to_lin


class MentalRayEngine(bpy.types.RenderEngine):
    bl_idname = "MENTALRAY"
    bl_label = "Mental Ray (Softimage 3/4)"
    bl_use_preview = False

    def render(self, depsgraph):
        scene = depsgraph.scene
        scale = scene.render.resolution_percentage / 100.0
        w = max(1, int(scene.render.resolution_x * scale))
        h = max(1, int(scene.render.resolution_y * scale))

        self.update_stats("", "Mental Ray: exporting scene")
        try:
            mr_scene, st = export_scene(depsgraph, w, h)
        except ExportError as e:
            self.report({'ERROR'}, str(e))
            return
        display_referred = scene.mentalray.display_referred

        for i, total, x0, y0, x1, y1, rgba in render_buckets(mr_scene, st):
            if self.test_break():
                break
            buf = rgba.astype(np.float64)
            if display_referred:
                a = buf[..., 3:4]
                un = np.where(a > 0, buf[..., :3] / np.maximum(a, 1e-8), 0.0)
                buf[..., :3] = srgb_to_lin(np.clip(un, 0.0, 1.0)) * a
            bw, bh = x1 - x0, y1 - y0
            # Blender's image origin is bottom-left
            result = self.begin_result(x0, h - y1, bw, bh)
            layer = result.layers[0].passes["Combined"]
            flat = np.ascontiguousarray(buf[::-1], dtype=np.float32).reshape(-1, 4)
            try:
                layer.rect = flat.tolist()
            except Exception:
                layer.rect.foreach_set(flat.reshape(-1))       # newer API fallback
            self.end_result(result)
            self.update_progress((i + 1) / total)
            self.update_stats("", f"Mental Ray: bucket {i + 1}/{total}")
