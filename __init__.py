bl_info = {
    "name": "Mental Ray (Softimage 3/4 era)",
    "author": "Community",
    "version": (0, 3, 0),
    "blender": (4, 0, 0),
    "location": "Render Properties > Render Engine > Mental Ray (Softimage 3/4)",
    "description": "Standalone ray-trace render engine modelled on mental ray 3.x for 90s-style CGI",
    "category": "Render",
}

# bpy is only available inside Blender; the renderer core (mentalray.core) must stay importable without it.
try:
    import bpy
except ImportError:
    bpy = None


def _reload_all():
    """Re-import every submodule in dependency order. Blender re-executes this file when an add-on is
    re-enabled or updated, but submodules stay cached in sys.modules; reloading only some of them leaves
    stale code behind (e.g. an old `core` missing names a newer `props` imports)."""
    import importlib
    import sys
    base = __name__
    order = ["core.scene", "core.bvh", "core.sky", "core.textures", "core.tracer", "core.sampler", "core",
             "props", "export", "engine", "ui"]
    for name in order:
        mod = sys.modules.get(f"{base}.{name}")
        if mod is not None:
            importlib.reload(mod)


if bpy is not None:
    if "props" in locals():
        _reload_all()
    from . import props, export, engine, ui


def register():
    props.register()
    bpy.utils.register_class(engine.MentalRayEngine)
    ui.register()


def unregister():
    ui.unregister()
    bpy.utils.unregister_class(engine.MentalRayEngine)
    props.unregister()
