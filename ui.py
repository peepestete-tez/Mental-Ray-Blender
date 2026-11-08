import bpy

ENGINE = 'MENTALRAY'
_SHADEABLE = {'MESH', 'CURVE', 'SURFACE', 'META', 'FONT'}


class _Base:
    COMPAT_ENGINES = {ENGINE}

    @classmethod
    def poll(cls, context):
        return context.engine == ENGINE


class MENTALRAY_OT_apply_period_color(bpy.types.Operator):
    bl_idname = "mentalray.apply_period_color"
    bl_label = "Apply Period Colour Settings"
    bl_description = "Set the view transform to Standard (no tone mapping) so output matches the raw shaded values"

    def execute(self, context):
        vs = context.scene.view_settings
        vs.view_transform = 'Standard'
        try:
            vs.look = 'None'
        except TypeError:
            pass
        vs.exposure = 0.0
        vs.gamma = 1.0
        try:
            context.scene.display_settings.display_device = 'sRGB'
        except TypeError:
            pass
        self.report({'INFO'}, "View transform set to Standard")
        return {'FINISHED'}


class MENTALRAY_OT_assign_materials(bpy.types.Operator):
    bl_idname = "mentalray.assign_materials"
    bl_label = "Add Material to Objects Without One"
    bl_description = "Give every mesh/curve object in the scene that has no material a new Mental Ray material"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        count = 0
        for ob in context.scene.objects:
            if ob.type not in _SHADEABLE or ob.data is None or not hasattr(ob.data, "materials"):
                continue
            if any(slot.material is not None for slot in ob.material_slots):
                continue
            mat = bpy.data.materials.new(name="MR " + ob.name)
            if len(ob.data.materials) == 0:
                ob.data.materials.append(mat)
            else:
                ob.material_slots[0].material = mat
            count += 1
        self.report({'INFO'}, f"Added materials to {count} object(s)")
        return {'FINISHED'}


class RENDER_PT_mentalray_sampling(_Base, bpy.types.Panel):
    bl_label = "Sampling"
    bl_space_type = 'PROPERTIES'
    bl_region_type = 'WINDOW'
    bl_context = "render"

    def draw(self, context):
        ms = context.scene.mentalray
        col = self.layout.column()
        row = col.row(align=True)
        row.prop(ms, "min_samples", text="Min")
        row.prop(ms, "max_samples", text="Max")
        col.prop(ms, "contrast", text="Contrast")
        col.separator()
        col.prop(ms, "filter")
        col.prop(ms, "filter_size")
        col.prop(ms, "bucket_size")


class RENDER_PT_mentalray_trace(_Base, bpy.types.Panel):
    bl_label = "Ray Tracing"
    bl_space_type = 'PROPERTIES'
    bl_region_type = 'WINDOW'
    bl_context = "render"

    def draw(self, context):
        ms = context.scene.mentalray
        col = self.layout.column(align=True)
        col.prop(ms, "trace_reflect")
        col.prop(ms, "trace_refract")
        col.prop(ms, "trace_sum")
        self.layout.prop(ms, "shadow_depth")


class RENDER_PT_mentalray_scene(_Base, bpy.types.Panel):
    bl_label = "Scene"
    bl_space_type = 'PROPERTIES'
    bl_region_type = 'WINDOW'
    bl_context = "render"

    def draw(self, context):
        ms = context.scene.mentalray
        col = self.layout.column()
        col.prop(ms, "ambient")
        col.prop(ms, "light_scale")
        col.prop(ms, "fog_enable")
        sub = col.column()
        sub.active = ms.fog_enable
        sub.prop(ms, "fog_color")
        sub.prop(ms, "fog_density")


class RENDER_PT_mentalray_output(_Base, bpy.types.Panel):
    bl_label = "Output"
    bl_space_type = 'PROPERTIES'
    bl_region_type = 'WINDOW'
    bl_context = "render"

    def draw(self, context):
        ms = context.scene.mentalray
        col = self.layout.column()
        col.prop(ms, "display_referred")
        col.prop(ms, "clamp")
        col.operator("mentalray.apply_period_color")


class MATERIAL_PT_mentalray_slots(_Base, bpy.types.Panel):
    """Slot list + New button, so the shader panel is reachable on objects that have no material yet."""
    bl_label = "Material"
    bl_space_type = 'PROPERTIES'
    bl_region_type = 'WINDOW'
    bl_context = "material"

    @classmethod
    def poll(cls, context):
        ob = context.object
        return context.engine == ENGINE and ob is not None and ob.type in _SHADEABLE

    def draw(self, context):
        layout = self.layout
        ob = context.object
        row = layout.row()
        row.template_list("MATERIAL_UL_matslots", "", ob, "material_slots", ob, "active_material_index", rows=3)
        col = row.column(align=True)
        col.operator("object.material_slot_add", icon='ADD', text="")
        col.operator("object.material_slot_remove", icon='REMOVE', text="")
        layout.template_ID(ob, "active_material", new="material.new")
        layout.operator("mentalray.assign_materials", icon='MATERIAL')


class MATERIAL_PT_mentalray(_Base, bpy.types.Panel):
    bl_label = "Mental Ray Shader"
    bl_space_type = 'PROPERTIES'
    bl_region_type = 'WINDOW'
    bl_context = "material"

    @classmethod
    def poll(cls, context):
        return context.engine == ENGINE and context.material is not None

    def draw(self, context):
        mat = context.material
        p = mat.mentalray
        col = self.layout.column()
        col.prop(p, "shader")
        col.prop(p, "use_viewport_color")
        if p.use_viewport_color:
            col.prop(mat, "diffuse_color", text="Diffuse")
        else:
            col.prop(p, "diffuse")
        col.prop(p, "ambient")
        if p.shader in {'PHONG', 'BLINN'}:
            col.prop(p, "specular")
            col.prop(p, "shininess")
        col.prop(p, "emission")
        col.separator()
        col.prop(p, "reflectivity")
        col.prop(p, "transparency")
        col.prop(p, "ior")


def _draw_texture(layout, t, colors=True):
    """Shared drawing for a MentalRayTexture slot."""
    col = layout.column()
    col.prop(t, "kind")
    if t.kind == 'NONE':
        return False
    if t.kind == 'IMAGE':
        col.template_ID(t, "image", open="image.open")
        row = col.row(align=True)
        row.prop(t, "repeat_u")
        row.prop(t, "repeat_v")
        return True
    if t.kind == 'NOISE':
        col.prop(t, "noise_type")
    col.prop(t, "mapping")
    if t.kind == 'CHECKER' and t.mapping == 'UV':
        row = col.row(align=True)
        row.prop(t, "repeat_u")
        row.prop(t, "repeat_v")
    else:
        col.prop(t, "scale")
    if t.kind == 'NOISE':
        if t.mapping == 'UV' and t.noise_type in {'WOOD'}:
            col.label(text="Wood rings centre on UV (0, 0)", icon='INFO')
        col.prop(t, "octaves")
        sub = col.column(align=True)
        sub.prop(t, "lacunarity")
        sub.prop(t, "gain")
        if t.noise_type in {'MARBLE', 'WOOD'}:
            col.prop(t, "distortion")
        col.prop(t, "seed")
    if colors:
        col.separator()
        col.prop(t, "color_a")
        col.prop(t, "color_b")
    return True


class MATERIAL_PT_mentalray_color_tex(_Base, bpy.types.Panel):
    bl_label = "Colour Texture"
    bl_parent_id = "MATERIAL_PT_mentalray"
    bl_space_type = 'PROPERTIES'
    bl_region_type = 'WINDOW'
    bl_context = "material"

    @classmethod
    def poll(cls, context):
        return context.engine == ENGINE and context.material is not None

    def draw(self, context):
        p = context.material.mentalray
        if _draw_texture(self.layout, p.color_tex, colors=(p.color_tex.kind != 'IMAGE')):
            self.layout.prop(p, "texture_mix")


class MATERIAL_PT_mentalray_bump(_Base, bpy.types.Panel):
    bl_label = "Bump"
    bl_parent_id = "MATERIAL_PT_mentalray"
    bl_space_type = 'PROPERTIES'
    bl_region_type = 'WINDOW'
    bl_context = "material"

    @classmethod
    def poll(cls, context):
        return context.engine == ENGINE and context.material is not None

    def draw(self, context):
        p = context.material.mentalray
        if _draw_texture(self.layout, p.bump_tex, colors=False):
            self.layout.prop(p, "bump_strength")


class DATA_PT_mentalray_light(_Base, bpy.types.Panel):
    bl_label = "Mental Ray Light"
    bl_space_type = 'PROPERTIES'
    bl_region_type = 'WINDOW'
    bl_context = "data"

    @classmethod
    def poll(cls, context):
        return context.engine == ENGINE and context.light is not None

    def draw(self, context):
        li = context.light
        p = li.mentalray
        col = self.layout.column()
        col.prop(li, "type")
        col.prop(li, "color")
        col.prop(li, "energy", text="Strength" if li.type == 'SUN' else "Power (W)")
        if li.type == 'SPOT':
            col.prop(li, "spot_size")
            col.prop(li, "spot_blend")
        if li.type == 'AREA':
            col.prop(li, "shape")
            col.prop(li, "size")
            if li.shape in {'RECTANGLE', 'ELLIPSE'}:
                col.prop(li, "size_y")
            row = col.row(align=True)
            row.prop(p, "area_u")
            row.prop(p, "area_v")
        col.prop(p, "shadows")
        if li.type != 'SUN':
            col.prop(p, "falloff")
            if p.falloff != 'NONE':
                col.prop(p, "falloff_ref")


class WORLD_PT_mentalray_sky(_Base, bpy.types.Panel):
    bl_label = "Mental Ray Sky"
    bl_space_type = 'PROPERTIES'
    bl_region_type = 'WINDOW'
    bl_context = "world"

    def draw(self, context):
        layout = self.layout
        scene = context.scene
        world = scene.world
        layout.template_ID(scene, "world", new="world.new")
        if world is None:
            return
        p = world.mentalray
        layout.prop(p, "use_sky")
        col = layout.column()
        col.active = p.use_sky
        col.prop(p, "preset")
        col.prop(p, "visible_to_camera")
        if not p.use_sky:
            col = layout.column()
            col.prop(world, "color", text="Background")


class WORLD_PT_mentalray_sky_colors(_Base, bpy.types.Panel):
    bl_label = "Gradient"
    bl_parent_id = "WORLD_PT_mentalray_sky"
    bl_space_type = 'PROPERTIES'
    bl_region_type = 'WINDOW'
    bl_context = "world"

    @classmethod
    def poll(cls, context):
        w = context.scene.world
        return context.engine == ENGINE and w is not None and w.mentalray.use_sky

    def draw(self, context):
        p = context.scene.world.mentalray
        col = self.layout.column()
        col.prop(p, "zenith")
        col.prop(p, "horizon")
        col.prop(p, "ground")
        col.prop(p, "gradient_power")


class WORLD_PT_mentalray_sky_sun(_Base, bpy.types.Panel):
    bl_label = "Sun"
    bl_parent_id = "WORLD_PT_mentalray_sky"
    bl_space_type = 'PROPERTIES'
    bl_region_type = 'WINDOW'
    bl_context = "world"

    @classmethod
    def poll(cls, context):
        w = context.scene.world
        return context.engine == ENGINE and w is not None and w.mentalray.use_sky

    def draw_header(self, context):
        self.layout.prop(context.scene.world.mentalray, "sun_enable", text="")

    def draw(self, context):
        p = context.scene.world.mentalray
        col = self.layout.column()
        col.active = p.sun_enable
        col.prop(p, "sun_object")
        if p.sun_object is None:
            col.prop(p, "sun_azimuth")
            col.prop(p, "sun_elevation")
        col.prop(p, "sun_size")
        col.prop(p, "sun_glow")
        col.prop(p, "sun_color")


class WORLD_PT_mentalray_sky_clouds(_Base, bpy.types.Panel):
    bl_label = "Clouds"
    bl_parent_id = "WORLD_PT_mentalray_sky"
    bl_space_type = 'PROPERTIES'
    bl_region_type = 'WINDOW'
    bl_context = "world"

    @classmethod
    def poll(cls, context):
        w = context.scene.world
        return context.engine == ENGINE and w is not None and w.mentalray.use_sky

    def draw_header(self, context):
        self.layout.prop(context.scene.world.mentalray, "clouds", text="")

    def draw(self, context):
        p = context.scene.world.mentalray
        col = self.layout.column()
        col.active = p.clouds
        col.prop(p, "cloud_coverage")
        col.prop(p, "cloud_scale")
        col.prop(p, "cloud_softness")
        col.prop(p, "cloud_color")
        col.prop(p, "cloud_shade")
        col.prop(p, "cloud_seed")


class WORLD_PT_mentalray_sky_stars(_Base, bpy.types.Panel):
    bl_label = "Stars"
    bl_parent_id = "WORLD_PT_mentalray_sky"
    bl_space_type = 'PROPERTIES'
    bl_region_type = 'WINDOW'
    bl_context = "world"
    bl_options = {'DEFAULT_CLOSED'}

    @classmethod
    def poll(cls, context):
        w = context.scene.world
        return context.engine == ENGINE and w is not None and w.mentalray.use_sky

    def draw_header(self, context):
        self.layout.prop(context.scene.world.mentalray, "stars", text="")

    def draw(self, context):
        p = context.scene.world.mentalray
        col = self.layout.column()
        col.active = p.stars
        col.prop(p, "star_density")
        col.prop(p, "star_brightness")


# Order matters: panels in one tab are drawn in registration order.
classes = (
    MENTALRAY_OT_apply_period_color,
    MENTALRAY_OT_assign_materials,
    RENDER_PT_mentalray_sampling,
    RENDER_PT_mentalray_trace,
    RENDER_PT_mentalray_scene,
    RENDER_PT_mentalray_output,
    MATERIAL_PT_mentalray_slots,
    MATERIAL_PT_mentalray,
    MATERIAL_PT_mentalray_color_tex,
    MATERIAL_PT_mentalray_bump,
    DATA_PT_mentalray_light,
    WORLD_PT_mentalray_sky,
    WORLD_PT_mentalray_sky_colors,
    WORLD_PT_mentalray_sky_sun,
    WORLD_PT_mentalray_sky_clouds,
    WORLD_PT_mentalray_sky_stars,
)


def _compat_panels():
    """Built-in panels (object, camera, mesh data, output format, colour management, ...) that should also
    appear for this engine. Panels we replace with our own (materials, worlds, lights) are skipped so
    nothing is drawn twice. Which engine ids exist differs between Blender versions, so look for any of
    the basic ones."""
    basic = {'BLENDER_RENDER', 'BLENDER_WORKBENCH'}
    out = []
    for panel in bpy.types.Panel.__subclasses__():
        name = panel.__name__
        compat = getattr(panel, 'COMPAT_ENGINES', None)
        if not compat or not (compat & basic):
            continue
        if name in {'VIEWLAYER_PT_filter', 'VIEWLAYER_PT_layer_passes'}:
            continue
        if name.startswith(('MATERIAL_PT_', 'WORLD_PT_', 'RENDER_PT_mentalray')):
            continue
        if name.startswith('DATA_PT_') and 'light' in name.lower():
            continue
        out.append(panel)
    return out


def register():
    for c in classes:
        bpy.utils.register_class(c)
    for panel in _compat_panels():
        panel.COMPAT_ENGINES.add(ENGINE)


def unregister():
    for panel in bpy.types.Panel.__subclasses__():
        compat = getattr(panel, 'COMPAT_ENGINES', None)
        if compat and ENGINE in compat and not panel.__name__.startswith(
                ('MATERIAL_PT_mentalray', 'WORLD_PT_mentalray', 'RENDER_PT_mentalray', 'DATA_PT_mentalray')):
            compat.discard(ENGINE)
    for c in reversed(classes):
        bpy.utils.unregister_class(c)
