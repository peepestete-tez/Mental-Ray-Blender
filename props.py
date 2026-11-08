import bpy
from bpy.props import (BoolProperty, EnumProperty, FloatProperty, FloatVectorProperty,
                       IntProperty, PointerProperty)

from .core import DEFAULT_FILTER_SIZE, SKY_PRESETS


def _filter_changed(self, context):
    self.filter_size = DEFAULT_FILTER_SIZE.get(self.filter, 1.0)


def _sky_preset_changed(self, context):
    p = SKY_PRESETS.get(self.preset)
    if not p:
        return
    for key, val in p.items():
        setattr(self, key, val)


class MentalRaySceneSettings(bpy.types.PropertyGroup):
    # Sampling (mental ray convention: level n = 4**n samples per pixel,
    # negative levels = one sample per 2**|n| x 2**|n| pixels)
    min_samples: IntProperty(name="Min Samples", default=0, min=-3, max=4,
                             description="Minimum sample level (0 = 1 sample/pixel, 1 = 4, 2 = 16, -1 = 1 per 2x2 pixels)")
    max_samples: IntProperty(name="Max Samples", default=2, min=-3, max=4,
                             description="Maximum sample level reached by adaptive subdivision")
    contrast: FloatVectorProperty(name="Contrast", size=4, default=(0.1, 0.1, 0.1, 0.1), min=0.0, max=1.0,
                                  description="Per-channel (R, G, B, A) colour difference that triggers subdivision")
    filter: EnumProperty(name="Filter", default='box',
                         items=[('box', "Box", ""), ('triangle', "Triangle", ""), ('gaussian', "Gaussian", ""),
                                ('mitchell', "Mitchell", ""), ('lanczos', "Lanczos", "")],
                         update=_filter_changed)
    filter_size: FloatProperty(name="Filter Size", default=1.0, min=0.5, max=8.0,
                               description="Filter width in pixels (box 1, triangle 2, gaussian 3, mitchell/lanczos 4 are the classic values)")
    bucket_size: IntProperty(name="Bucket Size", default=32, min=8, max=256, subtype='PIXEL')

    # Tracing
    trace_reflect: IntProperty(name="Reflection", default=2, min=0, max=16)
    trace_refract: IntProperty(name="Refraction", default=4, min=0, max=16)
    trace_sum: IntProperty(name="Sum", default=6, min=0, max=32,
                           description="Maximum combined reflection + refraction depth")
    shadow_depth: IntProperty(name="Shadow Depth", default=4, min=1, max=16,
                              description="How many transparent surfaces a shadow ray can pass through")

    # Scene
    ambient: FloatVectorProperty(name="Ambient Light", subtype='COLOR', size=3, default=(0.0, 0.0, 0.0), min=0.0, max=1.0,
                                 description="Global ambient light. Black (the default) means no ambient lighting at all; "
                                             "reflections still show the sky")
    light_scale: FloatProperty(name="Light Scale", default=1.0, min=0.0, soft_max=10.0,
                               description="Point/spot/area intensity = Blender watts / 1000 * scale. Sun strength is used as is")
    fog_enable: BoolProperty(name="Fog", default=False)
    fog_color: FloatVectorProperty(name="Fog Color", subtype='COLOR', size=3, default=(0.5, 0.5, 0.5), min=0.0, max=1.0)
    fog_density: FloatProperty(name="Fog Density", default=0.02, min=0.0, soft_max=1.0, precision=4)

    # Output
    display_referred: BoolProperty(
        name="Period-correct Colour", default=True,
        description="Shade directly in display values like a 1990s renderer (no linear workflow). "
                    "Colours are converted to sRGB for shading and back to linear for Blender, so the 'Standard' view transform shows the raw numbers")
    clamp: BoolProperty(name="Clamp Output", default=True, description="Clip colours to 0..1 like an 8-bit pipeline")


class MentalRayTexture(bpy.types.PropertyGroup):
    """One texture slot (used for both the colour and the bump channel of a material)."""
    kind: EnumProperty(name="Type", default='NONE',
                       items=[('NONE', "None", ""),
                              ('NOISE', "Noise", "Solid 3D noise: fractal, turbulence, marble, wood or cells"),
                              ('CHECKER', "Checker", ""),
                              ('IMAGE', "Image", "")])
    noise_type: EnumProperty(name="Noise", default='FRACTAL',
                             items=[('FRACTAL', "Fractal", "Soft cloudy fractal noise"),
                                    ('TURBULENCE', "Turbulence", "Sharp, veiny sum of absolute noise"),
                                    ('MARBLE', "Marble", "Sine stripes warped by turbulence"),
                                    ('WOOD', "Wood", "Concentric rings warped by turbulence"),
                                    ('CELLS', "Cells", "Cellular / stucco / orange-peel")])
    mapping: EnumProperty(name="Mapping", default='OBJECT',
                          items=[('OBJECT', "Object", "Solid texture that sticks to the object (no UVs needed)"),
                                 ('WORLD', "World", "Solid texture fixed in space; the object moves through it"),
                                 ('UV', "UV", "Use the mesh UV coordinates")])
    image: PointerProperty(name="Image", type=bpy.types.Image)
    color_a: FloatVectorProperty(name="Colour 1", subtype='COLOR_GAMMA', size=3, default=(0.1, 0.1, 0.1), min=0.0, max=1.0,
                                 description="Noise: colour at the low end. Checker: first colour")
    color_b: FloatVectorProperty(name="Colour 2", subtype='COLOR_GAMMA', size=3, default=(0.9, 0.9, 0.9), min=0.0, max=1.0,
                                 description="Noise: colour at the high end. Checker: second colour")
    scale: FloatProperty(name="Scale", default=4.0, min=0.001, soft_max=100.0,
                         description="Noise / solid checker frequency (features per unit)")
    octaves: IntProperty(name="Octaves", default=4, min=1, max=10)
    lacunarity: FloatProperty(name="Lacunarity", default=2.0, min=1.1, max=4.0)
    gain: FloatProperty(name="Gain", default=0.5, min=0.1, max=0.95)
    distortion: FloatProperty(name="Distortion", default=2.0, min=0.0, max=20.0,
                              description="How much turbulence warps marble veins / wood rings")
    seed: IntProperty(name="Seed", default=0, min=0, max=100000)
    repeat_u: FloatProperty(name="U Repeat", default=1.0, min=0.01, soft_max=100.0)
    repeat_v: FloatProperty(name="V Repeat", default=1.0, min=0.01, soft_max=100.0)


class MentalRayMaterialSettings(bpy.types.PropertyGroup):
    shader: EnumProperty(name="Shader", default='PHONG',
                         items=[('CONSTANT', "Constant", "Flat colour, no lighting"),
                                ('LAMBERT', "Lambert", "Diffuse only"),
                                ('PHONG', "Phong", "Diffuse + Phong highlight"),
                                ('BLINN', "Blinn", "Diffuse + Blinn highlight")])
    use_viewport_color: BoolProperty(name="Use Viewport Display Colour", default=True,
                                     description="Take the diffuse colour from Material > Viewport Display")
    diffuse: FloatVectorProperty(name="Diffuse", subtype='COLOR_GAMMA', size=3, default=(0.8, 0.8, 0.8), min=0.0, max=1.0)
    ambient: FloatVectorProperty(name="Ambient", subtype='COLOR_GAMMA', size=3, default=(1.0, 1.0, 1.0), min=0.0, max=1.0)
    specular: FloatVectorProperty(name="Specular", subtype='COLOR_GAMMA', size=3, default=(1.0, 1.0, 1.0), min=0.0, max=1.0)
    shininess: FloatProperty(name="Shininess", default=30.0, min=0.0, soft_max=500.0)
    emission: FloatVectorProperty(name="Incandescence", subtype='COLOR_GAMMA', size=3, default=(0.0, 0.0, 0.0), min=0.0, max=1.0)
    reflectivity: FloatProperty(name="Reflectivity", default=0.0, min=0.0, max=1.0)
    transparency: FloatProperty(name="Transparency", default=0.0, min=0.0, max=1.0)
    ior: FloatProperty(name="Refractive Index", default=1.0, min=0.1, max=5.0)
    color_tex: PointerProperty(type=MentalRayTexture)
    texture_mix: FloatProperty(name="Texture Mix", default=1.0, min=0.0, max=1.0,
                               description="How much the colour texture replaces the diffuse colour")
    bump_tex: PointerProperty(type=MentalRayTexture)
    bump_strength: FloatProperty(name="Bump Strength", default=1.0, min=-20.0, max=20.0, soft_min=-3.0, soft_max=3.0,
                                 description="Steepness of the bump relative to the texture's feature size, so it looks "
                                             "the same at any Scale. Negative values invert it")


class MentalRayLightSettings(bpy.types.PropertyGroup):
    falloff: EnumProperty(name="Falloff", default='NONE',
                          items=[('NONE', "None", "No distance attenuation (period default)"),
                                 ('LINEAR', "Linear (1/d)", ""),
                                 ('QUADRATIC', "Quadratic (1/d²)", "")])
    falloff_ref: FloatProperty(name="Reference Distance", default=10.0, min=0.001,
                               description="Distance at which the light is at full intensity")
    shadows: BoolProperty(name="Cast Shadows", default=True)
    area_u: IntProperty(name="U Samples", default=4, min=1, max=16)
    area_v: IntProperty(name="V Samples", default=4, min=1, max=16)


def _sun_poll(self, ob):
    return ob.type == 'LIGHT'


class MentalRaySkySettings(bpy.types.PropertyGroup):
    """90s CGI sky. Seen by the camera and by reflective / refractive surfaces; it does not light the scene."""
    use_sky: BoolProperty(name="Use Sky", default=True,
                          description="Use the procedural sky as background and as the environment that reflections and refractions see")
    preset: EnumProperty(name="Preset", default='DAY', update=_sky_preset_changed,
                         items=[('DAY', "Day", "Classic blue sky with puffy clouds"),
                                ('SUNSET', "Sunset", "Low orange sun"),
                                ('NIGHT', "Night", "Dark sky with stars"),
                                ('VAPOR', "Vapor", "Magenta dusk with stars"),
                                ('CLEAR', "Clear", "Cloudless blue")])
    visible_to_camera: BoolProperty(name="Visible to Camera", default=True,
                                    description="Show the sky as render background. Off keeps the world colour in view but still reflects the sky")
    zenith: FloatVectorProperty(name="Zenith", subtype='COLOR_GAMMA', size=3, default=(0.10, 0.30, 0.75), min=0.0, max=1.0)
    horizon: FloatVectorProperty(name="Horizon", subtype='COLOR_GAMMA', size=3, default=(0.65, 0.80, 0.95), min=0.0, max=1.0)
    ground: FloatVectorProperty(name="Ground", subtype='COLOR_GAMMA', size=3, default=(0.25, 0.22, 0.20), min=0.0, max=1.0)
    gradient_power: FloatProperty(name="Gradient Power", default=0.6, min=0.05, max=4.0,
                                  description="Below 1 pushes the horizon colour higher into the sky")
    sun_enable: BoolProperty(name="Sun", default=True)
    sun_object: PointerProperty(name="Sun Light", type=bpy.types.Object, poll=_sun_poll,
                                description="Optional: take the sun direction from this light (its pointing direction, reversed)")
    sun_azimuth: FloatProperty(name="Azimuth", default=25.0, min=-360.0, max=360.0)
    sun_elevation: FloatProperty(name="Elevation", default=45.0, min=-10.0, max=90.0)
    sun_size: FloatProperty(name="Size", default=2.5, min=0.1, max=20.0, description="Angular radius in degrees")
    sun_glow: FloatProperty(name="Glow", default=0.5, min=0.0, max=3.0)
    sun_color: FloatVectorProperty(name="Sun Colour", subtype='COLOR_GAMMA', size=3, default=(1.0, 0.95, 0.8), min=0.0, max=1.0)
    clouds: BoolProperty(name="Clouds", default=True)
    cloud_coverage: FloatProperty(name="Coverage", default=0.5, min=0.0, max=1.0)
    cloud_scale: FloatProperty(name="Scale", default=2.0, min=0.2, max=20.0)
    cloud_softness: FloatProperty(name="Softness", default=0.35, min=0.01, max=1.0)
    cloud_color: FloatVectorProperty(name="Cloud Colour", subtype='COLOR_GAMMA', size=3, default=(1.0, 1.0, 1.0), min=0.0, max=1.0)
    cloud_shade: FloatVectorProperty(name="Cloud Shade", subtype='COLOR_GAMMA', size=3, default=(0.55, 0.60, 0.72), min=0.0, max=1.0)
    cloud_seed: IntProperty(name="Seed", default=1, min=0, max=100000)
    stars: BoolProperty(name="Stars", default=False)
    star_density: FloatProperty(name="Density", default=0.05, min=0.0, max=0.5)
    star_brightness: FloatProperty(name="Brightness", default=1.0, min=0.0, max=3.0)


classes = (MentalRaySceneSettings, MentalRayTexture, MentalRayMaterialSettings, MentalRayLightSettings, MentalRaySkySettings)


def register():
    for c in classes:
        bpy.utils.register_class(c)
    bpy.types.Scene.mentalray = PointerProperty(type=MentalRaySceneSettings)
    bpy.types.Material.mentalray = PointerProperty(type=MentalRayMaterialSettings)
    bpy.types.Light.mentalray = PointerProperty(type=MentalRayLightSettings)
    bpy.types.World.mentalray = PointerProperty(type=MentalRaySkySettings)


def unregister():
    del bpy.types.World.mentalray
    del bpy.types.Light.mentalray
    del bpy.types.Material.mentalray
    del bpy.types.Scene.mentalray
    for c in reversed(classes):
        bpy.utils.unregister_class(c)
