"""Plain-data scene description for the standalone renderer core (no bpy)."""
from dataclasses import dataclass, field
import numpy as np

# Shader models (Softimage 3D style)
CONSTANT, LAMBERT, PHONG, BLINN = 0, 1, 2, 3


@dataclass
class Texture:
    kind: str = "checker"            # 'checker', 'image' or 'noise'
    image: object = None             # (H, W, 3) float array, row 0 = v 0 (Blender convention)
    color_a: tuple = (1.0, 1.0, 1.0)  # low end of a noise ramp / checker colour 1
    color_b: tuple = (0.0, 0.0, 0.0)  # high end of a noise ramp / checker colour 2
    mapping: str = "UV"              # 'UV', 'OBJECT' (solid, sticks to the object) or 'WORLD'
    repeat_u: float = 1.0            # image / UV-checker tiling
    repeat_v: float = 1.0
    scale: float = 4.0               # noise and solid-checker frequency
    noise_type: str = "FRACTAL"      # FRACTAL, TURBULENCE, MARBLE, WOOD, CELLS
    octaves: int = 4
    lacunarity: float = 2.0
    gain: float = 0.5
    distortion: float = 2.0          # marble / wood turbulence amount
    seed: int = 0
    feature_uv: float = 0.0          # image textures: size of one feature in UV units (0 = unknown); sets bump scale


@dataclass
class Material:
    name: str = "default"
    shader: int = PHONG
    diffuse: tuple = (0.8, 0.8, 0.8)
    ambient: tuple = (1.0, 1.0, 1.0)     # multiplies the scene ambient colour
    specular: tuple = (1.0, 1.0, 1.0)
    shininess: float = 30.0               # specular exponent
    reflectivity: float = 0.0
    transparency: float = 0.0
    ior: float = 1.0
    emission: tuple = (0.0, 0.0, 0.0)
    texture: object = None
    texture_mix: float = 1.0
    bump: object = None                   # Texture used as a height field
    bump_strength: float = 0.5


@dataclass
class Light:
    kind: str = "point"                   # point | spot | directional | area
    position: tuple = (0.0, 0.0, 0.0)
    direction: tuple = (0.0, 0.0, -1.0)   # direction the light points
    color: tuple = (1.0, 1.0, 1.0)        # already multiplied by intensity
    falloff: int = 0                      # exponent: 0 none, 1 linear, 2 quadratic
    falloff_ref: float = 10.0             # distance at which intensity is unattenuated
    cone: float = 0.7854                  # spot outer half-angle (radians)
    blend: float = 0.15                   # spot soft edge fraction
    area_u: tuple = (1.0, 0.0, 0.0)       # half-extent vectors of a rectangular area light
    area_v: tuple = (0.0, 1.0, 0.0)
    area_samples: tuple = (4, 4)
    shadows: bool = True


@dataclass
class Camera:
    origin: tuple = (0.0, -5.0, 1.0)
    right: tuple = (1.0, 0.0, 0.0)
    up: tuple = (0.0, 0.0, 1.0)
    forward: tuple = (0.0, 1.0, 0.0)
    tan_x: float = 0.5
    tan_y: float = 0.375
    shift_x: float = 0.0                  # NDC offsets
    shift_y: float = 0.0
    ortho: bool = False
    ortho_x: float = 1.0                  # half extents for orthographic cameras
    ortho_y: float = 0.75


@dataclass
class Settings:
    width: int = 320
    height: int = 240
    min_samples: int = 0
    max_samples: int = 2
    contrast: tuple = (0.1, 0.1, 0.1, 0.1)
    filter: str = "box"
    filter_size: float = 1.0
    bucket_size: int = 32
    trace_reflect: int = 2
    trace_refract: int = 4
    trace_sum: int = 6
    shadow_depth: int = 4
    clamp: bool = True
    transparent_film: bool = False
    fog: bool = False
    fog_color: tuple = (0.5, 0.5, 0.5)
    fog_density: float = 0.02


@dataclass
class Scene:
    camera: Camera = field(default_factory=Camera)
    lights: list = field(default_factory=list)
    materials: list = field(default_factory=lambda: [Material()])
    ambient: tuple = (0.0, 0.0, 0.0)      # no ambient light by default
    background: tuple = (0.0, 0.0, 0.0)
    sky: object = None                    # core.sky.Sky or None; seen by camera and reflected/refracted rays
    # Geometry, one row per triangle
    tri_pos: np.ndarray = field(default_factory=lambda: np.zeros((0, 3, 3)))
    tri_nrm: np.ndarray = field(default_factory=lambda: np.zeros((0, 3, 3)))
    tri_uv: np.ndarray = field(default_factory=lambda: np.zeros((0, 3, 2)))
    tri_mat: np.ndarray = field(default_factory=lambda: np.zeros((0,), dtype=np.int64))
    # World -> object transforms ((K, 3, 4)) and which one each triangle uses, so solid textures
    # with 'OBJECT' mapping stay attached to the object. Empty = treat object space as world space.
    xforms: np.ndarray = field(default_factory=lambda: np.zeros((0, 3, 4)))
    tri_xf: np.ndarray = field(default_factory=lambda: np.zeros((0,), dtype=np.int64))
