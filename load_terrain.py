import json
import mmap
import numpy as np
from ursina import *
from OpenGL.GL import glReadPixels, GL_RGBA, GL_UNSIGNED_BYTE
from PIL import Image
import time
import os

def create_texture_atlas():
    """
    Create a texture atlas in memory or load an existing one if it already exists.
    Returns the generated or loaded texture atlas and texture regions for UV mapping.
    """
    atlas_path = './assets/map/texture_atlas.png'

    # Check if the texture atlas already exists
    if os.path.exists(atlas_path):
        print(f"Loading existing texture atlas from {atlas_path}")
        texture_atlas = Image.open(atlas_path).convert('RGBA')
        atlas_height = texture_atlas.height
        target_size = 1024  # Same as the size used during atlas creation
        texture_regions = {
            'room1': (0, 0 / atlas_height, 1, target_size / atlas_height),
            'room2': (0, target_size / atlas_height, 1, 2 * target_size / atlas_height),
            'hallway': (0, 2 * target_size / atlas_height, 1, 3 * target_size / atlas_height),
        }
        return texture_atlas, texture_regions

    # If the atlas doesn't exist, generate it
    print(f"Creating new texture atlas at {atlas_path}")
    textures = {
        'room1': Image.open('./assets/map/pavement.jpg').convert('RGBA'),
        'room2': Image.open('./assets/map/pavement2.jpg').convert('RGBA'),
        'hallway': Image.open('./assets/map/grass.jpg').convert('RGBA')
    }

    target_size = (1024, 1024)
    for key in textures:
        textures[key] = textures[key].resize(target_size, Image.Resampling.LANCZOS)

    atlas_width = target_size[0]
    atlas_height = target_size[1] * len(textures)
    texture_atlas = Image.new('RGBA', (atlas_width, atlas_height))

    current_y = 0
    texture_regions = {}
    for key, texture in textures.items():
        texture_atlas.paste(texture, (0, current_y))
        texture_regions[key] = (
            0, current_y / atlas_height,
            1, (current_y + target_size[1]) / atlas_height
        )
        current_y += target_size[1]

    # Save the texture atlas
    texture_atlas.save(atlas_path)
    return texture_atlas, texture_regions

def load_terrain(positions, sizes, position, scale):
    """
    Enhanced terrain loading function with combined hover/collision entities.
    Args:
        positions (list): List of (x, y, z) positions for each terrain section
        sizes (list): List of (width, height, depth) sizes for each section
        position (tuple): Global position offset for the entire terrain
        scale (float): Global scale factor for the entire terrain
    Returns:
        tuple: (floor Entity, list of hover/collision entities)
    """
    # Generate the texture atlas and UV regions
    texture_atlas, texture_regions = create_texture_atlas()
    texture_atlas = Texture(texture_atlas)

    # Define terrain sections
    sections = [
        {'type': 'room1', 'position': Vec3(*positions[0]), 'size': Vec3(*sizes[0])},
        {'type': 'hallway', 'position': Vec3(*positions[1]), 'size': Vec3(*sizes[1])},
        {'type': 'room2', 'position': Vec3(*positions[2]), 'size': Vec3(*sizes[2])},
    ]

    # Create main floor entity
    floor = Entity(
        model=Mesh(),
        texture=texture_atlas,
        scale=scale,
        position=position
    )

    # Initialize mesh data
    vertices = []
    uvs = []
    triangles = []
    entities = []
    vertex_index = 0

    for section in sections:
        pos = section['position']
        size = section['size']
        uv = texture_regions[section['type']]

        # Apply global scale to position and size
        scaled_pos = pos * scale + Vec3(*position)
        scaled_size = size * scale

        # Create vertices for this section
        new_vertices = [
            Vec3(pos.x, pos.y, pos.z),
            Vec3(pos.x + size.x, pos.y, pos.z),
            Vec3(pos.x + size.x, pos.y, pos.z + size.z),
            Vec3(pos.x, pos.y, pos.z + size.z),
        ]
        vertices.extend(new_vertices)

        # Create UVs with proper mapping
        new_uvs = [
            Vec2(uv[0], uv[1]),  # Bottom left
            Vec2(uv[2], uv[1]),  # Bottom right
            Vec2(uv[2], uv[3]),  # Top right
            Vec2(uv[0], uv[3]),  # Top left
        ]
        uvs.extend(new_uvs)

        # Create triangles
        triangles.extend([
            vertex_index, vertex_index + 1, vertex_index + 2,
            vertex_index, vertex_index + 2, vertex_index + 3
        ])
        vertex_index += 4

        # Create combined hover/collision entity that matches floor section exactly
        entity = Entity(
            model=Mesh(
                vertices=[Vec3(0, 0, 0), Vec3(1, 0, 0), Vec3(1, 0, 1), Vec3(0, 0, 1)],
                triangles=[0, 1, 2, 0, 2, 3],
                uvs=[Vec2(0, 0), Vec2(1, 0), Vec2(1, 1), Vec2(0, 1)]
            ),
            position=scaled_pos,
            scale=scaled_size,
            color=color.rgba(1, 1, 1, 0),  # Transparent by default
            collider='mesh',  # Use mesh collider for precise hovering and collision
            visible=False
        )

        entities.append(entity)

    # Update floor mesh
    floor.model.vertices = vertices
    floor.model.uvs = uvs
    floor.model.triangles = triangles
    floor.model.generate()

    return floor, entities