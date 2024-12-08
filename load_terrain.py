import json
import numpy as np
from ursina import *
from PIL import Image
import time
import os
from perlin_noise import PerlinNoise
import random
from random import randint, uniform
from noise import pnoise2
import matplotlib.pyplot as plt
from io import BytesIO
import mmap

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

def create_texture_atlas(texture_paths):
    """
    Create a texture atlas dynamically from a list of textures.

    Args:
        texture_paths (dict): Dictionary of texture names and their file paths.

    Returns:
        tuple: Texture atlas and texture regions for UV mapping.
    """
    atlas_path = './assets/map/texture_atlas.png'

    # Check if the texture atlas already exists
    if os.path.exists(atlas_path):
        print(f"Loading existing texture atlas from {atlas_path}")
        texture_atlas = Image.open(atlas_path).convert('RGBA')
        atlas_height = texture_atlas.height
        target_size = 1024  # Same as the size used during atlas creation
        texture_regions = {
            key: (
                0, i * target_size / atlas_height,
                1, (i + 1) * target_size / atlas_height
            )
            for i, key in enumerate(texture_paths.keys())
        }
        return texture_atlas, texture_regions

    # If the atlas doesn't exist, generate it
    print(f"Creating new texture atlas at {atlas_path}")
    textures = {
        name: Image.open(path).convert('RGBA') for name, path in texture_paths.items()
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

def load_terrain(positions, sizes, position, scale, texture_paths):
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
    texture_atlas, texture_regions = create_texture_atlas(texture_paths)
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

def apply_warp(point, warp_pts):
    """Apply perspective warp to a point based on the warp points, including the z-coordinate."""
    x, y = point

    # Calculate weights for bilinear interpolation
    p1, p2, p3, p4 = warp_pts
    f1 = (1 - x) * (1 - y)
    f2 = x * (1 - y)
    f3 = x * y
    f4 = (1 - x) * y

    # Interpolate the point based on warp points
    warped_x = f1 * p1[0] + f2 * p2[0] + f3 * p3[0] + f4 * p4[0]
    warped_y = f1 * p1[1] + f2 * p2[1] + f3 * p3[1] + f4 * p4[1]
    warped_z = f1 * p1[2] + f2 * p2[2] + f3 * p3[2] + f4 * p4[2]

    return warped_x, warped_y, warped_z

def load_terrain_warp(positions, sizes, position, scale, warp_points=None):
    """
    Enhanced terrain loading function with perspective warping.
    
    Args:
        positions (list): List of (x, y, z) positions for each terrain section
        sizes (list): List of (width, height, depth) sizes for each section
        position (tuple): Global position offset for the entire terrain
        scale (float): Global scale factor for the entire terrain
        warp_points (list, optional): Optional list of 4 points for perspective warping
    
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

    # Prepare warp points if not provided
    if warp_points is None:
        warp_points = [
            (0, 0, 0),  # top-left
            (1, 0, 0),  # top-right
            (1, 1, 0),  # bottom-right
            (0, 1, 0)   # bottom-left
        ]

    for section in sections:
        pos = section['position']
        size = section['size']
        uv = texture_regions[section['type']]

        # Apply global scale to position and size
        scaled_pos = pos * scale + Vec3(*position)
        scaled_size = size * scale

        # Create vertices for this section with warping
        original_vertices = [
            Vec3(pos.x, pos.y, pos.z),
            Vec3(pos.x + size.x, pos.y, pos.z),
            Vec3(pos.x + size.x, pos.y, pos.z + size.z),
            Vec3(pos.x, pos.y, pos.z + size.z),
        ]

        # Warp vertices based on warp points
        warped_vertices = []
        for i, vert in enumerate(original_vertices):
            # Normalize vertex position relative to section
            norm_x = (vert.x - pos.x) / size.x
            norm_y = (vert.z - pos.z) / size.z

            # Apply warp
            warped_x, warped_y, warped_z = apply_warp((norm_x, norm_y), warp_points)

            # Denormalize back to world space
            warped_vert = Vec3(
                pos.x + warped_x * size.x,
                vert.y + warped_z * size.y,  # Apply warping to the y-coordinate as well
                pos.z + warped_y * size.z
            )
            warped_vertices.append(warped_vert)

        vertices.extend(warped_vertices)

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

        # Create collision entity with the same warped vertices
        collision_entity = Entity(
            model=Mesh(
                vertices=warped_vertices,
                triangles=[0, 1, 2, 0, 2, 3],
                uvs=new_uvs
            ),
            position=floor.position,  # Use the same position as floor
            scale=floor.scale,  # Use the same scale as floor
            color=color.rgba(1, 1, 1, 0),  # Transparent by default
            collider='mesh',
            visible=False  # Set to True for debugging; change to False for production
        )
        entities.append(collision_entity)

    # Update floor mesh
    floor.model.vertices = vertices
    floor.model.uvs = uvs
    floor.model.triangles = triangles
    floor.model.generate()

    return floor, entities
    
def load_terrain_warp(positions, sizes, position, scale, texture_paths, warp_points_list=None):
    """
    Enhanced terrain loading function with perspective warping for each section.
    
    Args:
        positions (list): List of (x, y, z) positions for each terrain section
        sizes (list): List of (width, height, depth) sizes for each section
        position (tuple): Global position offset for the entire terrain
        scale (float): Global scale factor for the entire terrain
        warp_points_list (list, optional): Optional list of warp points for each section
    
    Returns:
        tuple: (floor Entity, list of hover/collision entities)
    """
    # Generate the texture atlas and UV regions
    texture_atlas, texture_regions = create_texture_atlas(texture_paths)
    texture_atlas = Texture(texture_atlas)

    # Define terrain sections dynamically based on texture_paths 
    sections = [] 
    for i, (section_type, texture_path) in enumerate(texture_paths.items()): 
        sections.append({ 
            'type': section_type, 
            'position': Vec3(*positions[i]), 
            'size': Vec3(*sizes[i])
        })
        
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

    # Ensure we have warp points for each section
    if warp_points_list is None:
        warp_points_list = [
            [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0)] for _ in sections
        ]

    for section, warp_points in zip(sections, warp_points_list):
        pos = section['position']
        size = section['size']
        uv = texture_regions[section['type']]

        # Apply global scale to position and size
        scaled_pos = pos * scale + Vec3(*position)
        scaled_size = size * scale

        # Create vertices for this section with warping
        original_vertices = [
            Vec3(pos.x, pos.y, pos.z),
            Vec3(pos.x + size.x, pos.y, pos.z),
            Vec3(pos.x + size.x, pos.y, pos.z + size.z),
            Vec3(pos.x, pos.y, pos.z + size.z),
        ]

        # Warp vertices based on warp points
        warped_vertices = []
        for i, vert in enumerate(original_vertices):
            # Normalize vertex position relative to section
            norm_x = (vert.x - pos.x) / size.x
            norm_y = (vert.z - pos.z) / size.z

            # Apply warp
            warped_x, warped_y, warped_z = apply_warp((norm_x, norm_y), warp_points)

            # Denormalize back to world space
            warped_vert = Vec3(
                pos.x + warped_x * size.x,
                vert.y + warped_z * size.y,  # Apply warping to the y-coordinate as well
                pos.z + warped_y * size.z
            )
            warped_vertices.append(warped_vert)

        vertices.extend(warped_vertices)

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

        # Create collision entity with the same warped vertices
        collision_entity = Entity(
            model=Mesh(
                vertices=warped_vertices,
                triangles=[0, 1, 2, 0, 2, 3],
                uvs=new_uvs
            ),
            position=floor.position,  # Use the same position as floor
            scale=floor.scale,  # Use the same scale as floor
            color=color.rgba(1, 1, 1, 0),  # Transparent by default
            collider='mesh',
            visible=False  # Set to True for debugging; change to False for production
        )
        entities.append(collision_entity)

    # Update floor mesh
    floor.model.vertices = vertices
    floor.model.uvs = uvs
    floor.model.triangles = triangles
    floor.model.generate()

    return floor, entities

def generate_section_type_map(num_sections_x, num_sections_z, texture_paths, biome_percentages):
    """
    Generate a section type map based on Perlin noise, biomes, and given percentages.

    Args:
        num_sections_x (int): Number of terrain sections along the x-axis.
        num_sections_z (int): Number of terrain sections along the z-axis.
        texture_paths (dict): Dictionary of texture paths.
        biome_percentages (dict): Dictionary mapping each biome type to its percentage.

    Returns:
        dict: A dictionary mapping section coordinates to biome types.
    """
    seed = random.randint(0, 1000)
    noise_scale = 0.1
    section_type_map = {}
    total_sections = num_sections_x * num_sections_z

    # Calculate the exact number of tiles for each biome based on percentages
    biome_quotas = {biome: int(total_sections * percentage) for biome, percentage in biome_percentages.items()}
    
    # Fill quotas with Perlin noise values
    noise_values = []
    for z in range(num_sections_z):
        for x in range(num_sections_x):
            noise_value = pnoise2((x + seed) * noise_scale, (z + seed) * noise_scale)
            noise_values.append((noise_value, (x, z)))
    
    # Sort the tiles by noise value
    noise_values.sort(key=lambda nv: nv[0])
    
    # Assign biomes based on sorted noise values and quotas
    index = 0
    for biome, quota in biome_quotas.items():
        for _ in range(quota):
            if index < len(noise_values):
                _, coord = noise_values[index]
                section_type_map[coord] = biome
                index += 1

    # In case there are any remaining tiles not assigned due to rounding,
    # assign the remaining biomes in a fair way
    remaining_biomes = list(biome_quotas.keys())
    while index < len(noise_values):
        _, coord = noise_values[index]
        biome = remaining_biomes[index % len(remaining_biomes)]
        section_type_map[coord] = biome
        index += 1

    return section_type_map

def generate_and_load_terrain(self, num_sections_x, num_sections_z, biome_percentages, tile_size=(4, 1, 4), position=(0, 0, 0), 
                              scale=1.0, texture_paths=None, max_height=10, towns=3, gold_mine_chance=0.02, minimap_path="./output/terrain_visualization.png", debug=True):
    """
    Generate terrain positions, sizes, and a floor with hover/collision entities, including towns, gold mines, and roads.

    Args:
        num_sections_x (int): Number of terrain sections along the x-axis.
        num_sections_z (int): Number of terrain sections along the z-axis.
        biome_percentages (dict): Percentage of biomes for each section type.
        tile_size (tuple): Size of each tile (width, height, depth).
        position (tuple): Position offset for the entire terrain.
        scale (float): Global scale factor for the terrain.
        texture_paths (dict): Dictionary of texture paths mapped to section types.
        max_height (float): Maximum height variation.
        towns (int): Number of towns.
        gold_mine_chance (float): Chance of a gold mine spawning.
        debug (bool): If True, generates debug outputs (e.g., logs and graphs).

    Returns:
        tuple: (floor Entity, list of hover/collision entities, positions)
    """
    self.terrain_start_x = position[0]
    self.terrain_start_z = position[2]

    noise = PerlinNoise(octaves=4, seed=random.randint(0, 1000))
    if isinstance(tile_size, int):
        tile_size = (tile_size, 1, tile_size)

    if isinstance(scale, (int, float)):
        scale = Vec3(scale, scale, scale)

    # Prepare output directories for debugging
    if debug:
        os.makedirs("./output", exist_ok=True)
        cactus_log_path = "./output/cactus_log.txt"

    # Generate section type map
    section_type_map = generate_section_type_map(num_sections_x, num_sections_z, texture_paths, biome_percentages)

    # Generate height map
    height_map = [[
        noise([x * scale.x, z * scale.z]) * max_height for x in range(num_sections_x + 1)
    ] for z in range(num_sections_z + 1)]

    # Function to smooth heights around a plateau
    def smooth_heights(x, z, plateau_height, radius=2):
        for dz in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                nx, nz = x + dx, z + dz
                if 0 <= nx < num_sections_x + 1 and 0 <= nz < num_sections_z + 1:
                    distance = max(abs(dx), abs(dz))
                    blend_factor = max(0, 1 - distance / (radius + 1))
                    height_map[nz][nx] = (
                        blend_factor * plateau_height +
                        (1 - blend_factor) * height_map[nz][nx]
                    )

    # Place towns
    town_positions = []
    for _ in range(towns):
        while True:
            town_x = random.randint(0, num_sections_x - 5)
            town_z = random.randint(0, num_sections_z - 5)
            if all(abs(town_x - tx) >= 5 and abs(town_z - tz) >= 5 for tx, tz in town_positions):
                break
        town_positions.append((town_x, town_z))
        plateau_height = max(
            height_map[z][x] for z in range(town_z, town_z + 5) for x in range(town_x, town_x + 5)
        )
        for z in range(town_z, town_z + 5):
            for x in range(town_x, town_x + 5):
                height_map[z][x] = plateau_height
                section_type_map[(x, z)] = 'pavement2'
        smooth_heights(town_x + 2, town_z + 2, plateau_height, radius=3)

    # Place gold mines
    gold_mine_positions = []
    for z in range(num_sections_z):
        for x in range(num_sections_x):
            if random.random() < gold_mine_chance:
                gold_mine_positions.append((x, z))
                plateau_height = height_map[z][x]
                height_map[z][x] = plateau_height
                section_type_map[(x, z)] = 'pavement2'
                smooth_heights(x, z, plateau_height, radius=2)

    # Create roads connecting towns and gold mines
    def add_road(x1, z1, x2, z2):
        for x in range(min(x1, x2), max(x1, x2) + 1):
            section_type_map[(x, z1)] = 'pavement'
        for z in range(min(z1, z2), max(z1, z2) + 1):
            section_type_map[(x2, z)] = 'pavement'
        smooth_heights(x1, z1, height_map[z1][x1], radius=2)

    for i in range(len(town_positions) - 1):
        x1, z1 = town_positions[i]
        x2, z2 = town_positions[i + 1]
        add_road(x1 + 2, z1 + 2, x2 + 2, z2 + 2)

    for gx, gz in gold_mine_positions:
        nearest_town = min(town_positions, key=lambda t: abs(t[0] - gx) + abs(t[1] - gz))
        add_road(gx, gz, nearest_town[0] + 2, nearest_town[1] + 2)

    # Prepare entity lists
    positions = []
    hover_entities = []
    cacti_positions = []

    # Default texture path
    if texture_paths is None:
        texture_paths = {'default': './assets/map/Pablo_img.jpg'}

    # Create texture atlas
    texture_atlas, texture_regions = create_texture_atlas(texture_paths)
    texture_atlas = Texture(texture_atlas)

    # Create floor entity
    floor = Entity(
        model=Mesh(),
        texture=texture_atlas,
        scale=scale,
        position=Vec3(*position)
    )

    vertices = []
    uvs = []
    triangles = []
    vertex_index = 0

    for z in range(num_sections_z):
        for x in range(num_sections_x):
            pos_x = x * tile_size[0] * scale.x + position[0]
            pos_z = z * tile_size[2] * scale.z + position[2]
            vert_heights = [
                height_map[z][x],
                height_map[z][x + 1],
                height_map[z + 1][x + 1],
                height_map[z + 1][x]
            ]
            pos_y = sum(vert_heights) / 4
            positions.append((pos_x, pos_y, pos_z))

            warped_vertices = [
                Vec3(pos_x, vert_heights[0] * scale.y, pos_z),
                Vec3(pos_x + tile_size[0] * scale.x, vert_heights[1] * scale.y, pos_z),
                Vec3(pos_x + tile_size[0] * scale.x, vert_heights[2] * scale.y, pos_z + tile_size[2] * scale.z),
                Vec3(pos_x, vert_heights[3] * scale.y, pos_z + tile_size[2] * scale.z),
            ]

            vertices.extend(warped_vertices)

            # Texture mapping
            uv = texture_regions[section_type_map.get((x, z), 'default')]
            new_uvs = [
                Vec2(uv[0], uv[1]),
                Vec2(uv[2], uv[1]),
                Vec2(uv[2], uv[3]),
                Vec2(uv[0], uv[3]),
            ]
            uvs.extend(new_uvs)
            triangles.extend([
                vertex_index, vertex_index + 1, vertex_index + 2,
                vertex_index, vertex_index + 2, vertex_index + 3
            ])
            vertex_index += 4

            # Hover entity
            hover_entity = Entity(
                model=Mesh(
                    vertices=warped_vertices,
                    triangles=[0, 1, 2, 0, 2, 3],
                    uvs=new_uvs
                ),
                position=floor.position,
                scale=floor.scale,
                color=color.rgba(1, 1, 1, 0),
                collider='mesh',
                visible=False
            )
            hover_entities.append(hover_entity)

            pos_x = x * tile_size[0] * scale.x + position[0]
            pos_z = z * tile_size[2] * scale.z + position[2]

            # Cactus placement (only on desert tiles)
            if section_type_map.get((x, z)) == 'Pablo_img' and random.random() < 0.3:
                center_x = pos_x + (tile_size[0] * scale.x / 2)
                center_z = pos_z + (tile_size[2] * scale.z / 2)
                
                # Compute height at the center using bilinear interpolation
                def bilinear_interpolate(heights, x_ratio, z_ratio):
                    bottom_edge = heights[0] * (1 - x_ratio) + heights[1] * x_ratio
                    top_edge = heights[3] * (1 - x_ratio) + heights[2] * x_ratio
                    return bottom_edge * (1 - z_ratio) + top_edge * z_ratio

                # Clamp and smooth heights
                smoothed_heights = [
                    max(min(h, max_height * scale.y), -max_height * scale.y) for h in vert_heights
                ]
                center_y = bilinear_interpolate(smoothed_heights, 0.5, 0.5)

                cacti_positions.append((center_x, center_y, center_z))
                
                # Debug logging
                if debug:
                    with open(cactus_log_path, "a") as cactus_log_file:
                        cactus_log_file.write(f"Cactus: Tile ({x}, {z}), Pos ({center_x:.2f}, {center_y:.2f}, {center_z:.2f})\n")

                # Load cacus model
                cactus_entity = Entity(
                    model=load_model('assets/map/Tall_cactus_no_flo_obj/0dddc35f4b62_Tall_cactus__no_flo.obj'),
                    texture=load_texture('assets/map/Tall_cactus_no_flo_obj/0dddc35f4b62_Tall_cactus__no_flo_texture_kd.jpg'),
                    position=Vec3(center_x, center_y, center_z),
                    scale=Vec3(
                        random.uniform(0.8, 1.5),
                        random.uniform(0.8, 1.5),
                        random.uniform(0.8, 1.5)
                    ),
                    rotation_y=random.uniform(0, 360)
                )              

    # Update floor mesh
    floor.model.vertices = vertices
    floor.model.uvs = uvs
    floor.model.triangles = triangles
    floor.model.generate()

    # Optional debug visualization
    if debug:
        plt.figure(figsize=(12, 10))
        ax = plt.gca()
        
        # Plot tiles with biome coloring
        for z in range(num_sections_z):
            for x in range(num_sections_x):
                biome = section_type_map.get((x, z), "default")
                color_new = "yellow" if biome == "Pablo_img" else "green"
                ax.add_patch(plt.Rectangle((x, z), 1, 1, color=color_new, edgecolor="black", alpha=0.7))
        
        # Add cactus markers
        if cacti_positions:
            cactus_x = [cx / tile_size[0] / scale.x for cx, _, _ in cacti_positions]
            cactus_z = [cz / tile_size[2] / scale.z for _, _, cz in cacti_positions]
            plt.scatter(cactus_x, cactus_z, color='red', marker='^', label='Cactus', s=100)
        
        # Create color patches for legend
        from matplotlib.patches import Patch
        handles = [
            Patch(color='yellow', label='Pablo_img'),
            Patch(color='green', label='Other Biome'),
            plt.Line2D([], [], color='red', marker='^', linestyle='None', markersize=10, label='Cactus')
        ]
        
        plt.legend(handles=handles, loc="upper right")
        
        # Add title, labels, and grid
        plt.title("Tile Biomes and Cactus Placement")
        plt.xlabel("X-axis (Tiles)")
        plt.ylabel("Z-axis (Tiles)")
        plt.grid(True, which="both", linestyle="--", linewidth=0.5)
        
        # Ensure x and y axis have same scale
        plt.axis('equal')
        
        # Adjust plot to ensure all elements are visible
        plt.tight_layout()
        
        # Ensure output directory exists
        os.makedirs("./output", exist_ok=True)
        
        # Save the visualization
        plt.savefig(minimap_path, dpi=300)
        print(f"Visualization saved to {minimap_path}")
        plt.close()
        
        # Create a simple map visualization without axis, labels, and grid
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Plot tiles with biome coloring
        for z in range(num_sections_z):
            for x in range(num_sections_x):
                biome = section_type_map.get((x, z), "default")
                color_new = "yellow" if biome == "Pablo_img" else "green"
                ax.add_patch(plt.Rectangle((x, z), 1, 1, color=color_new, edgecolor="black", alpha=0.7))
        
        # Add cactus markers
        if cacti_positions:
            cactus_x = [cx / tile_size[0] / scale.x for cx, _, _ in cacti_positions]
            cactus_z = [cz / tile_size[2] / scale.z for _, _, cz in cacti_positions]
            plt.scatter(cactus_x, cactus_z, color='red', marker='^', label='Cactus', s=100)
        
        # Remove axis
        ax.axis('off')
        ax.set_xticks([]) 
        ax.set_yticks([])
        
        # Remove any extra margins 
        ax.margins(0) 
        
        # Set the limits of the axis to match the plot area (this can be used to zoom in later)
        #ax.set_xlim(0, 10) 
        #ax.set_ylim(0, 10)
        
        # Adjust plot to ensure all elements are visible
        plt.tight_layout()
        
        # Adjust plot margins 
        fig.subplots_adjust(left=0, right=1, top=1, bottom=0)

        # Save plot to a buffer
        buf = BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
        buf.seek(0)
        plt.close()
        
        # Create mmap from buffer size
        buf_size = len(buf.getvalue())
        frame_minimap_mm = mmap.mmap(-1, buf_size, access=mmap.ACCESS_WRITE)
        frame_minimap_mm.write(buf.getvalue())
        frame_minimap_mm.seek(0)

        # Convert mmap to PIL Image
        pil_image = Image.open(frame_minimap_mm)
        pil_image = pil_image.convert('RGBA')

        # Convert PIL Image to texture data compatible with Ursina
        minimap_texture = Texture(pil_image)

        # Create the minimap entity 
        self.minimap = Entity( 
            parent=camera.ui, 
            model='cube', 
            texture=minimap_texture, 
            scale=(0.4, 0.4), # Adjust the scale as needed 
            position=(0.3, 0.3) # Adjust the position to place it in the upper right corner 
        )
        
        # Create the dot entity 
        self.dot = Entity( 
            parent=self.minimap, # Parent it to the minimap so it moves with it 
            model='quad', # Use 'quad' for a simple 2D plane 
            color=color.white, 
            scale=(0.02, 0.02), # Adjust scale to make it a dot 
            position=(-0.5, -0.5) # Initial position 
        )
        
        # Clean up mmap
        frame_minimap_mm.close()

    return floor, hover_entities, positions