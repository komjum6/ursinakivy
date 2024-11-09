import json
import mmap
import numpy as np
from ursina import *
from OpenGL.GL import glReadPixels, GL_RGBA, GL_UNSIGNED_BYTE
import time

# Initialize the Ursina application in offscreen mode
app = Ursina(window_type='offscreen', size=(1000, 800))

# Create a ground plane for the player to stand on
ground = Entity(model='plane', scale=16, color=color.brown, position=(4, 0, 4), collider='box')

# Add some extra cubes at various positions for decoration
for pos in [(2, 0, 2), (6, 0, 6), (2, 0, 6), (6, 0, 2)]:
    Entity(position=Vec3(*pos), model='cube', color=color.green)

# Load the GLTF model
skull_entity = Entity( 
    model=load_model('assets/map/skull_me_no_eyes_retopo_colored.obj'), 
    texture=load_texture('assets/map/texture273.png'), 
    position=Vec3(4, 2, 4),
    scale=(1,1,1)
)

# Add ambient and directional lighting
ambient_light = AmbientLight()
ambient_light.color = color.rgb(150, 150, 150)

directional_light = DirectionalLight()
directional_light.look_at(Vec3(-1, -1, -1))

# Memory-mapped files for screenshot data, key states, and mouse position
frame_mm = mmap.mmap(-1, 1000 * 800 * 4, access=mmap.ACCESS_WRITE, tagname='UrsinaMMap')
key_mm = mmap.mmap(-1, 2048, access=mmap.ACCESS_READ, tagname='KeyMap')
mouse_mm = mmap.mmap(-1, 64, access=mmap.ACCESS_READ, tagname='MouseMap')

# Custom controller to handle first-person movement and camera behavior
class CustomController(Entity):
    def __init__(self, entity, **kwargs):
        super().__init__(**kwargs)
        self.entity = entity
        self.speed = 5
        self.height = 1.1  # Adjusted to reduce the bounding box height
        #self.entity.rotation_x = 0
        #self.entity.rotation_y = 0
        self.mouse_sensitivity = Vec2(0.2, 0.2)
        self.zoom_level = 10
        self.zoom_speed = 2.5
        self.mouse_pos = {'x': 0, 'y': 0, 'scroll_x': 0, 'scroll_y': 0}
        self.prev_mouse_pos = {'x': 0, 'y': 0}
        camera.position = self.entity.position - self.entity.forward * self.zoom_level + Vec3(0, self.height + 2, 0)
        self.v_pressed = False
        self.camera_angle = 0  # Initialize the camera angle for rotation around the object

        # Key states initialization
        self.key_states = {chr(i): False for i in range(32, 127)}
        self.key_states.update({
            'tab': False, 'shift': False, 'ctrl': False, 'alt': False,
            'capslock': False, 'backspace': False, 'enter': False,
            'space': False, 'left': False, 'right': False, 'up': False, 'down': False,
            'esc': False, 'del': False
        })

        # Initial camera setup
        self.update_camera_position()

        # Gravity and jump settings
        self.gravity = 0.5
        self.grounded = False
        self.jump_velocity = 0.2  # Initial velocity for jump
        self.vertical_velocity = 0  # Tracks upward/downward movement

        # Collision settings
        self.traverse_target = scene
        self.ignore_list = [self]
        for key, value in kwargs.items():
            setattr(self, key, value)

        # Ground check
        self.check_ground()

    def update_camera_position(self):
        """Updates the camera to follow the player from behind at a certain zoom level."""

        # Check if the 'v' key is pressed
        v = self.key_states.get('v')
        
        # Toggle the v_pressed variable based on the key state
        if v and not self.v_pressed:
            self.v_pressed = True
        elif not v and self.v_pressed:
            self.v_pressed = False
        
        if self.v_pressed:
            scroll_x = self.mouse_pos.get('scroll_x', 0)
            
            # Update the camera angle based on scroll_x input
            self.camera_angle += scroll_x * 0.01  # Adjust the multiplier as needed
            
            # Compute the rotated camera position based on camera_angle and zoom_level
            x_offset = self.zoom_level * math.cos(self.camera_angle)
            z_offset = self.zoom_level * math.sin(self.camera_angle)
            
            # Calculate the target position for the camera
            target_position = self.entity.position + Vec3(x_offset, self.height + 2, z_offset)
            
            # Determine the final camera position based on scroll_x input
            if scroll_x != 0:
                # Linearly interpolate the camera's position towards the target position
                camera.position = lerp(camera.position, target_position, 0.1)  # Adjust the lerp factor as needed
                camera.position = Vec3(camera.position.x - self.entity.forward.x / self.zoom_level, camera.position.y, camera.position.z)
        else:
            camera.position = self.entity.position - self.entity.forward * self.zoom_level + Vec3(0, self.height + 2, 0)
        
        # Update the camera's rotation to look at the entity
        camera.rotation = Vec3(0, 0, 0)
        camera.look_at(self.entity.position)

    def check_ground(self):
        """Ensure the player is above the ground level."""
        ray = raycast(self.entity.world_position + Vec3(0, self.height, 0), Vec3(0, -1, 0), traverse_target=self.traverse_target, ignore=self.ignore_list)
        if ray.hit:
            self.entity.y = max(ray.world_point.y + self.height, self.entity.y)
            self.grounded = ray.distance <= self.height + 0.1
        else:
            self.grounded = False

    def is_valid_json(self, data):
        try:
            parsed = json.loads(data)
            if all(k in parsed for k in self.key_states.keys()):
                return parsed
        except ValueError:
            pass
        return None

    def read_mouse_pos(self):
        mouse_mm.seek(0)
        mouse_data = mouse_mm.read(64).decode('utf-8').strip('\x00')
        try:
            self.mouse_pos = json.loads(mouse_data)
        except json.JSONDecodeError:
            pass

    def update(self):
        key_mm.seek(0)
        key_data = key_mm.read(2048).decode('utf-8').strip('\x00')
        new_key_states = self.is_valid_json(key_data)
        if new_key_states:
            self.key_states = new_key_states

        self.read_mouse_pos()

        # Update zoom based on scroll value
        #scroll_x = self.mouse_pos.get('scroll_x', 0)
        scroll_y = self.mouse_pos.get('scroll_y', 0)
        if scroll_y != 0:  # Apply zoom level changes only if there is a scroll input
            self.zoom_level = max(5, min(15, self.zoom_level - scroll_y * (self.zoom_speed / 100)))
            print(self.zoom_level)
            self.update_camera_position()
            #self.mouse_pos['scroll'] = 0  # Reset scroll after applying zoom to prevent continuous zoom

        mouse_dx = self.mouse_pos['x'] - self.prev_mouse_pos['x']
        mouse_dy = self.mouse_pos['y'] - self.prev_mouse_pos['y']
        self.prev_mouse_pos = self.mouse_pos

        # Movement controls
        direction = Vec3(0, 0, 0)
        if self.key_states.get('w'):
            direction += self.entity.forward
        if self.key_states.get('s'):
            direction -= self.entity.forward
        if self.key_states.get('a'):
            direction -= self.entity.right
        if self.key_states.get('d'):
            direction += self.entity.right

        self.entity.position += direction * self.speed * time.dt

        # Apply jump
        if self.key_states.get('space') and self.grounded:
            self.vertical_velocity = self.jump_velocity
            self.grounded = False

        # Apply gravity
        self.vertical_velocity -= self.gravity * time.dt
        self.entity.y += self.vertical_velocity

        # Check if we've landed
        self.check_ground()
        if self.grounded and self.vertical_velocity < 0:
            self.vertical_velocity = 0  # Reset vertical velocity upon landing

        # Mouse look controls
        self.entity.rotation_y += mouse_dx * self.mouse_sensitivity[0]
        self.entity.rotation_x = clamp(self.entity.rotation_x - mouse_dy * self.mouse_sensitivity[1], -45, 45)
        self.update_camera_position()

controller = CustomController(skull_entity, gravity=True)

last_time = time.time()
screenshot_interval = 1 / 60

def save_screenshot(mm, width, height):
    app.graphicsEngine.renderFrame()
    pixels = np.zeros((height, width, 4), dtype=np.uint8)
    glReadPixels(0, 0, width, height, GL_RGBA, GL_UNSIGNED_BYTE, pixels)
    pixels = np.flipud(pixels)
    mm.seek(0)
    mm.write(pixels.flatten())

def update():
    controller.update()
    global last_time
    current_time = time.time()
    if current_time - last_time >= screenshot_interval:
        save_screenshot(frame_mm, 1000, 800)
        last_time = current_time

app.update = update
app.run()
frame_mm.close()
key_mm.close()
mouse_mm.close()
