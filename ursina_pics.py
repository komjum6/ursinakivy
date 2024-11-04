import json
import mmap
import numpy as np
from ursina import Ursina, Entity, color, Vec3, DirectionalLight, AmbientLight
from OpenGL.GL import glReadPixels, GL_RGBA, GL_UNSIGNED_BYTE
import time

# Initialize the Ursina application in offscreen mode
app = Ursina(window_type='offscreen', size=(1000, 800))

# Create a ground plane for the player to stand on
ground = Entity(model='plane', scale=16, color=color.brown, position=(4, 0, 4), collider='box')

# Add some extra cubes at various positions for decoration
for pos in [(2, 0, 2), (6, 0, 6), (2, 0, 6), (6, 0, 2)]:
    Entity(position=Vec3(*pos), model='cube', color=color.green)

# Add ambient and directional lighting
ambient_light = AmbientLight()
ambient_light.color = color.rgb(150, 150, 150)  # Soft ambient light

directional_light = DirectionalLight()
directional_light.look_at(Vec3(-1, -1, -1))  # Diagonal light direction

# Memory-mapped files for screenshot data and key states
frame_mm = mmap.mmap(-1, 1000 * 800 * 4, access=mmap.ACCESS_WRITE, tagname='UrsinaMMap')
key_mm = mmap.mmap(-1, 1024, access=mmap.ACCESS_READ, tagname='KeyMap')

# Custom controller to handle first-person movement without mouse locking
class CustomController:
    def __init__(self, entity):
        self.entity = entity
        self.speed = 5
        # Start with an initial "no movement" state
        self.key_states = {'w': False, 'a': False, 's': False, 'd': False}

    def is_valid_json(self, data):
        """Utility function to check if the data is valid JSON and matches the expected structure."""
        try:
            parsed = json.loads(data)
            if all(k in parsed for k in self.key_states.keys()):
                return parsed
        except ValueError:
            pass
        return None

    def update(self):
        # Read key states from shared memory every frame
        key_mm.seek(0)
        key_data = key_mm.read(1024).decode('utf-8').strip('\x00')
        #print(f"Raw key data: {key_data}")  # Debugging statement

        # Validate and load the key state JSON, else retain the previous state
        new_key_states = self.is_valid_json(key_data)
        if new_key_states:
            self.key_states = new_key_states
            #print(f"Updated key states: {self.key_states}")  # Debugging statement
        else:
            #print("Invalid data detected; retaining last known good key states.")
            pass

        # Movement controls
        if self.key_states.get('w'):
            #print("Moving forward")
            self.entity.position += self.entity.forward * self.speed * time.dt
        if self.key_states.get('s'):
            #print("Moving backward")
            self.entity.position -= self.entity.forward * self.speed * time.dt
        if self.key_states.get('a'):
            #print("Moving left")
            self.entity.position -= self.entity.right * self.speed * time.dt
        if self.key_states.get('d'):
            #print("Moving right")
            self.entity.position += self.entity.right * self.speed * time.dt

# Instantiate the custom controller
player = Entity(model='cube', color=color.orange, position=(4, 2, 4))  # Start above the ground
controller = CustomController(player)

# Screenshot interval setup
last_time = time.time()
screenshot_interval = 1 / 60  # Capture at 60 FPS

def save_screenshot(mm, width, height):
    # Render the frame without scheduling conflicts
    app.graphicsEngine.renderFrame()

    # Read pixels from OpenGL framebuffer
    pixels = np.zeros((height, width, 4), dtype=np.uint8)
    glReadPixels(0, 0, width, height, GL_RGBA, GL_UNSIGNED_BYTE, pixels)

    # Flip the image vertically to correct orientation
    pixels = np.flipud(pixels)

    # Write pixel data to memory-mapped file
    mm.seek(0)
    mm.write(pixels.flatten())

def update():
    # Update player position based on key states
    controller.update()

    # Capture screenshots at intervals
    global last_time
    current_time = time.time()
    if current_time - last_time >= screenshot_interval:
        save_screenshot(frame_mm, 1000, 800)
        last_time = current_time

# Bind the update function to Ursina's update loop
app.update = update

# Run the Ursina application
app.run()
frame_mm.close()
key_mm.close()
