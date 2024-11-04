import numpy as np
from kivy.app import App
from kivy.uix.anchorlayout import AnchorLayout
from kivy.uix.button import Button
from kivy.uix.image import Image as KivyImage
from kivy.graphics.texture import Texture as KivyTexture
from kivy.clock import Clock
from kivy.core.window import Window
from PIL import Image
import mmap
import subprocess
import json

class KivyUrsinaApp(App):
    def build(self):
        # Initialize the AnchorLayout
        self.layout = AnchorLayout(anchor_x='center', anchor_y='top')

        # Create the button and add it to the top of the layout
        button = Button(text="Kivy Button", size_hint=(0.2, 0.1), pos_hint={'center_x': 0.5, 'top': 1})
        button.bind(on_press=self.on_button_click)
        self.layout.add_widget(button, index=0)

        # Define the width and height of the Ursina screen
        self.width, self.height = 1000, 800

        # Create a Kivy texture to render Ursina's frame
        self.kivy_texture = KivyTexture.create(size=(self.width, self.height), colorfmt='rgba')
        self.image = KivyImage(texture=self.kivy_texture)
        
        # Set the image anchor to the center-bottom and add it to the layout
        self.image.size_hint = (1.0, 1.0)
        self.image.height = self.height
        self.layout.add_widget(self.image, index=1)

        # Start the Ursina process
        self.ursina_process = subprocess.Popen(['python', 'ursina_pics.py'])

        # Access memory-mapped files for frame and key states
        self.frame_mm = mmap.mmap(-1, self.width * self.height * 4, tagname='UrsinaMMap')
        self.key_mm = mmap.mmap(-1, 1024, tagname='KeyMap')

        # Initialize key states for communication
        self.key_states = {'w': False, 's': False, 'a': False, 'd': False}
        
        # Register key events
        Window.bind(on_key_down=self.on_key_down)
        Window.bind(on_key_up=self.on_key_up)

        # Schedule the update to capture Ursina's frame
        Clock.schedule_interval(self.update_ursina_texture, 1 / 60)

        return self.layout

    def on_key_down(self, window, key, *args):
        key_name = Window._system_keyboard.keycode_to_string(key)
        if key_name in self.key_states and not self.key_states[key_name]:  # Detect state change
            self.key_states[key_name] = True
            self.update_keymap()  # Write new state

    def on_key_up(self, window, key, *args):
        key_name = Window._system_keyboard.keycode_to_string(key)
        if key_name in self.key_states and self.key_states[key_name]:  # Detect state change
            self.key_states[key_name] = False
            self.update_keymap()  # Write new state

    def update_keymap(self):
        # Convert key_states to JSON, then encode to 1024-byte fixed length
        key_data = json.dumps(self.key_states).encode('utf-8')
        padded_data = key_data.ljust(1024, b'\x00')  # Pad to ensure consistent size

        # Write key states to shared memory as JSON and flush
        self.key_mm.seek(0)
        self.key_mm.write(padded_data)
        self.key_mm.flush()
        #print(f"Updated key states: {self.key_states}")  # Debugging statement

    def update_ursina_texture(self, *args):
        # Read from memory-mapped file and update texture
        self.frame_mm.seek(0)
        data = self.frame_mm.read(self.width * self.height * 4)
        self.shared_array = np.flipud(np.frombuffer(data, dtype=np.uint8).reshape((self.height, self.width, 4)))
        self.kivy_texture.blit_buffer(self.shared_array.flatten(), colorfmt='rgba', bufferfmt='ubyte')
        self.image.texture = self.kivy_texture
        self.image.canvas.ask_update()

    def on_button_click(self, instance):
        self.update_ursina_texture()
        # Save the current image
        img = Image.fromarray(self.shared_array, 'RGBA')
        img.save('saved_image.png')
        print("Screenshot made")

    def on_stop(self):
        self.ursina_process.terminate()
        self.frame_mm.close()
        self.key_mm.close()

if __name__ == '__main__':
    KivyUrsinaApp().run()
