import os
import json
import mmap
import numpy as np
from kivy.app import App
from kivy.uix.popup import Popup
from kivy.uix.label import Label
from kivy.uix.anchorlayout import AnchorLayout
from kivy.uix.button import Button
from kivy.uix.image import Image as KivyImage
from kivy.graphics.texture import Texture as KivyTexture
from kivy.clock import Clock
from kivy.core.window import Window
from PIL import Image
import pygame
import subprocess
from pynput import mouse

# Initialize Pygame only once
pygame.init()

class KivyUrsinaApp(App):
    def build(self):
        # This changes the cursor visability
        Window.show_cursor = True
    
        self.layout = AnchorLayout(anchor_x='center', anchor_y='top')
        button = Button(text="Kivy Button", size_hint=(0.2, 0.1), pos_hint={'center_x': 0.5, 'top': 1})
        button.bind(on_press=self.on_button_click)
        self.layout.add_widget(button, index=0)

        # Set image display size and create a texture
        self.width, self.height = 1000, 800
        self.kivy_texture = KivyTexture.create(size=(self.width, self.height), colorfmt='rgba')
        self.image = KivyImage(texture=self.kivy_texture)
        self.image.size_hint = (1.0, 1.0)
        self.image.height = self.height
        self.layout.add_widget(self.image, index=1)

        # Start the Ursina process
        self.ursina_process = subprocess.Popen(['python', 'ursina_pics.py'])
        self.frame_mm = mmap.mmap(-1, self.width * self.height * 4, tagname='UrsinaMMap')

        # Memory-mapped files for keyboard and mouse
        self.key_mm = mmap.mmap(-1, 2048, tagname='KeyMap')
        self.mouse_pos_mm = mmap.mmap(-1, 64, tagname='MousePosMap')
        self.mouse_clicked_mm = mmap.mmap(-1, 84, tagname='MouseClickedMap')
        self.mouse_hover_mm = mmap.mmap(-1, 84, tagname='MouseHoverMap')

        # Initialize key states
        self.key_states = {chr(i): False for i in range(32, 127)}
        self.key_states.update({
            'space': False, 'tab': False, 'shift': False, 'ctrl': False, 'alt': False,
            'capslock': False, 'backspace': False, 'enter': False,
            'left': False, 'right': False, 'up': False, 'down': False,
            'esc': False, 'del': False, 'm': False
        })

        # Toggle variable for switching renders
        self.use_pygame_render = False

        # Bind key events
        Window.bind(on_key_down=self.on_key_down)
        Window.bind(on_key_up=self.on_key_up)
        Window.bind(mouse_pos=self.on_mouse_move)
        Window.bind(on_mouse_down=self.on_mouse_click)

        # Use pynput to listen for scroll events
        self.mouse_listener = mouse.Listener(on_scroll=self.on_mouse_wheel)
        self.mouse_listener.start()

        Clock.schedule_interval(self.update_ursina_texture, 1 / 60)
        return self.layout

    def on_key_down(self, window, key, *args):
        key_name = Window._system_keyboard.keycode_to_string(key)
        special_key_map = {
            'spacebar': 'space', 'capslock': 'capslock', 'escape': 'esc',
            'delete': 'del', 'enter': 'enter', 'backspace': 'backspace'
        }
        if key_name in special_key_map:
            key_name = special_key_map[key_name]
        if key_name and key_name in self.key_states and not self.key_states[key_name]:
            self.key_states[key_name] = True
            self.update_keymap()

            # Toggle between Ursina and Pygame rendering on 'p' press
            if key_name == 'p':
                self.use_pygame_render = not self.use_pygame_render  # Toggle state
                print(f"'p' pressed: Toggling render mode to {'Pygame' if self.use_pygame_render else 'Ursina'}")
                self.show_popup(f"'p' pressed: Toggling render mode to {'Pygame' if self.use_pygame_render else 'Ursina'}")
                if self.use_pygame_render:
                    self.run_pygame_render()  # Render Pygame image
                    
            if key_name == 'm':
                print("m")
                
            # Toggle the editor on 'e' press
            if key_name == 'e':
                self.update_keymap()
                self.show_popup("Toggling Editor Mode")

    def on_key_up(self, window, key, *args):
        key_name = Window._system_keyboard.keycode_to_string(key)
        special_key_map = {
            'spacebar': 'space', 'capslock': 'capslock', 'escape': 'esc',
            'delete': 'del', 'enter': 'enter', 'backspace': 'backspace'
        }
        if key_name in special_key_map:
            key_name = special_key_map[key_name]
        if key_name and key_name in self.key_states and self.key_states[key_name]:
            self.key_states[key_name] = False
            self.update_keymap()

    def on_mouse_move(self, window, pos):
        # Get normalized coordinates for the Ursina viewport
        norm_x = pos[0] / self.image.width
        norm_y = pos[1] / self.image.height
        
        # Only mark as hovered if within the Ursina render bounds
        hovered = 0 <= norm_x <= 1 and 0 <= norm_y <= 1
        
        # Prepare the hover data for Ursina
        hover_data = json.dumps({'x_norm': norm_x, 'y_norm': norm_y, 'hovered': hovered}).encode('utf-8')
        padded_data = hover_data.ljust(84, b'\x00')

        # Write hover data to the memory-mapped file
        self.mouse_hover_mm.seek(0)
        self.mouse_hover_mm.write(padded_data)
        self.mouse_hover_mm.flush()
        
        # Write position and scroll data to the memory-mapped file
        mouse_pos_data = json.dumps({'x': pos[0], 'y': pos[1], 'scroll': 0}).encode('utf-8')
        padded_data = mouse_pos_data.ljust(64, b'\x00')
        self.mouse_pos_mm.seek(0)
        self.mouse_pos_mm.write(padded_data)
        self.mouse_pos_mm.flush()

    def on_mouse_wheel(self, x, y, dx, dy):
        mouse_pos_data = json.dumps({
            'x': Window.mouse_pos[0],
            'y': Window.mouse_pos[1],
            'scroll_x': dx,
            'scroll_y': dy,
        }).encode('utf-8')
        padded_data = mouse_pos_data.ljust(64, b'\x00')
        self.mouse_pos_mm.seek(0)
        self.mouse_pos_mm.write(padded_data)
        self.mouse_pos_mm.flush()

    def on_mouse_click(self, *args):
        # Capture click position in Kivy image coordinates
        click_x, click_y = args[1], args[2]
        #print(click_x)
        #print(click_y)
        print("click")

        # Normalize the position based on image dimensions (assuming top-left origin)
        norm_x = click_x / self.image.width
        norm_y = click_y / self.image.height

        # Package normalized coordinates in shared memory format
        click_data = json.dumps({'x_click': norm_x, 'y_click': norm_y, 'clicked': True}).encode('utf-8')
        padded_data = click_data.ljust(84, b'\x00')

        # Write click data to `mouse_mm`
        self.mouse_clicked_mm.seek(0)
        self.mouse_clicked_mm.write(padded_data)
        self.mouse_clicked_mm.flush()

    def update_keymap(self):
        key_data = json.dumps(self.key_states).encode('utf-8')
        padded_data = key_data.ljust(2048, b'\x00')
        self.key_mm.seek(0)
        self.key_mm.write(padded_data)
        self.key_mm.flush()

    def update_ursina_texture(self, *args):
        if self.use_pygame_render:
            # If using Pygame render, keep updating the texture with the Pygame image
            self.run_pygame_render()
        else:
            # Regular Ursina texture update
            self.frame_mm.seek(0)
            data = self.frame_mm.read(self.width * self.height * 4)
            self.shared_array = np.flipud(np.frombuffer(data, dtype=np.uint8).reshape((self.height, self.width, 4)))
            self.kivy_texture.blit_buffer(self.shared_array.flatten(), colorfmt='rgba', bufferfmt='ubyte')
            self.image.texture = self.kivy_texture
            self.image.canvas.ask_update()

    def run_pygame_render(self):
        """Run Pygame rendering and display output in Kivy."""
        # Create an off-screen surface for rendering
        screen = pygame.Surface((self.width, self.height))

        # Fill the screen and draw shapes
        screen.fill((255, 255, 255))
        pygame.draw.rect(screen, (0, 128, 255), pygame.Rect(30, 30, 60, 60))
        pygame.draw.circle(screen, (255, 0, 0), (400, 300), 50)

        # Convert Pygame surface to a NumPy array and to Kivy texture
        pygame_image = pygame.surfarray.array3d(screen).transpose([1, 0, 2])
        pil_image = Image.fromarray(pygame_image, 'RGB').convert('RGBA')
        self.update_kivy_texture(pil_image)

    def update_kivy_texture(self, pil_image):
        """Update the Kivy texture with the image from Pygame."""
        pil_image = pil_image.resize((self.width, self.height))
        texture_data = pil_image.tobytes()
        self.kivy_texture.blit_buffer(texture_data, colorfmt='rgba', bufferfmt='ubyte')
        self.image.texture = self.kivy_texture
        self.image.canvas.ask_update()

    def on_button_click(self, instance):
        self.update_ursina_texture()
        img = Image.fromarray(self.shared_array, 'RGBA')
        img.save('saved_image.png')
        print("Screenshot made")

    def show_popup(self, message): 
        # Create a new AnchorLayout for the popup content 
        popup_layout = AnchorLayout(anchor_x='center', anchor_y='top') 
        
        # Create a label with the message 
        message_label = Label(text=message) 
        
        # Add the message label to the popup layout 
        popup_layout.add_widget(message_label) 
        
        # Create a close button 
        close_button = Button(text='Close', size_hint=(1, 0.2)) 
        close_button.bind(on_press=self.dismiss_popup) 
        
        # Add the close button to the popup layout 
        popup_layout.add_widget(close_button) 
        
        # Create and open the popup 
        self.popup = Popup(title='Toggle Game Engine', content=popup_layout, size_hint=(0.6, 0.4)) 
        self.popup.open()

    def dismiss_popup(self, instance): 
        self.popup.dismiss()
        # Remove the widgets from the popup layout after closing the popup 
        self.popup.content.clear_widgets()

    def on_stop(self):
        self.ursina_process.terminate()
        self.frame_mm.close()
        self.key_mm.close()
        self.mouse_pos_mm.close()
        self.mouse_clicked_mm.close()
        self.mouse_listener.stop()

if __name__ == '__main__':
    KivyUrsinaApp().run()
