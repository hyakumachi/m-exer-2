import cv2
import numpy as np
import requests
from tensorflow.keras.models import load_model
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.camera import Camera
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.clock import Clock
from kivy.uix.scrollview import ScrollView
from kivy.core.window import Window

# Nutrition API function
def get_nutrition(fruit_name):  # Returns dictionary
    api_key = "yKjMzroQauMJKq2XO60IcK9DbUbxpqvPUmGaGBNR"  # BERNARD USDA API key
    url = "https://api.nal.usda.gov/fdc/v1/foods/search"

    params = {
        "query": fruit_name,
        "api_key": api_key,
        "pageSize": 1
    }

    response = requests.get(url, params=params)
    
    if response.status_code == 200:
        data = response.json()
        if data.get("foods"):
            nutrients = data["foods"][0].get("foodNutrients", [])
            nutrition_info = {
                nutrient["nutrientName"]: nutrient["value"] for nutrient in nutrients
            }
            return nutrition_info
        else:
            return {"error": "No data found."}
    else:
        return {"error": "API request failed."}

# Recipes API function
def get_recipes(fruit_name):  # Returns list
    api_key = "5fa8872fa7734a2cb4607720dff15622"  # BERNARD Spoonacular API key
    uri = "https://api.spoonacular.com/recipes/complexSearch"
    params = {
        "query": fruit_name,
        "apiKey": api_key,
        "number": 8
    }
    response = requests.get(uri, params=params)
    
    if response.status_code == 200:
        recipes = response.json()["results"]
        return [recipe["title"] for recipe in recipes]
    else:
        return ["No recipes found."]

# Load ML model and class indices
model = load_model('fruit_recognition_model.h5')  # Ensure this path is correct
fruit_classes = {
    0: "Eggplant",
    1: "Ginger",
    2: "Lemon",
    3: "Okra",
    4: "Orange"
}

class FruitDetectionApp(App):
    def __init__(self, **kwargs):
        super(FruitDetectionApp, self).__init__(**kwargs)
        self.current_fruit = None
        self.nutrition_cache = {}  # Cache for nutrition data
        self.recipes_cache = {}    # Cache for recipes

    def build(self):
        # Set window size (simulates Android portrait mode)
        Window.size = (720, 1280)
        Window.clearcolor = (0.94, 0.94, 0.94, 1)  # Light gray background

        # Main layout
        layout = BoxLayout(orientation='vertical', padding=15, spacing=10)

        # Camera widget
        self.camera = Camera(resolution=(640, 480), play=False, size_hint=(1, 0.4))

        # Detection label
        self.label = Label(
            text="Press Start to Detect",
            size_hint=(1, 0.05),
            font_size='24sp',
            bold=True,
            color=(0.2, 0.2, 0.2, 1),
            halign='center',
            valign='middle'
        )

        # Info layout for nutrition and recipes
        info_layout = BoxLayout(orientation='vertical', size_hint=(1, 0.45))

        # Nutrition section
        nutrition_container = BoxLayout(orientation='vertical', size_hint=(1, 0.5))
        nutrition_header = Label(
            text="Nutrition Info",
            size_hint=(1, 0.1),
            font_size='20sp',
            bold=True,
            color=(0.172, 0.471, 0.451, 1)  # Teal
        )
        nutrition_scroll = ScrollView(do_scroll_x=False, size_hint=(1, 0.9))
        self.nutrition_label = Label(
            text="",
            size_hint_y=None,
            font_size='18sp',
            color=(0.2, 0.2, 0.2, 1),
            halign='left',
            valign='top',
            text_size=(Window.width - 60, None),  # Dynamic width with padding
            padding=(10, 10)  # Padding to avoid edge cutoff
        )
        # Dynamically set height based on content
        self.nutrition_label.bind(texture_size=lambda instance, value: setattr(instance, 'height', value[1] + 20))
        nutrition_scroll.add_widget(self.nutrition_label)
        nutrition_container.add_widget(nutrition_header)
        nutrition_container.add_widget(nutrition_scroll)

        # Recipes section
        recipes_container = BoxLayout(orientation='vertical', size_hint=(1, 0.5))
        recipes_header = Label(
            text="Recipes",
            size_hint=(1, 0.1),
            font_size='20sp',
            bold=True,
            color=(0.172, 0.471, 0.451, 1)  # Teal
        )
        recipes_scroll = ScrollView(do_scroll_x=False, size_hint=(1, 0.9))
        self.recipes_label = Label(
            text="",
            size_hint_y=None,
            font_size='18sp',
            color=(0.2, 0.2, 0.2, 1),
            halign='left',
            valign='top',
            text_size=(Window.width - 60, None),  # Dynamic width with padding
            padding=(10, 10)  # Padding to avoid edge cutoff
        )
        # Dynamically set height based on content
        self.recipes_label.bind(texture_size=lambda instance, value: setattr(instance, 'height', value[1] + 20))
        recipes_scroll.add_widget(self.recipes_label)
        recipes_container.add_widget(recipes_header)
        recipes_container.add_widget(recipes_scroll)

        # Add sections to info layout
        info_layout.add_widget(nutrition_container)
        info_layout.add_widget(recipes_container)

        # Start/Stop button
        self.btn = Button(
            text="Start",
            size_hint=(1, 0.1),
            background_color=(0.172, 0.471, 0.451, 1),  # Teal
            background_normal='',
            font_size='22sp',
            bold=True,
            color=(1, 1, 1, 1)
        )
        self.btn.bind(on_press=self.toggle_camera)

        # Add widgets to main layout
        layout.add_widget(self.camera)
        layout.add_widget(self.label)
        layout.add_widget(info_layout)
        layout.add_widget(self.btn)

        # Bind window resize event to update text size
        Window.bind(on_resize=self.update_text_size)

        return layout

    def update_text_size(self, instance, width, height):
        # Adjust text_size based on new window width
        self.nutrition_label.text_size = (width - 60, None)
        self.recipes_label.text_size = (width - 60, None)

    def toggle_camera(self, instance):
        if not self.camera.play:
            self.camera.play = True
            self.btn.text = "Stop"
            self.btn.background_color = (0.867, 0.271, 0.271, 1)  # Red
            Clock.schedule_interval(self.detect_fruit, 1)
        else:
            self.camera.play = False
            self.btn.text = "Start"
            self.btn.background_color = (0.172, 0.471, 0.451, 1)  # Teal
            Clock.unschedule(self.detect_fruit)

    def detect_fruit(self, dt):
        frame = self.camera.texture
        if frame:
            frame = np.frombuffer(frame.pixels, dtype=np.uint8).reshape((480, 640, 4))
            frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2RGB)

            resized = cv2.resize(frame, (100, 100))
            normalized = resized / 255.0
            input_tensor = np.expand_dims(normalized, axis=0)

            predictions = model.predict(input_tensor, verbose=0)
            class_idx = np.argmax(predictions)
            confidence = np.max(predictions)
            fruit_name = fruit_classes[class_idx]
            self.label.text = f"{fruit_name} ({confidence:.2f})"

            # Only fetch data if the fruit changes
            if fruit_name != self.current_fruit:
                self.current_fruit = fruit_name

                # Nutrition info (check cache first)
                if fruit_name in self.nutrition_cache:
                    nutrition_text = self.nutrition_cache[fruit_name]
                else:
                    nutrition = get_nutrition(fruit_name)
                    nutrition_text = "\n".join([f"{k}: {v}" for k, v in list(nutrition.items())[:5]]) if nutrition else "Not Found"
                    self.nutrition_cache[fruit_name] = nutrition_text
                self.nutrition_label.text = nutrition_text

                # Recipes (check cache first)
                if fruit_name in self.recipes_cache:
                    recipes_text = self.recipes_cache[fruit_name]
                else:
                    recipes = get_recipes(fruit_name)
                    recipes_text = "\n".join(recipes[:5]) if recipes else "Not Found"
                    self.recipes_cache[fruit_name] = recipes_text
                self.recipes_label.text = recipes_text

if __name__ == "__main__":
    FruitDetectionApp().run()