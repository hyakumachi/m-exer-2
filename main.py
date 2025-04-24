import cv2
import numpy as np
import json
import requests
import threading

from tensorflow.keras.models import load_model
from kivy.app import App
from kivy.uix.image import Image
from kivy.graphics.texture import Texture
from kivy.clock import Clock, mainthread
from kivy.uix.label import Label
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.scrollview import ScrollView
from kivy.core.window import Window

# Load Keras .h5 model
model = load_model('fruit_recognition_model.h5')

# Load class indices
with open('class_indices.json', 'r') as f:
    fruit_classes = json.load(f)

def get_recipes(fruit_name):
    api_key = "198ce7a9d14c498ca8f519efcb536cd1" #SPOONACULAR API
    #ADDU ACC API KEY: 5fa8872fa7734a2cb4607720dff15622
    url = "https://api.spoonacular.com/recipes/complexSearch" 
    params = {"query": fruit_name, "apiKey": api_key, "number": 5}

    response = requests.get(url, params=params)

    if response.status_code == 402:
        return ["API limit reached. Try again later."]
    elif response.status_code == 200:
        recipes = response.json().get("results", [])
        return [recipe["title"] for recipe in recipes] if recipes else ["No recipes found."]
    
    return ["Error fetching recipes."]

def get_nutrition(fruit_name):
    api_key = "yKjMzroQauMJKq2XO60IcK9DbUbxpqvPUmGaGBNR" #USDA API
    #ADDU ACC API KEY: yKjMzroQauMJKq2XO60IcK9DbUbxpqvPUmGaGBNR
    url = "https://api.nal.usda.gov/fdc/v1/foods/search"
    params = {"query": fruit_name, "api_key": api_key, "pageSize": 1}
    
    response = requests.get(url, params=params)
    
    if response.status_code == 200:
        data = response.json()
        foods = data.get("foods", [])
        if foods:
            nutrients = foods[0].get("foodNutrients", [])
            return {nutrient["nutrientName"]: nutrient["value"] for nutrient in nutrients[:5]}
    
    return {"error": "No data found."}

class FruitRecognitionApp(App):
    def build(self):
        Window.clearcolor = (0.0, 0.439, 0.235, 1)
        
        self.layout = BoxLayout(orientation='vertical')
        self.image = Image()
        self.label = Label(text='Detecting...', font_size='20sp', size_hint=(1, 0.1))
        
        # Horizontal layout for recipes and nutrition
        self.info_layout = BoxLayout(orientation='horizontal', size_hint=(1, 0.3), spacing=10)

        # ScrollView for recipes
        self.recipes_scroll = ScrollView(size_hint=(0.5, 1))
        self.recipes_label = Label(text='Recipes:\n-', font_size='15sp', size_hint_y=None, height=100)
        self.recipes_label.bind(texture_size=self.update_label_height)
        self.recipes_scroll.add_widget(self.recipes_label)

        # ScrollView for nutrition
        self.nutrition_scroll = ScrollView(size_hint=(0.5, 1))
        self.nutrition_label = Label(text='Nutrition:\n-', font_size='15sp', size_hint_y=None, height=100)
        self.nutrition_label.bind(texture_size=self.update_label_height)
        self.nutrition_scroll.add_widget(self.nutrition_label)

        # Add widgets to the horizontal layout
        self.info_layout.add_widget(self.recipes_scroll)
        self.info_layout.add_widget(self.nutrition_scroll)

        self.start_stop_button = Button(text="Start Camera", size_hint=(1, 0.1))
        self.start_stop_button.bind(on_press=self.toggle_camera)
        
        self.layout.add_widget(self.image)
        self.layout.add_widget(self.label)
        self.layout.add_widget(self.info_layout)  # Add updated layout
        self.layout.add_widget(self.start_stop_button)
        
        self.capture = None
        self.running = False
        return self.layout

    def toggle_camera(self, instance):
        if self.running:
            self.running = False
            self.start_stop_button.text = "Start Camera"
            self.start_stop_button.background_color = (0.117, 0.565, 1, 1)
            if self.capture:
                self.capture.release()
        else:
            self.running = True
            self.start_stop_button.text = "Stop Camera"
            self.start_stop_button.background_color = (1, 0.2, 0.2, 1)
            self.capture = cv2.VideoCapture(0)
            Clock.schedule_interval(self.update, 1.0 / 15.0)

    def preprocess_image(self, frame):
        resized = cv2.resize(frame, (100, 100))
        normalized = resized.astype(np.float32) / 255.0
        return np.expand_dims(normalized, axis=0)

    def predict(self, input_tensor):
        predictions = model.predict(input_tensor)
        return predictions

    def update(self, dt):
        if not self.running or self.capture is None:
            return
        
        ret, frame = self.capture.read()
        if not ret:
            return
        
        input_tensor = self.preprocess_image(frame)
        predictions = self.predict(input_tensor)
        class_idx = np.argmax(predictions)
        confidence = np.max(predictions)
        fruit_name = fruit_classes[class_idx]  

        label = f"{fruit_name} ({confidence:.2f})"
        
        self.label.text = label
        
        threading.Thread(target=self.fetch_info, args=(fruit_name,), daemon=True).start()
        
        frame = cv2.flip(frame, 0)
        buf = frame.tobytes()
        texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
        texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
        self.image.texture = texture

    def fetch_info(self, fruit_name):
        if not self.running:
            return

        recipes = get_recipes(fruit_name)
        nutrition = get_nutrition(fruit_name)

        if not self.running:
            return

        recipes_text = "\n".join(recipes)
        nutrition_text = "\n".join([f"{k}: {v}" for k, v in nutrition.items()])

        self.update_ui(recipes_text, nutrition_text)

    @mainthread
    def update_ui(self, recipes_text, nutrition_text):
        self.recipes_label.text = f"Recipes:\n{recipes_text}"
        self.nutrition_label.text = f"Nutrition:\n{nutrition_text}"

    def update_label_height(self, instance, value):
        instance.height = instance.texture_size[1]  # Adjust height dynamically

    def on_stop(self):
        if self.capture:
            self.capture.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    FruitRecognitionApp().run()


#CODE NI LOEJEE
# import cv2
# import numpy as np
# import requests
# from tensorflow.keras.models import load_model
# from kivy.app import App
# from kivy.uix.boxlayout import BoxLayout
# from kivy.uix.camera import Camera
# from kivy.uix.button import Button
# from kivy.uix.label import Label
# from kivy.clock import Clock
# from kivy.uix.scrollview import ScrollView
# from kivy.core.window import Window

# # Nutrition API function
# def get_nutrition(fruit_name):  # Returns dictionary
#     api_key = "yKjMzroQauMJKq2XO60IcK9DbUbxpqvPUmGaGBNR"  # BERNARD USDA API key
#     url = "https://api.nal.usda.gov/fdc/v1/foods/search"

#     params = {
#         "query": fruit_name,
#         "api_key": api_key,
#         "pageSize": 1
#     }

#     response = requests.get(url, params=params)
    
#     if response.status_code == 200:
#         data = response.json()
#         if data.get("foods"):
#             nutrients = data["foods"][0].get("foodNutrients", [])
#             nutrition_info = {
#                 nutrient["nutrientName"]: nutrient["value"] for nutrient in nutrients
#             }
#             return nutrition_info
#         else:
#             return {"error": "No data found."}
#     else:
#         return {"error": "API request failed."}

# # Recipes API function
# def get_recipes(fruit_name):  # Returns list
#     api_key = "5fa8872fa7734a2cb4607720dff15622"  # BERNARD Spoonacular API key
#     uri = "https://api.spoonacular.com/recipes/complexSearch"
#     params = {
#         "query": fruit_name,
#         "apiKey": api_key,
#         "number": 8
#     }
#     response = requests.get(uri, params=params)
    
#     if response.status_code == 200:
#         recipes = response.json()["results"]
#         return [recipe["title"] for recipe in recipes]
#     else:
#         return ["No recipes found."]

# # Load ML model and class indices
# model = load_model('fruit_recognition_model.h5')  # Ensure this path is correct
# fruit_classes = {
#     0: "Eggplant",
#     1: "Ginger",
#     2: "Lemon",
#     3: "Okra",
#     4: "Orange"
# }

# class FruitDetectionApp(App):
#     def __init__(self, **kwargs):
#         super(FruitDetectionApp, self).__init__(**kwargs)
#         self.current_fruit = None
#         self.nutrition_cache = {}  # Cache for nutrition data
#         self.recipes_cache = {}    # Cache for recipes

#     def build(self):
#         # Set window size (simulates Android portrait mode)
#         Window.size = (720, 1280)
#         Window.clearcolor = (0.94, 0.94, 0.94, 1)  # Light gray background

#         # Main layout
#         layout = BoxLayout(orientation='vertical', padding=15, spacing=10)

#         # Camera widget
#         self.camera = Camera(resolution=(640, 480), play=False, size_hint=(1, 0.4))

#         # Detection label
#         self.label = Label(
#             text="Press Start to Detect",
#             size_hint=(1, 0.05),
#             font_size='24sp',
#             bold=True,
#             color=(0.2, 0.2, 0.2, 1),
#             halign='center',
#             valign='middle'
#         )

#         # Info layout for nutrition and recipes
#         info_layout = BoxLayout(orientation='vertical', size_hint=(1, 0.45))

#         # Nutrition section
#         nutrition_container = BoxLayout(orientation='vertical', size_hint=(1, 0.5))
#         nutrition_header = Label(
#             text="Nutrition Info",
#             size_hint=(1, 0.1),
#             font_size='20sp',
#             bold=True,
#             color=(0.172, 0.471, 0.451, 1)  # Teal
#         )
#         nutrition_scroll = ScrollView(do_scroll_x=False, size_hint=(1, 0.9))
#         self.nutrition_label = Label(
#             text="",
#             size_hint_y=None,
#             font_size='18sp',
#             color=(0.2, 0.2, 0.2, 1),
#             halign='left',
#             valign='top',
#             text_size=(Window.width - 60, None),  # Dynamic width with padding
#             padding=(10, 10)  # Padding to avoid edge cutoff
#         )
#         # Dynamically set height based on content
#         self.nutrition_label.bind(texture_size=lambda instance, value: setattr(instance, 'height', value[1] + 20))
#         nutrition_scroll.add_widget(self.nutrition_label)
#         nutrition_container.add_widget(nutrition_header)
#         nutrition_container.add_widget(nutrition_scroll)

#         # Recipes section
#         recipes_container = BoxLayout(orientation='vertical', size_hint=(1, 0.5))
#         recipes_header = Label(
#             text="Recipes",
#             size_hint=(1, 0.1),
#             font_size='20sp',
#             bold=True,
#             color=(0.172, 0.471, 0.451, 1)  # Teal
#         )
#         recipes_scroll = ScrollView(do_scroll_x=False, size_hint=(1, 0.9))
#         self.recipes_label = Label(
#             text="",
#             size_hint_y=None,
#             font_size='18sp',
#             color=(0.2, 0.2, 0.2, 1),
#             halign='left',
#             valign='top',
#             text_size=(Window.width - 60, None),  # Dynamic width with padding
#             padding=(10, 10)  # Padding to avoid edge cutoff
#         )
#         # Dynamically set height based on content
#         self.recipes_label.bind(texture_size=lambda instance, value: setattr(instance, 'height', value[1] + 20))
#         recipes_scroll.add_widget(self.recipes_label)
#         recipes_container.add_widget(recipes_header)
#         recipes_container.add_widget(recipes_scroll)

#         # Add sections to info layout
#         info_layout.add_widget(nutrition_container)
#         info_layout.add_widget(recipes_container)

#         # Start/Stop button
#         self.btn = Button(
#             text="Start",
#             size_hint=(1, 0.1),
#             background_color=(0.172, 0.471, 0.451, 1),  # Teal
#             background_normal='',
#             font_size='22sp',
#             bold=True,
#             color=(1, 1, 1, 1)
#         )
#         self.btn.bind(on_press=self.toggle_camera)

#         # Add widgets to main layout
#         layout.add_widget(self.camera)
#         layout.add_widget(self.label)
#         layout.add_widget(info_layout)
#         layout.add_widget(self.btn)

#         # Bind window resize event to update text size
#         Window.bind(on_resize=self.update_text_size)

#         return layout

#     def update_text_size(self, instance, width, height):
#         # Adjust text_size based on new window width
#         self.nutrition_label.text_size = (width - 60, None)
#         self.recipes_label.text_size = (width - 60, None)

#     def toggle_camera(self, instance):
#         if not self.camera.play:
#             self.camera.play = True
#             self.btn.text = "Stop"
#             self.btn.background_color = (0.867, 0.271, 0.271, 1)  # Red
#             Clock.schedule_interval(self.detect_fruit, 1)
#         else:
#             self.camera.play = False
#             self.btn.text = "Start"
#             self.btn.background_color = (0.172, 0.471, 0.451, 1)  # Teal
#             Clock.unschedule(self.detect_fruit)

#     def detect_fruit(self, dt):
#         frame = self.camera.texture
#         if frame:
#             frame = np.frombuffer(frame.pixels, dtype=np.uint8).reshape((480, 640, 4))
#             frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2RGB)

#             resized = cv2.resize(frame, (100, 100))
#             normalized = resized / 255.0
#             input_tensor = np.expand_dims(normalized, axis=0)

#             predictions = model.predict(input_tensor, verbose=0)
#             class_idx = np.argmax(predictions)
#             confidence = np.max(predictions)
#             fruit_name = fruit_classes[class_idx]
#             self.label.text = f"{fruit_name} ({confidence:.2f})"

#             # Only fetch data if the fruit changes
#             if fruit_name != self.current_fruit:
#                 self.current_fruit = fruit_name

#                 # Nutrition info (check cache first)
#                 if fruit_name in self.nutrition_cache:
#                     nutrition_text = self.nutrition_cache[fruit_name]
#                 else:
#                     nutrition = get_nutrition(fruit_name)
#                     nutrition_text = "\n".join([f"{k}: {v}" for k, v in list(nutrition.items())[:5]]) if nutrition else "Not Found"
#                     self.nutrition_cache[fruit_name] = nutrition_text
#                 self.nutrition_label.text = nutrition_text

#                 # Recipes (check cache first)
#                 if fruit_name in self.recipes_cache:
#                     recipes_text = self.recipes_cache[fruit_name]
#                 else:
#                     recipes = get_recipes(fruit_name)
#                     recipes_text = "\n".join(recipes[:5]) if recipes else "Not Found"
#                     self.recipes_cache[fruit_name] = recipes_text
#                 self.recipes_label.text = recipes_text

# if __name__ == "__main__":
#     FruitDetectionApp().run()