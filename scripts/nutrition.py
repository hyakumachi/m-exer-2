import requests

def get_nutrition(fruit_name):  # Returns array
    api_key = "DBW6ar1kolQBGc0NNeTAihBaFtAiWYmQiWHo8Jdl"  # Replace api key
    url = "https://api.nal.usda.gov/fdc/v1/foods/search"  # FIXED URL

    params = {
        "query": fruit_name,
        "api_key": api_key,  # USDA API uses "api_key" instead of "apiKey"
        "pageSize": 1  # USDA API uses "pageSize" instead of "number"
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

# Example usage
fruit_name = "beans"
nutrition = get_nutrition(fruit_name)
print(f"Nutritional information for {fruit_name}: {nutrition}")
