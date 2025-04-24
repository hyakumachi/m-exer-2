import requests

def get_recipes(fruit_name): # Returns array
	api_key = "c9becf792f5648968445e5f885b12401" # Replace api key from spoonacular
	uri = "https://api.spoonacular.com/recipes/complexSearch" # Rep
	params = {
		"query": fruit_name,
		"apiKey": api_key,
		"number": 8 # NUmber of recipies to retrieve
	}
	response = requests.get(uri, params=params)
	
	# Check if code has errors
	if response.status_code == 200:
		recipes = response.json()["results"]
		return [recipe["title"] for recipe in recipes] # return recipe titles
	else: 
		return ["No recipies found. "] # Handle errors

# Example usage
# fruit_name = "carrot"
# recipes = get_recipes(fruit_name)
# print(f"Recipes for {fruit_name}: {recipes}")