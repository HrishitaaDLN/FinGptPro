import google.generativeai as genai

API_KEY = "AIzaSyASmQ8ALQ95ps--RaZKJXab9yGh7oogb14"
genai.configure(api_key=API_KEY)

try:
    for m in genai.list_models():
        if "generateContent" in m.supported_generation_methods:
            print(m.name)
except Exception as e:
    print("❌ Error:", e)
