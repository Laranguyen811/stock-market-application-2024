import google.generativeai as genai
import os

GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
genai.configure(api_key=GOOGLE_API_KEY)
for gemini_model in genai.list_models():
    if 'createTunedModel' in gemini_model.supported_generation_methods:
        print(gemini_model.name)