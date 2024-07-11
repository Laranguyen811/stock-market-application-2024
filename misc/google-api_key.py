import export as export
import os
import google.generativeai as genai
genai.configure(api_key=os.environ['GOOGLE_API_KEY'])
gemini_model = genai.GenerativeModel(model_name="gemini-1.5-flash")
