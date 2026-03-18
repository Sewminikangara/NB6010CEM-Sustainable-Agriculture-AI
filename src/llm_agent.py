import os
import json
import google.generativeai as genai
from dotenv import load_dotenv

import config

load_dotenv()


class PlantAdvisor:
    def __init__(self, ontology_path=None):
        if ontology_path is None:
            ontology_path = os.path.join(config.BASE_DIR, "ontology", "plant_disease_ontology.json")

        with open(ontology_path, 'r') as f:
            self.ontology = json.load(f)["diseases"]

        self.api_key = os.getenv("GEMINI_API_KEY")
        if self.api_key:
            genai.configure(api_key=self.api_key)
            self.model = genai.GenerativeModel('gemini-flash-latest')
        else:
            self.model = None

    def get_advice(self, disease_name):
        info = self.ontology.get(disease_name, None)

        if not info:
            return "No specific information found for this disease in our database."

        base_advice = (
            f"**Common Name:** {info['common_name']}\n"
            f"**Cause:** {info['cause']}\n"
            f"**Symptoms:** {info['symptoms']}\n"
            f"**Recommended Treatment:** {info['treatment']}"
        )

        if self.model:
            try:
                prompt = (
                    f"You are an agricultural expert. A plant has been diagnosed with {info['common_name']} "
                    f"({disease_name}). Based on the following facts:\n{base_advice}\n\n"
                    "Provide concise, practical advice for a farmer including biological and chemical control "
                    "methods and preventive measures. Keep it under 300 words."
                )
                response = self.model.generate_content(prompt)
                return response.text
            except Exception as e:
                print(f"LLM Error: {e}")
                return base_advice + "\n\n*(Advanced advisory unavailable. Displaying database reference.)*"

        return base_advice
