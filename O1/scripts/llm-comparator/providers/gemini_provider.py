import os
from google import genai


class GeminiProvider:

    def __init__(self, model):

        self.model = model
        self.cost_enabled = True

        self.client = genai.Client(
            api_key=os.getenv("GEMINI_API_KEY")
        )


    def generate(self, prompt):

        response = self.client.models.generate_content(
            model=self.model,
            contents=prompt
        )


        usage = response.usage_metadata


        return {
            "text": response.text,
            "input_tokens": usage.prompt_token_count,
            "output_tokens": usage.candidates_token_count,
            "total_tokens": usage.total_token_count
        }