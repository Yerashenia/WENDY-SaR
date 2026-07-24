from google import genai


class GeminiProvider:

    def __init__(self, model):

        self.model = model

        genai.configure()

        self.client = genai.GenerativeModel(
            model
        )


    def generate(self, prompt):

        response = self.client.generate_content(
            prompt
        )

        return response.text