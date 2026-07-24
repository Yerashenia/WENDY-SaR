from openai import OpenAI


class OpenAIProvider:

    def __init__(self, model):
        self.client = OpenAI()
        self.model = model


    def generate(self, prompt):

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )

        return response.choices[0].message.content