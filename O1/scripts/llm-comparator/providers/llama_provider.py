import ollama


class LlamaProvider:

    def __init__(self, model):
        self.model = model


    def generate(self, prompt):

        response = ollama.chat(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )

        return response["message"]["content"]