import anthropic


class AnthropicProvider:

    def __init__(self, model):

        self.client = anthropic.Anthropic()
        self.model = model


    def generate(self, prompt):

        response = self.client.messages.create(
            model=self.model,
            max_tokens=2048,
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )

        return response.content[0].text