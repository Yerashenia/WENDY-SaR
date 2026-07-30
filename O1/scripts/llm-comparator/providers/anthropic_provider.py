import anthropic


class AnthropicProvider:

    def __init__(self, model):

        self.client = anthropic.Anthropic()
        self.model = model
        self.cost_enabled = True


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


        return {
            "text": response.content[0].text,
            "input_tokens": response.usage.input_tokens,
            "output_tokens": response.usage.output_tokens,
            "total_tokens":
                response.usage.input_tokens
                +
                response.usage.output_tokens
        }