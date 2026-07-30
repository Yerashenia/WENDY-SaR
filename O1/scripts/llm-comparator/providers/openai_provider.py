from openai import OpenAI


class OpenAIProvider:

    def __init__(self, model):

        self.client = OpenAI()
        self.model = model
        self.cost_enabled = True


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

        usage = response.usage

        return {
            "text": response.choices[0].message.content,
            "input_tokens": usage.prompt_tokens,
            "output_tokens": usage.completion_tokens,
            "total_tokens": usage.total_tokens
        }