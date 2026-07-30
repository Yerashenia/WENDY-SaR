import ollama


class LlamaProvider:

    def __init__(self, model):

        self.model = model
        self.cost_enabled = False


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


        input_tokens = response.get(
            "prompt_eval_count",
            0
        )


        output_tokens = response.get(
            "eval_count",
            0
        )


        return {
            "text": response["message"]["content"],
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens":
                input_tokens
                +
                output_tokens
        }