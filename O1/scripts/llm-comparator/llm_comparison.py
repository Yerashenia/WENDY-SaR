import json
import csv
import os
import time

from dotenv import load_dotenv

from providers.openai_provider import OpenAIProvider
from providers.anthropic_provider import AnthropicProvider
from providers.gemini_provider import GeminiProvider
from providers.llama_provider import LlamaProvider


load_dotenv()


PROVIDERS = {
    "chatgpt": OpenAIProvider,
    "claude": AnthropicProvider,
    "gemini": GeminiProvider,
    "llama": LlamaProvider
}


# -------------------------
# File loading
# -------------------------

def load_json(path):
    with open(
        path,
        "r",
        encoding="utf-8"
    ) as file:
        return json.load(file)


def load_queries(path):
    with open(
        path,
        "r",
        encoding="utf-8"
    ) as file:
        return [
            line.strip()
            for line in file
            if line.strip()
        ]


# -------------------------
# Prompt construction
# -------------------------

def build_prompt(
    ontology,
    evidence,
    question
):
    return f"""
You are a conversational assistant called WENDY designed for helping University students in relating and learning module concepts.

You are an expert in helping students learn.

The user is a student.

You must answer using ONLY the provided ontology and evidence.

Do not use external knowledge.
Do not perform guess-work.
Do not mention the ontology or evidence.
Do not lie.

The user has access to the lecture slides, so signposting is helpful.

Be honest about your capabilities.


KNOWLEDGE GRAPH:

{json.dumps(
    ontology,
    indent=2
)}

EVIDENCE:

{json.dumps(
    evidence,
    indent=2
)}


QUESTION:

{question}


ANSWER:
"""


# -------------------------
# CSV output
# -------------------------

def save_csv(
    filename,
    rows
):
    directory = os.path.dirname(filename)

    if directory:
        os.makedirs(
            directory,
            exist_ok=True
        )

    file_exists = os.path.exists(filename)

    with open(
        filename,
        "a",
        newline="",
        encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "query_id",
                "llm",
                "query",
                "time_seconds",
                "response"
            ]
        )

        if not file_exists:
            writer.writeheader()

        writer.writerows(rows)


# -------------------------
# Create enabled providers
# -------------------------

def initialise_providers(config):
    providers = {}

    for name, settings in config["models"].items():
        if not settings.get(
            "enabled",
            False
        ):
            continue

        provider_class = PROVIDERS.get(name)

        if provider_class is None:
            print(
                f"Unknown provider: {name}"
            )
            continue

        providers[name] = provider_class(
            settings["model"]
        )

    return providers


# -------------------------
# Main
# -------------------------

def main():
    config = load_json(
        "config.json"
    )

    # Lecture notes knowledge graph
    ontology = load_json(
        "knowledge/ontology.json"
    )

    # Lecture evidence notes
    evidence = load_json(
        "knowledge/evidence.json"
    )

    # Questions (one per line)
    queries = load_queries(
        "queries/queries.txt"
    )

    # Initialise enabled LLMs once
    providers = initialise_providers(
        config
    )

    results = []

    for query_id, question in enumerate(
        queries,
        start=1
    ):
        print("\n")
        print("=" * 80)
        print(
            f"QUERY {query_id}: {question}"
        )

        prompt = build_prompt(
            ontology,
            evidence,
            question
        )

        for name, provider in providers.items():
            print("\n" + "-" * 60)
            print(
                f"Running {name}"
            )

            try:
                start = time.perf_counter()

                response = provider.generate(
                    prompt
                )

                elapsed = (
                    time.perf_counter()
                    -
                    start
                )

                print(response)

                print(
                    f"Time: {elapsed:.2f}s"
                )

                results.append(
                    {
                        "query_id": query_id,
                        "llm": name,
                        "query": question,
                        "time_seconds": round(
                            elapsed,
                            3
                        ),
                        "response": response
                    }
                )

            except Exception as e:
                print(
                    f"{name} failed:"
                )
                print(e)

                results.append(
                    {
                        "query_id": query_id,
                        "llm": name,
                        "query": question,
                        "time_seconds": 0,
                        "response": f"ERROR: {e}"
                    }
                )

    save_csv(
        config["output"]["csv_file"],
        results
    )

    print(
        "\nBenchmark complete. Results saved."
    )


if __name__ == "__main__":
    main()