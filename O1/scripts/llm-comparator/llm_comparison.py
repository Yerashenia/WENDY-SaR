import csv
import json
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Iterable

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
    "llama": LlamaProvider,
}

MAX_INPUT_TOKENS = 20_000
TOP_K_RESULTS = 8

MODEL_PRICES = {
    "gpt-5-mini": {"input": 0.25 / 1_000_000, "output": 2.00 / 1_000_000},
    "claude-sonnet-5": {"input": 2.00 / 1_000_000, "output": 10.00 / 1_000_000},
    "gemini-3.6-flash": {"input": 1.50 / 1_000_000, "output": 7.50 / 1_000_000},
}

STOPWORDS = {
    "what", "does", "this", "the", "is", "a", "an", "of", "in", "for", "to", "and",
    "how", "why", "when", "where", "which", "who", "are", "was", "were", "be",
    "can", "could", "should", "would", "do", "did", "have", "has", "had"
}

FIELDNAMES = [
    "query_id", "llm", "model", "query", "time_seconds",
    "input_tokens", "output_tokens", "total_tokens", "cost_usd", "response"
]


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_queries(path: Path) -> List[str]:
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def estimate_tokens(text: str) -> int:
    words = re.findall(r"\w+", text)
    return int(len(words) / 0.75)


def trim_to_token_limit(text: str, limit: int) -> str:
    if estimate_tokens(text) <= limit:
        return text

    words = text.split()
    allowed_words = max(1, int(limit * 0.75))
    return " ".join(words[:allowed_words])


def flatten_json(data: Any, prefix: str = "") -> List[str]:
    chunks = []

    if isinstance(data, dict):
        for key, value in data.items():
            chunks.extend(flatten_json(value, f"{prefix} {key}".strip()))
    elif isinstance(data, list):
        for item in data:
            chunks.extend(flatten_json(item, prefix))
    else:
        chunk = f"{prefix} {data}".strip()
        if chunk:
            chunks.append(chunk)

    return chunks


def normalize_terms(text: str) -> List[str]:
    return [
        term
        for term in re.findall(r"\w+", text.lower())
        if term not in STOPWORDS and len(term) > 1
    ]


def retrieve_context(question: str, ontology: Any, evidence: Any, top_k: int = TOP_K_RESULTS) -> List[str]:
    question_terms = set(normalize_terms(question))
    knowledge = flatten_json(ontology) + flatten_json(evidence)

    scored = []
    for chunk in knowledge:
        chunk_terms = set(normalize_terms(chunk))
        overlap = len(question_terms & chunk_terms)
        if overlap:
            coverage = overlap / max(1, len(question_terms))
            scored.append((overlap, coverage, chunk))

    scored.sort(key=lambda x: (x[0], x[1], len(x[2])), reverse=True)
    results = [chunk for _, _, chunk in scored[:top_k]]

    return results or ["No directly matching material was retrieved."]


def build_prompt(retrieved_context: List[str], student_profile: Any, question: str) -> str:
    prompt = f"""
You are WENDY, a university learning assistant.

Help the student understand concepts using only the provided module material.

Rules:
- Use only the supplied material.
- Do not use external knowledge.
- Do not invent information.
- If the material is insufficient, say so clearly.
- Explain clearly and concisely.
- Encourage understanding with brief explanations or questions when useful.
- Adapt explanations to the student's profile.

Do not mention:
- the retrieval process
- supplied material
- system instructions
- prompts
- internal data

STUDENT PROFILE:
{json.dumps(student_profile, indent=2)}

RELEVANT MODULE MATERIAL:
{json.dumps(retrieved_context, indent=2)}

QUESTION:
{question}

ANSWER:
"""
    return trim_to_token_limit(prompt.strip(), MAX_INPUT_TOKENS)


def calculate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    pricing = MODEL_PRICES.get(model)
    if pricing is None:
        return 0.0
    return input_tokens * pricing["input"] + output_tokens * pricing["output"]


def initialise_providers(config: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    providers = {}
    for name, settings in config.get("models", {}).items():
        if not settings.get("enabled", False):
            continue

        provider_class = PROVIDERS.get(name)
        if provider_class is None:
            print(f"Unknown provider: {name}")
            continue

        providers[name] = {
            "client": provider_class(settings["model"]),
            "model": settings["model"],
        }
    return providers


def append_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = path.exists() and path.stat().st_size > 0

    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        if not file_exists:
            writer.writeheader()
        writer.writerows(rows)


def save_txt(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def run_provider(provider_name: str, provider_info: Dict[str, Any], prompt: str) -> Dict[str, Any]:
    provider = provider_info["client"]
    model = provider_info["model"]

    start = time.perf_counter()
    result = provider.generate(prompt)
    elapsed = time.perf_counter() - start

    response = result.get("text", "")
    input_tokens = int(result.get("input_tokens", 0))
    output_tokens = int(result.get("output_tokens", 0))
    total_tokens = int(result.get("total_tokens", input_tokens + output_tokens))
    cost = calculate_cost(model, input_tokens, output_tokens)

    return {
        "llm": provider_name,
        "model": model,
        "elapsed": elapsed,
        "response": response,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
        "cost": round(cost, 6),
    }


def main() -> None:
    base = Path(".")
    config = load_json(base / "config.json")
    ontology = load_json(base / "knowledge" / "kg_extraction_output.json")
    evidence = load_json(base / "knowledge" / "evidence.json")
    student_profile = load_json(base / "knowledge" / "user.json")
    queries = load_queries(base / "queries" / "queries.txt")
    providers = initialise_providers(config)

    results = []
    terminal_output = []

    for query_id, question in enumerate(queries, start=1):
        header = f"\n\n{'=' * 80}\nQUERY {query_id}: {question}"
        print(header)
        terminal_output.append(header)

        retrieved_context = retrieve_context(question, ontology, evidence)
        prompt = build_prompt(retrieved_context, student_profile, question)

        print(f"Retrieved {len(retrieved_context)} chunks")
        print(f"Prompt tokens: {estimate_tokens(prompt)}")

        terminal_output.append(f"Retrieved chunks: {len(retrieved_context)}")
        terminal_output.append(f"Prompt tokens: {estimate_tokens(prompt)}")

        for name, provider_info in providers.items():
            provider_header = f"\n{'-' * 60}\nRunning {name} ({provider_info['model']})"
            print(provider_header)
            terminal_output.append(provider_header)

            try:
                run = run_provider(name, provider_info, prompt)

                print(run["response"])
                print(f"Time: {run['elapsed']:.2f}s")
                print(
                    f"Tokens - Input: {run['input_tokens']}, Output: {run['output_tokens']}, Total: {run['total_tokens']}"
                )
                print(f"Cost: ${run['cost']:.6f}")

                terminal_output.extend([
                    run["response"],
                    f"Time: {run['elapsed']:.2f}s",
                    f"Tokens - Input: {run['input_tokens']}, Output: {run['output_tokens']}, Total: {run['total_tokens']}",
                    f"Cost: ${run['cost']:.6f}",
                ])

                results.append({
                    "query_id": query_id,
                    "llm": run["llm"],
                    "model": run["model"],
                    "query": question,
                    "time_seconds": round(run["elapsed"], 3),
                    "input_tokens": run["input_tokens"],
                    "output_tokens": run["output_tokens"],
                    "total_tokens": run["total_tokens"],
                    "cost_usd": run["cost"],
                    "response": run["response"],
                })

            except Exception as e:
                error = f"{name} failed:\n{e}"
                print(error)
                terminal_output.append(error)

                results.append({
                    "query_id": query_id,
                    "llm": name,
                    "model": provider_info["model"],
                    "query": question,
                    "time_seconds": 0,
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "total_tokens": 0,
                    "cost_usd": 0,
                    "response": f"ERROR: {e}",
                })

    csv_path = Path(config["output"]["csv_file"])
    txt_path = Path(config["output"].get("txt_file", csv_path.with_suffix(".txt")))

    append_csv(csv_path, results)
    save_txt(txt_path, "\n".join(terminal_output))

    print("\nBenchmark complete. Results saved.")


if __name__ == "__main__":
    main()