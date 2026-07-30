import os
import re
import json
import time
import math
import logging
import urllib.request
from pathlib import Path
from collections import Counter
from difflib import SequenceMatcher

from dotenv import load_dotenv

# Neo4j is no longer required for the JSON-only extraction path.
try:
    from neo4j import GraphDatabase
    from neo4j.exceptions import ServiceUnavailable, SessionExpired, TransientError
except ImportError:
    GraphDatabase = None
    ServiceUnavailable = SessionExpired = TransientError = Exception

from llama_index.core import Document
from llama_index.core.node_parser import SentenceSplitter
from llama_index.llms.ollama import Ollama
from llama_index.readers.file import PDFReader

from rdflib import Graph, Literal, Namespace, RDF, URIRef
from pyshacl import validate


# -------------------------------------------------------------------
# Logging configuration
# -------------------------------------------------------------------

logging.basicConfig(level=logging.ERROR)
logging.getLogger("pypdf").setLevel(logging.ERROR)
logging.getLogger("pypdf._reader").setLevel(logging.ERROR)
logging.getLogger("httpx").setLevel(logging.ERROR)
logging.getLogger("httpcore").setLevel(logging.ERROR)


# -------------------------------------------------------------------
# Environment configuration
# -------------------------------------------------------------------

load_dotenv()

DATA_SOURCE = os.getenv("pdf_folder")
NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
NEO4J_DATABASE = os.getenv("NEO4J_DATABASE")

ONTOLOGY_FILE = os.getenv("ONTOLOGY_FILE", "../validation_schema/kg_ontology.owl")
SHACL_FILE = os.getenv("SHACL_FILE", "../validation_schema/kg_validation_shapes.ttl")

USE_EMBEDDINGS = os.getenv("USE_EMBEDDINGS", "true").lower() == "true"
OLLAMA_EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")
OLLAMA_EMBED_URL = os.getenv("OLLAMA_EMBED_URL", "http://localhost:11434/api/embeddings")
EMBEDDING_SIMILARITY_THRESHOLD = float(os.getenv("EMBEDDING_SIMILARITY_THRESHOLD", "0.88"))

OUTPUT_JSON_FILE = os.getenv("OUTPUT_JSON_FILE", "llm-comparator/knowledge/kg_extraction_output.json")
EVIDENCE_JSON_FILE = os.getenv("EVIDENCE_JSON_FILE", "llm-comparator/knowledge/evidence.json")
RUN_SHACL_VALIDATION = os.getenv("RUN_SHACL_VALIDATION", "false").lower() == "true"


# -------------------------------------------------------------------
# PDF loading and preprocessing
# -------------------------------------------------------------------

def load_pdf_documents(pdf_folder: str):
    """
    Load all PDF files from the configured folder.
    Each PDF page is loaded as a separate document and receives the source PDF path.
    """
    reader = PDFReader()
    documents = []

    for file in Path(pdf_folder).glob("*.pdf"):
        pages = reader.load_data(file)
        for page in pages:
            page.metadata["pdf_path"] = str(file)
        documents.extend(pages)

    return documents


def create_llm():
    """
    Create the local Ollama LLM used for metadata and KG extraction.
    """
    return Ollama(
        model="llama3",
        temperature=0.0,
        request_timeout=300.0,
    )


def extract_pdf_metadata(pdf_path: str, llm):
    """
    Extract trusted module name and week number from the first PDF page.
    """
    import pypdf

    reader = pypdf.PdfReader(pdf_path)
    first_page_text = reader.pages[0].extract_text() or ""

    prompt = f"""
Extract ONLY the Module name and Week number from the text below.

RULES:
- Module name is usually a course title, for example "Mathematics for Computing".
- Week number comes from "Lecture X" or "Week X".
- Return nothing except the required two lines.
- Format exactly like this:
Module:<module name>
Week:<week number>

Text:
{first_page_text}
"""

    response = llm.complete(prompt)
    output = response.text.strip()

    module_name = None
    week_number = None

    for line in output.splitlines():
        line = line.strip()
        if line.startswith("Module:"):
            module_name = line.replace("Module:", "", 1).strip()
        elif line.startswith("Week:"):
            week_number = line.replace("Week:", "", 1).strip()

    return module_name, week_number


def clean_text(text: str):
    """
    Apply light cleaning to extracted PDF text.
    """
    if not text:
        return ""

    text = text.replace("\n", " ").replace("\t", " ")
    text = " ".join(text.split())

    text = re.sub(r"([a-z])([A-Z])", r"\1 \2", text)
    text = re.sub(r"([A-Za-z])-\s+([A-Za-z])", r"\1\2", text)

    text = text.replace("º", "=")
    text = text.replace("∣", "|")

    text = re.sub(r"([!\"#$%&'*+,\-./:;<=>?@\\^_`|~]){5,}", " ", text)

    return text.strip()


def is_academic_content(text: str) -> bool:
    """
    Exclude pages that are mainly administrative rather than academic.
    """
    lower_text = text.lower()
    unwanted_keywords = [
        "module leader",
        "office hours",
        "book an office hours appointment",
        "module team",
        "discussion board",
        "blackboard",
        "attendance is important",
        "word cloud",
        "pollev",
        "independent study",
        "welcome to the maths for computing module",
        "seminars",
        "lecture guidelines",
        "use your university email",
        "module announcements",
    ]
    return not any(keyword in lower_text for keyword in unwanted_keywords)


def is_text_too_noisy(text: str) -> bool:
    """
    Reject chunks that contain mostly symbols or unreadable PDF noise.
    """
    if not text or len(text.strip()) < 40:
        return True

    non_alphanumeric = len(re.findall(r"[^A-Za-z0-9\s]", text))
    total = max(len(text), 1)
    symbol_ratio = non_alphanumeric / total

    if symbol_ratio > 0.48:
        return True

    words = re.findall(r"[A-Za-z]{3,}", text)
    if len(words) < 8:
        return True

    return False


def is_overview_or_agenda_chunk(text: str) -> bool:
    """
    Detect title/agenda chunks that only list lecture sections.
    """
    lower_text = text.lower()
    short_text = len(text.split()) <= 35

    has_agenda_markers = any(
        marker in lower_text
        for marker in [
            "module introduction",
            "lecture 1",
            "lecture 2",
            "key topics",
            "topics to review",
            "today we will cover",
        ]
    )

    has_explanation_markers = any(
        marker in lower_text
        for marker in [
            " is ",
            " are ",
            " means ",
            " refers to ",
            " defined as ",
            " consists of ",
            " contains ",
            " used for ",
            " used in ",
            " where ",
        ]
    )

    if short_text and has_agenda_markers and not has_explanation_markers:
        return True

    return False


def is_revision_list_chunk(text: str) -> bool:
    """
    Detect revision/task list chunks.
    """
    lower_text = text.lower()
    revision_markers = [
        "home revision task",
        "key topics to review",
        "activity: home revision",
    ]
    return any(marker in lower_text for marker in revision_markers)


def is_activity_or_example_chunk(text: str) -> bool:
    """
    Detect chunks that are mainly activities, answers, tasks, or examples.
    """
    lower_text = text.lower().strip()
    markers = [
        "activity",
        "task",
        "homework",
        "answer",
        "answers",
        "solve the following",
        "determine if",
        "example:",
        "examples",
        "detailed answer",
    ]
    return any(marker in lower_text[:220] for marker in markers)


def prepare_documents(documents, llm):
    """
    Clean loaded PDF pages and attach trusted module/week metadata.
    """
    cleaned_documents = []
    seen_pdfs = {}

    for doc in documents:
        pdf_path = doc.metadata.get("pdf_path")

        if pdf_path and pdf_path not in seen_pdfs:
            module_name, week_number = extract_pdf_metadata(pdf_path, llm)
            seen_pdfs[pdf_path] = {
                "module": module_name,
                "week": week_number,
            }

        if pdf_path in seen_pdfs:
            doc.metadata["module"] = seen_pdfs[pdf_path]["module"]
            doc.metadata["week"] = seen_pdfs[pdf_path]["week"]

        cleaned = clean_text(doc.text)

        if not is_academic_content(cleaned):
            continue
        if is_text_too_noisy(cleaned):
            continue
        if is_overview_or_agenda_chunk(cleaned):
            continue
        if is_revision_list_chunk(cleaned):
            continue

        cleaned_documents.append(
            Document(
                text=cleaned,
                metadata=doc.metadata,
            )
        )

    return cleaned_documents


def chunk_documents(documents, chunk_size=1000, chunk_overlap=150):
    """
    Split cleaned documents into semantic chunks.
    """
    splitter = SentenceSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    return splitter.get_nodes_from_documents(documents)


# -------------------------------------------------------------------
# LLM extraction
# -------------------------------------------------------------------

def extract_kg_json(text: str, llm):
    """
    Extract Topic and Concept information as JSON.
    """
    prompt = f"""
You are extracting a clean academic knowledge graph from university lecture text.

Return ONLY valid JSON.
Do not use markdown.
Do not explain.
Do not include comments.
Do not output placeholder text such as "short exact quote from the text".

TARGET GRAPH STRUCTURE:
Module -> HAS_WEEK -> Week
Week -> COVERS -> Topic
Topic -> HAS_CONCEPT -> Concept
Concept -> RELATED_TO -> Concept
Concept -> BUILDS_ON -> Concept
Concept -> PREREQUISITE_OF -> Concept

IMPORTANT:
- Do NOT create Definition nodes.
- Do NOT create Example nodes.
- Do NOT create Formula nodes.
- Do NOT create Application nodes.
- Store definition/example/formula/application only as text properties inside a Concept object.
- Do NOT create Module or Week objects. They are handled by metadata.

NODE QUALITY RULES:
- Extract only meaningful academic topics and concepts.
- A Topic should be a broader lecture section.
- A Concept should be a specific taught idea under a Topic.
- If the slide heading and concept are the same, choose a broader topic from the surrounding text.
- Do not extract table column labels as concepts.
- Do not extract activity names, question names, answers, or task labels as concepts.
- Do not extract full sentences as concept names.
- Do not extract symbols or example sets as concept names.
- Do not create a concept named "Formula", "Equation", "Notation", "Expression", "Attribute", "Property", or "Task" unless it is clearly a real taught mathematical concept.
- If a formula belongs to a concept, put it in that concept's "formula" field.
- If the text is mainly examples or activities, extract the underlying existing concept, not a new task concept.
- If the text is about applications of a concept, create the main concept and put the applications in its "application" field.
- If the text is about reasons/benefits of a concept, create the main concept and put the reasons in its "description" or "application" field.
- If the chunk only lists lecture headings or revision topics, return empty topics.
- If there is no valid academic content, return:
{{"topics": [], "relationships": []}}

GOOD EXAMPLES:
- "Sequences" can be Topic and "Sequence" can be Concept if the text defines what a sequence is.
- "Set Theory" can be Topic and "Set" can be Concept if the text defines what a set is.
- "Set Theory" can be Topic and "Proper Subset" can be Concept.
- "Intervals" should stay under Topic "Intervals", not "Number Theory".
- "Sets vs Intervals" should not create a concept called "Attribute Set".

JSON FORMAT:
{{
  "topics": [
    {{
      "name": "Set Theory",
      "confidence": 0.92,
      "source_quote": "Sets are fundamental objects in mathematics",
      "concepts": [
        {{
          "name": "Subset",
          "description": "A set A is a subset of set B if every element of A is also an element of B.",
          "formula": "A ⊆ B",
          "example": "A={{1,2}}, B={{1,2,3}}, so A⊆B",
          "application": "",
          "confidence": 0.95,
          "source_quote": "A set A is considered a subset of set B"
        }}
      ]
    }}
  ],
  "relationships": [
    {{
      "source": "Proper Subset",
      "type": "BUILDS_ON",
      "target": "Subset",
      "confidence": 0.85,
      "source_quote": "If A is a subset of B, and A is not equal to B"
    }}
  ]
}}

VALID RELATIONSHIP TYPES:
- RELATED_TO
- BUILDS_ON
- PREREQUISITE_OF

TEXT:
{text}
"""

    response = llm.complete(prompt)
    return response.text.strip()


def extract_json_object(raw_output: str):
    """
    Safely extract a JSON object from the LLM response.
    """
    if not raw_output:
        return {"topics": [], "relationships": []}

    raw_output = raw_output.strip()
    raw_output = raw_output.replace("```json", "").replace("```", "").strip()

    try:
        data = json.loads(raw_output)
        return normalise_extraction_json(data)
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", raw_output, flags=re.DOTALL)
    if not match:
        return {"topics": [], "relationships": []}

    try:
        data = json.loads(match.group(0))
        return normalise_extraction_json(data)
    except json.JSONDecodeError:
        return {"topics": [], "relationships": []}


def normalise_extraction_json(data):
    """
    Ensure the extracted JSON always has the expected top-level shape.
    """
    if not isinstance(data, dict):
        return {"topics": [], "relationships": []}

    topics = data.get("topics", [])
    relationships = data.get("relationships", [])

    if not isinstance(topics, list):
        topics = []
    if not isinstance(relationships, list):
        relationships = []

    return {
        "topics": topics,
        "relationships": relationships,
    }


# -------------------------------------------------------------------
# Automatic semantic filtering
# -------------------------------------------------------------------

_embedding_cache = {}


def safe_float(value, default=0.5):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def normalize_week_name(week_number):
    week_text = str(week_number).strip()
    if not week_text.lower().startswith("week"):
        week_text = f"Week {week_text}"
    return week_text


def normalize_name(value: str):
    if not value:
        return ""

    value = str(value)
    value = re.sub(r"\s+", " ", value).strip()
    value = re.sub(r"([A-Za-z])-\s+([A-Za-z])", r"\1\2", value)
    value = value.strip(" .,:;")
    value = value.strip("\"'“”‘’")

    if not value:
        return ""

    lower_value = value.lower()

    small_synonym_map = {
        "sets": "Set",
        "set": "Set",
        "attribute set": "Set",
        "number set": "Number Sets",
        "number sets": "Number Sets",
        "natural numbers": "Natural Numbers",
        "natural number": "Natural Numbers",
        "integer": "Integers",
        "integers": "Integers",
        "rational number": "Rational Numbers",
        "rational numbers": "Rational Numbers",
        "irrational number": "Irrational Numbers",
        "irrational numbers": "Irrational Numbers",
        "real number": "Real Numbers",
        "real numbers": "Real Numbers",
        "whole number": "Whole Numbers",
        "whole numbers": "Whole Numbers",
        "complex number": "Complex Numbers",
        "complex numbers": "Complex Numbers",
        "prime number": "Prime Numbers",
        "prime numbers": "Prime Numbers",
        "composite number": "Composite Numbers",
        "composite numbers": "Composite Numbers",
        "divine ratio": "Golden Ratio",
        "cardinality of a set": "Cardinality",
        "cardinality/length": "Cardinality",
        "set builder notation examples": "Set Builder Notation",
        "modular mathematics": "Modular Arithmetic",
        "modulo operation": "Modulo Operation",
        "modular arithmetic notation": "Modulo Operation",
        "binary number system": "Binary Number System",
        "decimal number system": "Decimal Number System",
        "geometric sequence formula": "Geometric Sequence",
        "arithmetic sequence formula": "Arithmetic Sequence",
        "interval task": "Interval Membership",
        "interval tasks": "Interval Membership",
        "clock time": "Clock Arithmetic",
        "complementation": "Complement",
        "set of ordered pairs": "Ordered Pair",
    }

    if lower_value in small_synonym_map:
        return small_synonym_map[lower_value]

    words = value.split()
    fixed_words = []

    for word in words:
        if word.isupper() and len(word) <= 5:
            fixed_words.append(word)
        elif any(char.isdigit() for char in word):
            fixed_words.append(word)
        elif "/" in word:
            parts = [part[:1].upper() + part[1:].lower() for part in word.split("/")]
            fixed_words.append("/".join(parts))
        else:
            fixed_words.append(word[:1].upper() + word[1:].lower())

    return " ".join(fixed_words)


def clean_property_text(value: str, max_length: int):
    if value is None:
        return ""

    value = str(value)
    value = re.sub(r"\s+", " ", value).strip()
    value = value.replace("º", "=")

    bad_values = {
        "none",
        "n/a",
        "null",
        "not provided",
        "(not provided)",
        "unknown",
        "short exact quote from the text",
        "short exact quote from the text that supports this topic",
        "short exact quote from the text that supports this concept",
        "short exact quote from the text supporting this relationship",
    }

    if value.lower() in bad_values:
        return ""

    if "short exact quote from the text" in value.lower():
        return ""

    if value in {"# = Q rem R", "!"}:
        return ""

    value = re.sub(r"\s+", " ", value).strip()

    if len(value) > max_length:
        value = value[:max_length].rsplit(" ", 1)[0].strip()

    return value


def meaningful_tokens(text: str):
    stopwords = {
        "a", "an", "the", "and", "or", "of", "to", "in", "on", "for",
        "with", "by", "from", "as", "is", "are", "be", "this", "that",
        "these", "those", "where", "which", "when", "then", "than",
        "into", "using", "used", "use", "all", "each", "both",
        "main", "basic", "basics", "introduction", "example", "examples",
        "activity", "task", "answer", "answers",
    }

    tokens = re.findall(r"[A-Za-z0-9]+", str(text).lower())

    return [
        token
        for token in tokens
        if len(token) >= 3 and token not in stopwords
    ]


def get_embedding(text: str):
    if not USE_EMBEDDINGS:
        return None

    text = normalize_name(text)
    if not text:
        return None

    if text in _embedding_cache:
        return _embedding_cache[text]

    payload = json.dumps(
        {
            "model": OLLAMA_EMBED_MODEL,
            "prompt": text,
        }
    ).encode("utf-8")

    request = urllib.request.Request(
        OLLAMA_EMBED_URL,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            result = json.loads(response.read().decode("utf-8"))
            embedding = result.get("embedding")

            if isinstance(embedding, list):
                _embedding_cache[text] = embedding
                return embedding
    except Exception:
        return None

    return None


def cosine_similarity(first_vector, second_vector):
    if not first_vector or not second_vector:
        return 0.0

    dot_product = sum(a * b for a, b in zip(first_vector, second_vector))
    first_norm = math.sqrt(sum(a * a for a in first_vector))
    second_norm = math.sqrt(sum(b * b for b in second_vector))

    if first_norm == 0 or second_norm == 0:
        return 0.0

    return dot_product / (first_norm * second_norm)


def embedding_similarity(first_name: str, second_name: str):
    first_embedding = get_embedding(first_name)
    second_embedding = get_embedding(second_name)

    if not first_embedding or not second_embedding:
        return 0.0

    return cosine_similarity(first_embedding, second_embedding)


def source_support_score(name: str, source_text: str, source_quote: str = "") -> float:
    if not name or not source_text:
        return 0.0

    name_lower = name.lower()
    text_lower = source_text.lower()
    quote_lower = (source_quote or "").lower()

    if name_lower in text_lower or (quote_lower and name_lower in quote_lower):
        return 1.0

    tokens = meaningful_tokens(name)
    if not tokens:
        return 0.0

    token_hits = sum(1 for token in tokens if token in text_lower)
    token_score = token_hits / max(len(tokens), 1)

    fuzzy_score = 0.0
    source_words = text_lower.split()
    window_size = max(len(name.split()) + 2, 3)

    for i in range(0, max(len(source_words) - window_size + 1, 1)):
        window = " ".join(source_words[i:i + window_size])
        score = SequenceMatcher(None, name_lower, window).ratio()
        fuzzy_score = max(fuzzy_score, score)

        if fuzzy_score >= 0.92:
            break

    return max(token_score, fuzzy_score)


def contains_math_expression(text: str) -> bool:
    if not text:
        return False

    math_symbols = [
        "=", "≡", "∪", "∩", "⊂", "⊆", "∈", "∉", "∅",
        "+", "-", "×", "÷", "/", "^", "√", "mod", "|",
        "≤", "≥", "<", ">", "∞",
    ]
    lower_text = str(text).lower()
    return any(symbol in lower_text for symbol in math_symbols)


def is_probably_table_attribute(name: str, source_text: str) -> bool:
    name = normalize_name(name)
    lower_name = name.lower()
    lower_text = source_text.lower()

    generic_table_attributes = {
        "collection", "ordering", "repetition", "notation", "indexing",
        "attribute", "attributes", "property", "properties", "definition",
        "description", "examples", "example", "length", "cardinality/length",
        "value", "values", "object", "objects", "method", "result", "results",
    }

    if lower_name not in generic_table_attributes:
        return False

    comparison_markers = [
        "property sets sequences",
        "attribute sets intervals",
        "sets sequences",
        "sets intervals",
        "definition",
        "notation",
        "ordering",
        "repetition",
        "indexing",
    ]

    return any(marker in lower_text for marker in comparison_markers)


def name_quality_score(name: str, source_text: str = "") -> float:
    if not name:
        return 0.0

    raw = str(name).strip()
    lower = raw.lower()

    if len(raw) < 3 or len(raw) > 80:
        return 0.0

    words = raw.split()
    if len(words) > 6:
        return 0.15

    sentence_markers = [
        " is ", " are ", " was ", " were ", " has ", " have ",
        " contains ", " consists ", " refers ", " represents ", " used to ",
    ]

    if any(marker in f" {lower} " for marker in sentence_markers):
        return 0.1

    instruction_starts = [
        "activity", "task", "question", "answer", "example",
        "exercise", "homework", "solve", "determine", "calculate",
    ]

    if any(lower.startswith(item) for item in instruction_starts):
        return 0.05

    symbol_count = len(re.findall(r"[^A-Za-z0-9\s\-/()]", raw))
    symbol_ratio = symbol_count / max(len(raw), 1)

    if symbol_ratio > 0.25:
        return 0.1

    alpha_count = len(re.findall(r"[A-Za-z]", raw))
    alpha_ratio = alpha_count / max(len(raw), 1)

    if alpha_ratio < 0.45:
        return 0.1

    if len(words) == 1 and len(raw) <= 2:
        return 0.0

    generic_noise = {
        "definition", "description", "property", "properties", "attribute",
        "attributes", "example", "examples", "formula", "formulas",
        "application", "applications", "notes", "note", "result", "results",
        "method", "methods", "step", "steps", "object", "objects", "value",
        "values", "usage", "reason", "reasons", "benefit", "benefits",
        "compatibility", "efficiency", "interval task", "attribute set",
    }

    if lower in generic_noise or is_probably_table_attribute(raw, source_text):
        return 0.05

    meaningful_count = len(meaningful_tokens(raw))

    if meaningful_count == 0:
        return 0.2
    if 1 <= meaningful_count <= 4:
        return 1.0

    return 0.7


def combined_confidence_score(
    name: str,
    source_text: str,
    llm_confidence: float,
    source_quote: str = "",
):
    quality = name_quality_score(name, source_text)
    support = source_support_score(name, source_text, source_quote)

    llm_confidence = safe_float(llm_confidence, 0.5)
    llm_confidence = max(0.0, min(1.0, llm_confidence))

    return (0.45 * quality) + (0.40 * support) + (0.15 * llm_confidence)


def similarity_score(first_name: str, second_name: str) -> float:
    first_name = normalize_name(first_name)
    second_name = normalize_name(second_name)

    if not first_name or not second_name:
        return 0.0

    if first_name.lower() == second_name.lower():
        return 1.0

    first_tokens = set(meaningful_tokens(first_name))
    second_tokens = set(meaningful_tokens(second_name))

    if first_tokens and second_tokens:
        token_overlap = len(first_tokens.intersection(second_tokens)) / len(first_tokens.union(second_tokens))
    else:
        token_overlap = 0.0

    sequence_score = SequenceMatcher(None, first_name.lower(), second_name.lower()).ratio()
    semantic_score = embedding_similarity(first_name, second_name)

    return max(token_overlap, sequence_score, semantic_score)


def is_topic_concept_duplicate(topic_name: str, concept_name: str) -> bool:
    topic_name = normalize_name(topic_name)
    concept_name = normalize_name(concept_name)

    if topic_name.lower() == concept_name.lower():
        return True

    valid_topic_concept_pairs = {
        ("sequences", "sequence"),
        ("set theory", "set"),
        ("set theory", "subset"),
        ("set theory", "proper subset"),
        ("set theory", "power set"),
        ("number theory", "integers"),
        ("number systems", "binary number system"),
        ("number systems", "decimal number system"),
        ("intervals", "interval"),
    }

    if (topic_name.lower(), concept_name.lower()) in valid_topic_concept_pairs:
        return False

    return similarity_score(topic_name, concept_name) >= EMBEDDING_SIMILARITY_THRESHOLD


def is_semantically_duplicate(first_name: str, second_name: str) -> bool:
    return similarity_score(first_name, second_name) >= EMBEDDING_SIMILARITY_THRESHOLD


def infer_broader_topic_name(topic_name: str, concept_names: list, source_text: str):
    topic_name = normalize_name(topic_name)
    lower_topic = topic_name.lower()
    lower_text = source_text.lower()

    if "interval" in lower_topic or "interval" in lower_text[:250]:
        return "Intervals"
    if "set operations" in lower_text[:250]:
        return "Set Operations"
    if "set algebra" in lower_topic or "fundamental laws of set algebra" in lower_text[:250]:
        return "Set Algebra"
    if "cartesian product" in lower_topic or "cartesian product" in lower_text[:250]:
        return "Set Theory"
    if "power set" in lower_topic or "power set" in lower_text[:250]:
        return "Set Theory"
    if "set builder" in lower_topic or "set builder notation" in lower_text[:250]:
        return "Set Theory"
    if "subset" in lower_topic or "subset" in lower_text[:250]:
        return "Set Theory"
    if "union" in lower_topic or "intersection" in lower_topic or "difference" in lower_topic:
        return "Set Operations"
    if "number format" in lower_topic or "number system" in lower_topic or "number systems" in lower_text[:250]:
        return "Number Systems"
    if "number type" in lower_topic or "number sets" in lower_text[:250]:
        return "Number Types"
    if "sequence" in lower_topic or "sequences" in lower_text[:250]:
        return "Sequences"
    if "modular" in lower_topic:
        return "Modular Arithmetic"
    if "cipher" in lower_topic or "encryption" in lower_text[:250]:
        return "Cryptography"
    if "integer" in lower_topic or "integers" in lower_text[:250]:
        return "Number Theory"

    if concept_names:
        shared_tokens = Counter()
        for concept_name in concept_names:
            for token in meaningful_tokens(concept_name):
                shared_tokens[token] += 1

        common_tokens = [token for token, count in shared_tokens.items() if count >= 2]
        if common_tokens:
            return normalize_name(" ".join(common_tokens[:3]))

    return topic_name


def is_application_chunk(source_text: str) -> bool:
    lower_text = source_text.lower()
    return "applications of" in lower_text or "used for" in lower_text or "used in" in lower_text


def is_reason_or_benefit_chunk(source_text: str) -> bool:
    lower_text = source_text.lower()
    markers = [
        "why are ", "why is ", "reasons for ", "benefits of ",
        "advantages of ", "why use ", "why do we ",
    ]
    return any(marker in lower_text for marker in markers)


def is_formula_label_concept(concept_name: str, concept: dict) -> bool:
    name = normalize_name(concept_name)
    lower_name = name.lower()

    has_math = contains_math_expression(concept.get("formula", "")) or contains_math_expression(concept.get("description", ""))

    formula_suffixes = [" formula", " notation", " expression", " equation"]
    if any(lower_name.endswith(suffix) for suffix in formula_suffixes) and has_math:
        if lower_name in {"linear equations", "quadratic equations", "set builder notation"}:
            return False
        return True

    return False


def is_weak_reason_label(concept_name: str, concept: dict, source_text: str) -> bool:
    if not is_reason_or_benefit_chunk(source_text):
        return False

    name = normalize_name(concept_name)
    lower_name = name.lower()

    description = clean_property_text(concept.get("description", ""), 500)
    formula = clean_property_text(concept.get("formula", ""), 250)
    example = clean_property_text(concept.get("example", ""), 300)

    if description or formula or example:
        return False

    technical_terms = [
        "operation", "operations", "indexing", "algorithm", "algorithms",
        "cryptography", "statistics", "probability", "geometry", "trigonometry", "bitwise",
    ]

    if any(term in lower_name for term in technical_terms):
        return False

    if len(name.split()) <= 3:
        return True

    return False


def is_bad_activity_concept(concept_name: str, source_text: str) -> bool:
    name = normalize_name(concept_name).lower()
    bad_names = {
        "activity", "task", "answer", "answers", "exercise",
        "question", "interval task", "interval membership task",
    }

    if name in bad_names:
        return True

    if name.endswith(" task") or name.endswith(" activity") or name.endswith(" answer"):
        return True

    if is_activity_or_example_chunk(source_text) and name in {"attribute set", "collection", "ordering", "repetition"}:
        return True

    return False


def should_keep_topic(topic: dict, source_text: str) -> bool:
    name = normalize_name(topic.get("name", ""))
    quote = clean_property_text(topic.get("source_quote", ""), 300)
    confidence = topic.get("confidence", 0.5)

    score = combined_confidence_score(
        name=name,
        source_text=source_text,
        llm_confidence=confidence,
        source_quote=quote,
    )
    return score >= 0.64


def should_keep_concept(concept: dict, source_text: str) -> bool:
    name = normalize_name(concept.get("name", ""))
    quote = clean_property_text(concept.get("source_quote", ""), 300)
    confidence = concept.get("confidence", 0.5)

    score = combined_confidence_score(
        name=name,
        source_text=source_text,
        llm_confidence=confidence,
        source_quote=quote,
    )

    if is_bad_activity_concept(name, source_text):
        return False

    if is_weak_reason_label(name, concept, source_text):
        return False

    return score >= 0.60


def normalize_relationship_type(rel_type: str):
    if not rel_type:
        return None

    rel_type = str(rel_type).strip().upper()
    valid_types = {"RELATED_TO", "BUILDS_ON", "PREREQUISITE_OF"}

    if rel_type not in valid_types:
        return None

    return rel_type


def derive_main_concept_from_text(source_text: str, fallback_name: str):
    text = source_text.strip()
    match = re.search(r"applications of ([A-Za-z0-9\s\-]+?)(?:•|:|\.|$)", text, flags=re.IGNORECASE)
    if match:
        return normalize_name(match.group(1))

    match = re.search(r"why (?:are|is|do|does) ([A-Za-z0-9\s\-]+?) (?:popular|essential|important|useful|used)", text, flags=re.IGNORECASE)
    if match:
        return normalize_name(match.group(1))

    return normalize_name(fallback_name)


def merge_text_items(items: list, max_length: int):
    clean_items = []
    for item in items:
        item = clean_property_text(item, 160)
        if item and item not in clean_items:
            clean_items.append(item)

    merged = "; ".join(clean_items)
    if len(merged) > max_length:
        merged = merged[:max_length].rsplit(";", 1)[0].strip()

    return merged


def extract_formula_from_text(text: str):
    if not text:
        return ""

    candidates = re.findall(r"([A-Za-z0-9_()]+(?:\s*[=≡]\s*[^,.;]+))", text)
    if candidates:
        return clean_property_text(candidates[0], 250)

    if contains_math_expression(text):
        return clean_property_text(text, 250)

    return ""


def infer_semantic_relationships(concept_names: list):
    relationships = []
    names = {name.lower(): name for name in concept_names}

    def add_if_present(source, rel_type, target):
        source_key = source.lower()
        target_key = target.lower()
        if source_key in names and target_key in names:
            relationships.append((names[source_key], rel_type, names[target_key]))

    add_if_present("Arithmetic Sequence", "BUILDS_ON", "Sequence")
    add_if_present("Geometric Sequence", "BUILDS_ON", "Sequence")
    add_if_present("Proper Subset", "BUILDS_ON", "Subset")
    add_if_present("Power Set", "BUILDS_ON", "Subset")
    add_if_present("Complement", "RELATED_TO", "Set Operations")
    add_if_present("Union", "RELATED_TO", "Set Operations")
    add_if_present("Intersection", "RELATED_TO", "Set Operations")

    return relationships


def add_node(nodes: dict, label: str, name: str, properties: dict = None):
    """
    Add a node to the in-memory graph payload.
    """
    name = normalize_name(name)
    if not name:
        return None

    key = (label, name)
    if key not in nodes:
        nodes[key] = {
            "label": label,
            "name": name,
            "properties": {},
        }

    if properties:
        for prop_key, prop_value in properties.items():
            if prop_value is None or prop_value == "":
                continue

            if prop_key == "confidence":
                existing = nodes[key]["properties"].get("confidence", 0.0)
                nodes[key]["properties"]["confidence"] = max(existing, prop_value)
                continue

            existing_value = nodes[key]["properties"].get(prop_key)
            if not existing_value:
                nodes[key]["properties"][prop_key] = prop_value

    return key


def add_relationship(relationships: list, sub_key, rel_type: str, obj_key):
    """
    Add a relationship to the in-memory graph payload.
    """
    if not sub_key or not obj_key or not rel_type:
        return

    relationship = (sub_key, rel_type, obj_key)
    if relationship not in relationships:
        relationships.append(relationship)


def extract_valid_concepts_for_topic(topic: dict, source_text: str):
    """
    Extract, filter, and transform concepts for a topic.
    """
    topic_name = normalize_name(topic.get("name", ""))
    concepts = topic.get("concepts", [])

    if not isinstance(concepts, list):
        return []

    valid_concepts = []
    promoted_properties = {}
    application_items = []
    reason_items = []

    for concept in concepts:
        if not isinstance(concept, dict):
            continue

        raw_concept_name = concept.get("name", "")
        concept_name = normalize_name(raw_concept_name)

        if not concept_name:
            continue

        description = clean_property_text(concept.get("description", ""), 500)
        formula = clean_property_text(concept.get("formula", ""), 250)
        example = clean_property_text(concept.get("example", ""), 300)
        application = clean_property_text(concept.get("application", ""), 300)
        source_quote = clean_property_text(concept.get("source_quote", ""), 300)

        if is_bad_activity_concept(concept_name, source_text):
            continue

        if is_probably_table_attribute(concept_name, source_text):
            continue

        if is_formula_label_concept(concept_name, concept):
            target_name = concept_name
            for suffix in [" Formula", " Notation", " Expression", " Equation"]:
                if target_name.endswith(suffix):
                    target_name = target_name.removesuffix(suffix)

            if not target_name or len(target_name) < 3:
                target_name = topic_name

            extracted_formula = formula or extract_formula_from_text(description)

            if target_name not in promoted_properties:
                promoted_properties[target_name] = {
                    "description": "",
                    "formula": "",
                    "example": "",
                    "application": "",
                    "confidence": 0.0,
                    "source_quote": source_quote,
                }

            if extracted_formula:
                promoted_properties[target_name]["formula"] = extracted_formula

            if description and not contains_math_expression(description):
                promoted_properties[target_name]["description"] = description

            promoted_properties[target_name]["confidence"] = max(
                promoted_properties[target_name]["confidence"],
                safe_float(concept.get("confidence", 0.7)),
            )

            continue

        if is_application_chunk(source_text) and not (description or formula or example):
            if application:
                application_items.append(application)
            elif concept_name:
                application_items.append(concept_name)
            continue

        if is_weak_reason_label(concept_name, concept, source_text):
            if description:
                reason_items.append(f"{concept_name}: {description}")
            else:
                reason_items.append(concept_name)
            continue

        if not should_keep_concept(concept, source_text):
            continue

        valid_concepts.append({
            "name": concept_name,
            "description": description,
            "formula": formula,
            "example": example,
            "application": application,
            "confidence": safe_float(concept.get("confidence", 0.7)),
            "source_quote": source_quote,
        })

    if application_items:
        main_concept = derive_main_concept_from_text(source_text, topic_name)
        app_text = merge_text_items(application_items, 300)

        valid_concepts.append({
            "name": main_concept,
            "description": app_text,
            "formula": "",
            "example": "",
            "application": app_text,
            "confidence": 0.85,
            "source_quote": app_text,
        })

    if reason_items:
        main_concept = derive_main_concept_from_text(source_text, topic_name)
        reason_text = merge_text_items(reason_items, 300)

        valid_concepts.append({
            "name": main_concept,
            "description": reason_text,
            "formula": "",
            "example": "",
            "application": reason_text,
            "confidence": 0.85,
            "source_quote": reason_text,
        })

    for target_name, properties in promoted_properties.items():
        target_name = normalize_name(target_name)
        if not target_name:
            continue

        existing = next(
            (item for item in valid_concepts if item["name"].lower() == target_name.lower()),
            None,
        )

        if existing:
            for prop in ["description", "formula", "example", "application"]:
                if properties.get(prop) and not existing.get(prop):
                    existing[prop] = properties[prop]
            existing["confidence"] = max(
                existing.get("confidence", 0.0),
                round(properties.get("confidence", 0.5), 3),
            )
        else:
            valid_concepts.append({
                "name": target_name,
                "description": properties.get("description", ""),
                "formula": properties.get("formula", ""),
                "example": properties.get("example", ""),
                "application": properties.get("application", ""),
                "confidence": round(properties.get("confidence", 0.7), 3),
                "source_quote": properties.get("source_quote", ""),
            })

    deduped = {}
    for concept in valid_concepts:
        canonical_name = normalize_name(concept["name"])
        key = canonical_name.lower()
        concept["name"] = canonical_name

        if key not in deduped:
            deduped[key] = concept
            continue

        existing = deduped[key]
        for prop in ["description", "formula", "example", "application"]:
            if concept.get(prop) and not existing.get(prop):
                existing[prop] = concept[prop]
        existing["confidence"] = max(
            existing.get("confidence", 0.0),
            concept.get("confidence", 0.0),
        )

    return list(deduped.values())


def build_graph_payload(extraction: dict, source_text: str, module_name: str, week_number: str, metadata: dict):
    """
    Convert filtered JSON extraction into nodes and relationships.
    """
    nodes = {}
    relationships = []

    module_name = normalize_name(module_name or "Unknown Module")
    week_name = normalize_week_name(week_number or "0")

    module_key = add_node(
        nodes, "Module", module_name,
        {"sourceFile": metadata.get("pdf_path", "")},
    )

    week_number_int = None
    week_number_match = re.search(r"\d+", str(week_name))
    if week_number_match:
        week_number_int = int(week_number_match.group(0))

    week_key = add_node(
        nodes, "Week", week_name,
        {
            "weekNumber": week_number_int,
            "sourceFile": metadata.get("pdf_path", ""),
        },
    )

    add_relationship(relationships, module_key, "HAS_WEEK", week_key)

    topics = extraction.get("topics", [])
    if not isinstance(topics, list):
        topics = []

    all_concept_names_in_chunk = []
    concept_name_to_key = {}

    for topic in topics:
        if not isinstance(topic, dict):
            continue

        raw_topic_name = topic.get("name", "")
        topic_name = normalize_name(raw_topic_name)

        if not topic_name:
            continue

        valid_concepts = extract_valid_concepts_for_topic(topic, source_text)

        if is_topic_concept_duplicate(topic_name, topic_name):
            concept_names = [item["name"] for item in valid_concepts]
            topic_name = infer_broader_topic_name(topic_name, concept_names, source_text)

        if not should_keep_topic(topic, source_text):
            if valid_concepts:
                concept_names = [item["name"] for item in valid_concepts]
                topic_name = infer_broader_topic_name(topic_name, concept_names, source_text)
            else:
                continue

        filtered_concepts = []
        for concept in valid_concepts:
            concept_name = concept["name"]
            has_properties = any([
                concept.get("description"),
                concept.get("formula"),
                concept.get("example"),
                concept.get("application"),
            ])

            if is_topic_concept_duplicate(topic_name, concept_name) and not has_properties:
                continue

            filtered_concepts.append(concept)

        if not filtered_concepts:
            continue

        topic_score = combined_confidence_score(
            name=topic_name,
            source_text=source_text,
            llm_confidence=topic.get("confidence", 0.5),
            source_quote=clean_property_text(topic.get("source_quote", ""), 300),
        )

        topic_key = add_node(
            nodes, "Topic", topic_name,
            {
                "confidence": round(topic_score, 3),
                "sourceFile": metadata.get("pdf_path", ""),
            },
        )

        add_relationship(relationships, week_key, "COVERS", topic_key)

        for concept in filtered_concepts:
            concept_name = normalize_name(concept.get("name", ""))
            if not concept_name:
                continue

            concept_key = add_node(
                nodes, "Concept", concept_name,
                {
                    "description": clean_property_text(concept.get("description", ""), 500),
                    "formula": clean_property_text(concept.get("formula", ""), 250),
                    "example": clean_property_text(concept.get("example", ""), 300),
                    "application": clean_property_text(concept.get("application", ""), 300),
                    "confidence": concept.get("confidence", 0.7),
                    "sourceFile": metadata.get("pdf_path", ""),
                },
            )

            concept_name_to_key[concept_name.lower()] = concept_key
            all_concept_names_in_chunk.append(concept_name)

            add_relationship(relationships, topic_key, "HAS_CONCEPT", concept_key)

    for source_name, rel_type, target_name in infer_semantic_relationships(all_concept_names_in_chunk):
        source_key = concept_name_to_key.get(source_name.lower())
        target_key = concept_name_to_key.get(target_name.lower())
        if source_key and target_key:
            add_relationship(relationships, source_key, rel_type, target_key)

    for rel in extraction.get("relationships", []):
        if not isinstance(rel, dict):
            continue

        rel_type = normalize_relationship_type(rel.get("type", ""))
        if not rel_type:
            continue

        source_name = normalize_name(rel.get("source", ""))
        target_name = normalize_name(rel.get("target", ""))

        if not source_name or not target_name:
            continue

        if is_semantically_duplicate(source_name, target_name):
            continue

        source_key = concept_name_to_key.get(source_name.lower())
        target_key = concept_name_to_key.get(target_name.lower())

        if source_key and target_key:
            add_relationship(relationships, source_key, rel_type, target_key)

    return {
        "nodes": nodes,
        "relationships": relationships,
    }


# -------------------------------------------------------------------
# Evidence Extraction Helpers
# -------------------------------------------------------------------

def extract_evidence_from_chunk(node, extraction, graph_payload):
    """
    Extract detailed text contexts, original quotes, and map extracted
    concepts back to source slide metadata.
    """
    metadata = node.metadata or {}
    pdf_path = metadata.get("pdf_path", "Unknown")
    module_name = metadata.get("module", "Unknown Module")
    week_number = metadata.get("week", "0")
    raw_text = node.text

    chunk_concepts = []
    for (label, name), node_data in graph_payload["nodes"].items():
        if label == "Concept":
            chunk_concepts.append({
                "name": name,
                "description": node_data.get("properties", {}).get("description", ""),
                "formula": node_data.get("properties", {}).get("formula", ""),
                "example": node_data.get("properties", {}).get("example", ""),
                "application": node_data.get("properties", {}).get("application", "")
            })

    quotes = []
    for topic in extraction.get("topics", []):
        for concept in topic.get("concepts", []):
            quote = concept.get("source_quote")
            if quote and quote not in quotes:
                quotes.append(quote)

    evidence_entry = {
        "chunk_id": getattr(node, "node_id", f"chunk_{hash(raw_text)}"),
        "source_file": pdf_path,
        "module": module_name,
        "week": week_number,
        "associated_concepts": [c["name"] for c in chunk_concepts],
        "extracted_quotes": quotes,
        "full_text": raw_text,
        "concept_details": chunk_concepts
    }

    return evidence_entry


def update_evidence_store(evidence_store, new_evidence):
    """
    Update running evidence array and append indexed concept entries.
    """
    evidence_store["chunks"].append(new_evidence)

    for concept in new_evidence["concept_details"]:
        c_name = concept["name"]
        if c_name not in evidence_store["concept_index"]:
            evidence_store["concept_index"][c_name] = []

        evidence_store["concept_index"][c_name].append({
            "source_file": new_evidence["source_file"],
            "week": new_evidence["week"],
            "quote": new_evidence["extracted_quotes"],
            "text": new_evidence["full_text"],
            "formula": concept["formula"],
            "example": concept["example"]
        })


# -------------------------------------------------------------------
# RDF / SHACL validation helpers
# -------------------------------------------------------------------

def graph_payload_to_rdf_graph(graph_payload):
    kg = Namespace("http://example.org/kg#")
    graph = Graph()
    graph.bind("kg", kg)

    def get_node_uri(label, name):
        safe_name = re.sub(r"[^A-Za-z0-9_]", "_", name)
        return URIRef(f"http://example.org/kg/{label}/{safe_name}")

    for (label, name), node_data in graph_payload["nodes"].items():
        node_uri = get_node_uri(label, name)
        graph.add((node_uri, RDF.type, kg[label]))
        graph.add((node_uri, kg.name, Literal(name)))

        for prop_key, prop_value in node_data.get("properties", {}).items():
            if prop_value is None or prop_value == "":
                continue
            graph.add((node_uri, kg[prop_key], Literal(prop_value)))

    for (sub_label, sub_name), rel_type, (obj_label, obj_name) in graph_payload["relationships"]:
        sub_uri = get_node_uri(sub_label, sub_name)
        obj_uri = get_node_uri(obj_label, obj_name)
        graph.add((sub_uri, kg[rel_type], obj_uri))

    return graph


def validate_graph_payload_with_shacl(
    graph_payload,
    ontology_file=ONTOLOGY_FILE,
    shacl_file=SHACL_FILE,
):
    if not graph_payload["nodes"] or not graph_payload["relationships"]:
        return False, "No nodes or relationships found."

    if not Path(ontology_file).exists():
        return False, f"Ontology file not found: {ontology_file}"

    if not Path(shacl_file).exists():
        return False, f"SHACL validation file not found: {shacl_file}"

    data_graph = graph_payload_to_rdf_graph(graph_payload)
    ontology_graph = Graph()
    ontology_graph.parse(ontology_file)

    shacl_graph = Graph()
    shacl_graph.parse(shacl_file)

    conforms, results_graph, results_text = validate(
        data_graph=data_graph,
        shacl_graph=shacl_graph,
        ont_graph=ontology_graph,
        inference="rdfs",
        abort_on_first=False,
        allow_infos=True,
        allow_warnings=True,
    )

    return conforms, results_text


# -------------------------------------------------------------------
# JSON-only export helpers
# -------------------------------------------------------------------

def merge_graph_payload(accumulated: dict, new_payload: dict):
    for (label, name), node_data in new_payload["nodes"].items():
        add_node(accumulated["nodes"], label, name, node_data.get("properties", {}))

    for sub_key, rel_type, obj_key in new_payload["relationships"]:
        add_relationship(accumulated["relationships"], sub_key, rel_type, obj_key)


def graph_payload_to_json_friendly(graph_payload: dict) -> dict:
    nodes_list = []
    for (label, name), node_data in graph_payload["nodes"].items():
        nodes_list.append({
            "label": label,
            "name": name,
            "properties": node_data.get("properties", {}),
        })

    rel_list = []
    for (sub_label, sub_name), rel_type, (obj_label, obj_name) in graph_payload["relationships"]:
        rel_list.append({
            "source": {"label": sub_label, "name": sub_name},
            "type": rel_type,
            "target": {"label": obj_label, "name": obj_name},
        })

    return {
        "nodes": nodes_list,
        "relationships": rel_list,
    }


def print_chunk_summary(nodes):
    week_counts = Counter(node.metadata.get("week") for node in nodes)
    print("\n--- WEEK DISTRIBUTION ---")
    for week, count in week_counts.items():
        print(f"Week {week}: {count} chunks")


def print_graph_payload(graph_payload: dict):
    print("\n--- FILTERED NODES ---")
    for (label, name), node_data in graph_payload["nodes"].items():
        print((label, name, node_data.get("properties", {})))

    print("\n--- FILTERED RELATIONSHIPS ---")
    for relationship in graph_payload["relationships"]:
        print(relationship)


def format_duration(seconds: float) -> str:
    """
    Format a duration in seconds as a human-readable string.
    """
    minutes, secs = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)

    if hours >= 1:
        return f"{int(hours)}h {int(minutes)}m {secs:.1f}s"
    if minutes >= 1:
        return f"{int(minutes)}m {secs:.1f}s"
    return f"{secs:.2f}s"


# -------------------------------------------------------------------
# Main pipeline
# -------------------------------------------------------------------

if __name__ == "__main__":
    run_start = time.perf_counter()

    llm = create_llm()

    print("Loading documents...")
    load_start = time.perf_counter()
    raw_documents = load_pdf_documents(DATA_SOURCE)
    load_elapsed = time.perf_counter() - load_start
    print(f"Loaded {len(raw_documents)} raw pages in {format_duration(load_elapsed)}")

    print("Preparing documents...")
    prepare_start = time.perf_counter()
    documents = prepare_documents(raw_documents, llm)
    prepare_elapsed = time.perf_counter() - prepare_start
    print(f"Prepared {len(documents)} documents in {format_duration(prepare_elapsed)}")

    print("Chunking documents...")
    chunk_start_time = time.perf_counter()
    nodes_to_process = chunk_documents(documents)
    chunking_elapsed = time.perf_counter() - chunk_start_time
    print(f"Created {len(nodes_to_process)} chunks in {format_duration(chunking_elapsed)}")

    print(f"\nTotal processable chunks: {len(nodes_to_process)}")
    print_chunk_summary(nodes_to_process)

    accumulated_payload = {
        "nodes": {},
        "relationships": [],
    }

    accumulated_evidence = {
        "chunks": [],
        "concept_index": {}
    }

    total_validated_chunks = 0
    total_failed_chunks = 0
    total_empty_chunks = 0

    chunk_durations = []

    for index, node in enumerate(nodes_to_process, start=1):
        chunk_start = time.perf_counter()
        try:
            print(f"\n--- PROCESSING CHUNK {index}/{len(nodes_to_process)} ---")
            print(node.text[:300], "...")

            if is_overview_or_agenda_chunk(node.text):
                total_empty_chunks += 1
                print("\n--- OVERVIEW / AGENDA CHUNK SKIPPED ---")
                continue

            if is_revision_list_chunk(node.text):
                total_empty_chunks += 1
                print("\n--- REVISION LIST CHUNK SKIPPED ---")
                continue

            raw_output = extract_kg_json(node.text, llm)
            print("\n--- RAW MODEL OUTPUT ---")
            print(raw_output)

            extraction = extract_json_object(raw_output)

            module_name = node.metadata.get("module")
            week_number = node.metadata.get("week")

            graph_payload = build_graph_payload(
                extraction,
                node.text,
                module_name,
                week_number,
                node.metadata,
            )

            print_graph_payload(graph_payload)

            if not graph_payload["nodes"] or not graph_payload["relationships"]:
                total_empty_chunks += 1
                print("\n--- CHUNK PRODUCED NO VALID GRAPH DATA. SKIPPED. ---")
                continue

            # Extract evidence for separate output
            chunk_evidence = extract_evidence_from_chunk(node, extraction, graph_payload)
            update_evidence_store(accumulated_evidence, chunk_evidence)

            if RUN_SHACL_VALIDATION:
                try:
                    conforms, validation_report = validate_graph_payload_with_shacl(graph_payload)
                except Exception as error:
                    print(f"\n--- SHACL VALIDATION SKIPPED (error: {error}) ---")
                    conforms = True

                if not conforms:
                    total_failed_chunks += 1
                    print("\n--- SHACL VALIDATION FAILED ---")
                    print(validation_report)
                    print("Chunk skipped. Nothing added to output JSON.")
                    continue

                print("\n--- SHACL VALIDATION PASSED ---")

            merge_graph_payload(accumulated_payload, graph_payload)

            total_validated_chunks += 1
            total_nodes_saved = len(accumulated_payload["nodes"])
            total_relationships_saved = len(accumulated_payload["relationships"])

            print(
                f"Merged into running graph. Running totals: "
                f"{total_nodes_saved} nodes, {total_relationships_saved} relationships."
            )
        finally:
            chunk_elapsed = time.perf_counter() - chunk_start
            chunk_durations.append(chunk_elapsed)
            print(f"Chunk {index} processed in {format_duration(chunk_elapsed)}")

    # Save Ontology Knowledge Graph JSON
    Path(OUTPUT_JSON_FILE).parent.mkdir(parents=True, exist_ok=True)
    Path(EVIDENCE_JSON_FILE).parent.mkdir(parents=True, exist_ok=True)

    final_json = graph_payload_to_json_friendly(accumulated_payload)
    with open(OUTPUT_JSON_FILE, "w", encoding="utf-8") as f:
        json.dump(final_json, f, indent=2, ensure_ascii=False)

    # Save Lecture Evidence JSON
    with open(EVIDENCE_JSON_FILE, "w", encoding="utf-8") as f:
        json.dump(accumulated_evidence, f, indent=2, ensure_ascii=False)

    run_elapsed = time.perf_counter() - run_start

    print("\nKG extraction complete.")
    print(f"Validated chunks merged: {total_validated_chunks}")
    print(f"Failed SHACL chunks: {total_failed_chunks}")
    print(f"Empty/noisy chunks skipped: {total_empty_chunks}")
    print(f"Total nodes in ontology output: {len(accumulated_payload['nodes'])}")
    print(f"Total relationships in ontology output: {len(accumulated_payload['relationships'])}")
    print(f"Ontology JSON saved to: {OUTPUT_JSON_FILE}")
    print(f"Evidence JSON saved to: {EVIDENCE_JSON_FILE}")

    if chunk_durations:
        avg_chunk_time = sum(chunk_durations) / len(chunk_durations)
        max_chunk_time = max(chunk_durations)
        print(f"Average time per chunk: {format_duration(avg_chunk_time)}")
        print(f"Slowest chunk: {format_duration(max_chunk_time)}")

    print(f"Total run time: {format_duration(run_elapsed)}")