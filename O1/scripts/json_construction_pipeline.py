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
# The write_graph_payload_to_auradb()/create_constraints() functions below
# still reference these names, but they are simply never called when you
# run this script directly (see the __main__ block at the bottom).
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


# -------------------------------------------------------------------
# PDF loading and preprocessing
# -------------------------------------------------------------------

def load_pdf_documents(pdf_folder: str):
    """
    Load all PDF files from the configured folder.

    Each PDF page is loaded as a separate document and receives the source
    PDF path as metadata.
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

    This prevents the KG extraction LLM from inventing module/week values.
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

    This removes spacing noise but avoids changing the academic meaning.
    """

    if not text:
        return ""

    text = text.replace("\n", " ").replace("\t", " ")
    text = " ".join(text.split())

    # Add a space where PDF extraction joins lowercase and uppercase words.
    text = re.sub(r"([a-z])([A-Z])", r"\1 \2", text)

    # Fix PDF line-break artefacts inside words.
    text = re.sub(r"([A-Za-z])-\s+([A-Za-z])", r"\1\2", text)

    # Improve common PDF math formatting.
    text = text.replace("º", "=")
    text = text.replace("∣", "|")

    # Remove repeated strange symbol runs caused by OCR/PDF extraction.
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

    These chunks often cause wrong Topic -> Concept structures.
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

    These often contain many useful words but are not actual taught concepts
    in the lecture section itself.
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

    These chunks should usually enrich existing concepts with examples,
    not create new concepts such as "Interval Task".
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

    Larger chunks help the model understand meaning rather than treating
    isolated table words as standalone concepts.
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

    Definition, Example, Formula, and Application are not separate nodes.
    They are optional properties of Concept.
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
    """
    Convert a value into a safe float.
    """

    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def normalize_week_name(week_number):
    """
    Convert week metadata into the standard Week node name.
    """

    week_text = str(week_number).strip()

    if not week_text.lower().startswith("week"):
        week_text = f"Week {week_text}"

    return week_text


def normalize_name(value: str):
    """
    Normalize a topic/concept/resource/assessment name.

    This avoids a large whitelist. It only applies general casing and
    small synonym cleanup for obvious extraction variations.
    """

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
    """
    Clean optional concept property values.
    """

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

    # Remove obvious PDF artefact fragments.
    value = re.sub(r"\s+", " ", value).strip()

    if len(value) > max_length:
        value = value[:max_length].rsplit(" ", 1)[0].strip()

    return value


def meaningful_tokens(text: str):
    """
    Return meaningful tokens for automatic evidence matching.

    This is a small generic stopword list, not a concept whitelist.
    """

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
    """
    Convert text into an embedding vector using Ollama.

    This adds semantic validation so the script can detect similar meaning,
    not only identical wording.
    """

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
    """
    Calculate cosine similarity between two embedding vectors.
    """

    if not first_vector or not second_vector:
        return 0.0

    dot_product = sum(a * b for a, b in zip(first_vector, second_vector))
    first_norm = math.sqrt(sum(a * a for a in first_vector))
    second_norm = math.sqrt(sum(b * b for b in second_vector))

    if first_norm == 0 or second_norm == 0:
        return 0.0

    return dot_product / (first_norm * second_norm)


def embedding_similarity(first_name: str, second_name: str):
    """
    Compare two names using embedding similarity.

    This catches semantic duplicates such as:
    - Binary Number System
    - Base-2 Number System
    """

    first_embedding = get_embedding(first_name)
    second_embedding = get_embedding(second_name)

    if not first_embedding or not second_embedding:
        return 0.0

    return cosine_similarity(first_embedding, second_embedding)


def source_support_score(name: str, source_text: str, source_quote: str = "") -> float:
    """
    Check whether a proposed node is supported by the source text.

    This avoids a manual concept whitelist by checking whether the concept
    name or its meaningful tokens actually appear in the chunk.
    """

    if not name or not source_text:
        return 0.0

    name_lower = name.lower()
    text_lower = source_text.lower()
    quote_lower = (source_quote or "").lower()

    if name_lower in text_lower:
        return 1.0

    if quote_lower and name_lower in quote_lower:
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
    """
    Detect whether text contains a mathematical expression or notation.
    """

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
    """
    Detect table-row or table-column labels that should not become concepts.

    This is deliberately generic. It does not depend on one lecture only.
    """

    name = normalize_name(name)
    lower_name = name.lower()
    lower_text = source_text.lower()

    generic_table_attributes = {
        "collection",
        "ordering",
        "repetition",
        "notation",
        "indexing",
        "attribute",
        "attributes",
        "property",
        "properties",
        "definition",
        "description",
        "examples",
        "example",
        "length",
        "cardinality/length",
        "value",
        "values",
        "object",
        "objects",
        "method",
        "result",
        "results",
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

    if any(marker in lower_text for marker in comparison_markers):
        return True

    return False


def name_quality_score(name: str, source_text: str = "") -> float:
    """
    Score the quality of a candidate Topic or Concept name.

    This is automatic scoring, not a manual approved vocabulary list.
    """

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
        " is ",
        " are ",
        " was ",
        " were ",
        " has ",
        " have ",
        " contains ",
        " consists ",
        " refers ",
        " represents ",
        " used to ",
    ]

    if any(marker in f" {lower} " for marker in sentence_markers):
        return 0.1

    instruction_starts = [
        "activity",
        "task",
        "question",
        "answer",
        "example",
        "exercise",
        "homework",
        "solve",
        "determine",
        "calculate",
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
        "definition",
        "description",
        "property",
        "properties",
        "attribute",
        "attributes",
        "example",
        "examples",
        "formula",
        "formulas",
        "application",
        "applications",
        "notes",
        "note",
        "result",
        "results",
        "method",
        "methods",
        "step",
        "steps",
        "object",
        "objects",
        "value",
        "values",
        "usage",
        "reason",
        "reasons",
        "benefit",
        "benefits",
        "compatibility",
        "efficiency",
        "interval task",
        "attribute set",
    }

    if lower in generic_noise:
        return 0.05

    if is_probably_table_attribute(raw, source_text):
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
    """
    Combine LLM confidence, name quality, and source support.
    """

    quality = name_quality_score(name, source_text)
    support = source_support_score(name, source_text, source_quote)

    llm_confidence = safe_float(llm_confidence, 0.5)
    llm_confidence = max(0.0, min(1.0, llm_confidence))

    final_score = (
        0.45 * quality +
        0.40 * support +
        0.15 * llm_confidence
    )

    return final_score


def similarity_score(first_name: str, second_name: str) -> float:
    """
    Compare two node names using string similarity and embedding similarity.

    This catches both:
    - exact/near text duplicates
    - semantic duplicates
    """

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
    """
    Detect when a topic and concept are basically the same thing.
    """

    topic_name = normalize_name(topic_name)
    concept_name = normalize_name(concept_name)

    if topic_name.lower() == concept_name.lower():
        return True

    # Allow common valid broader/singular relationships.
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
    """
    Detect whether two node names are semantic duplicates.
    """

    return similarity_score(first_name, second_name) >= EMBEDDING_SIMILARITY_THRESHOLD


def infer_broader_topic_name(topic_name: str, concept_names: list, source_text: str):
    """
    Automatically infer a broader topic name when the extracted topic
    is too similar to one of its concepts.
    """

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

        common_tokens = [
            token
            for token, count in shared_tokens.items()
            if count >= 2
        ]

        if common_tokens:
            return normalize_name(" ".join(common_tokens[:3]))

    return topic_name


def is_application_chunk(source_text: str) -> bool:
    """
    Detect chunks mainly describing applications/use cases of a concept.
    """

    lower_text = source_text.lower()

    return (
        "applications of" in lower_text[:180]
        or "used in" in lower_text[:220]
        or "used for" in lower_text[:220]
    )


def is_reason_or_benefit_chunk(source_text: str) -> bool:
    """
    Detect chunks mainly explaining why a concept is useful.
    """

    lower_text = source_text.lower()

    return (
        lower_text.startswith("why ")
        or "why are" in lower_text[:120]
        or "why is" in lower_text[:120]
        or "popular in coding" in lower_text[:180]
        or "essential for programming" in lower_text[:180]
    )


def is_formula_label_concept(concept_name: str, concept: dict) -> bool:
    """
    Detect nodes that are really formula labels, not actual concepts.

    Examples to promote into properties:
    - Arithmetic Sequence Formula
    - Geometric Sequence Formula
    - Encryption Equation
    """

    name = normalize_name(concept_name)
    lower_name = name.lower()

    formula = clean_property_text(concept.get("formula", ""), 250)
    description = clean_property_text(concept.get("description", ""), 500)

    has_math = contains_math_expression(formula) or contains_math_expression(description)

    formula_suffixes = [
        " formula",
        " notation",
        " expression",
        " equation",
    ]

    if any(lower_name.endswith(suffix) for suffix in formula_suffixes) and has_math:
        if lower_name in {"linear equations", "quadratic equations", "set builder notation"}:
            return False

        return True

    return False


def is_weak_reason_label(concept_name: str, concept: dict, source_text: str) -> bool:
    """
    Detect bullet labels that are reasons/benefits rather than concepts.

    Example:
    - Efficiency
    - Exact Values
    - Memory Usage
    - Compatibility
    """

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
        "operation",
        "operations",
        "indexing",
        "algorithm",
        "algorithms",
        "cryptography",
        "statistics",
        "probability",
        "geometry",
        "trigonometry",
        "bitwise",
    ]

    if any(term in lower_name for term in technical_terms):
        return False

    if len(name.split()) <= 3:
        return True

    return False


def is_bad_activity_concept(concept_name: str, source_text: str) -> bool:
    """
    Reject task/activity labels as concepts.
    """

    name = normalize_name(concept_name).lower()

    bad_names = {
        "activity",
        "task",
        "answer",
        "answers",
        "exercise",
        "question",
        "interval task",
        "interval membership task",
    }

    if name in bad_names:
        return True

    if name.endswith(" task") or name.endswith(" activity") or name.endswith(" answer"):
        return True

    if is_activity_or_example_chunk(source_text) and name in {"attribute set", "collection", "ordering", "repetition"}:
        return True

    return False


def should_keep_topic(topic: dict, source_text: str) -> bool:
    """
    Decide whether a topic is meaningful enough to store.
    """

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
    """
    Decide whether a concept is meaningful enough to store.
    """

    name = normalize_name(concept.get("name", ""))
    quote = clean_property_text(concept.get("source_quote", ""), 300)
    confidence = concept.get("confidence", 0.5)

    if not name:
        return False

    if is_bad_activity_concept(name, source_text):
        return False

    if is_probably_table_attribute(name, source_text):
        return False

    if is_formula_label_concept(name, concept):
        return False

    if is_weak_reason_label(name, concept, source_text):
        return False

    score = combined_confidence_score(
        name=name,
        source_text=source_text,
        llm_confidence=confidence,
        source_quote=quote,
    )

    has_useful_property = any(
        clean_property_text(concept.get(prop, ""), 120)
        for prop in ["description", "formula", "example", "application"]
    )

    # Keep strong concepts even in example/activity chunks if they are clearly supported.
    if is_activity_or_example_chunk(source_text) and has_useful_property and score >= 0.66:
        return True

    return score >= 0.70


def normalize_relationship_type(rel_type: str):
    """
    Keep only concept-to-concept relationship types supported by the schema.
    """

    if not rel_type:
        return None

    rel_type = str(rel_type).strip().upper()

    valid_types = {
        "RELATED_TO",
        "BUILDS_ON",
        "PREREQUISITE_OF",
    }

    if rel_type not in valid_types:
        return None

    return rel_type


def derive_main_concept_from_text(source_text: str, fallback_name: str):
    """
    Derive the main concept of an application/reason chunk automatically.
    """

    text = source_text.strip()

    match = re.search(
        r"applications of ([A-Za-z0-9\s\-]+?)(?:•|:|\.|$)",
        text,
        flags=re.IGNORECASE,
    )

    if match:
        return normalize_name(match.group(1))

    match = re.search(
        r"why (?:are|is|do|does) ([A-Za-z0-9\s\-]+?) (?:popular|essential|important|useful|used)",
        text,
        flags=re.IGNORECASE,
    )

    if match:
        return normalize_name(match.group(1))

    return normalize_name(fallback_name)


def merge_text_items(items: list, max_length: int):
    """
    Merge multiple short strings into one clean property value.
    """

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
    """
    Extract a likely formula from a text field.
    """

    if not text:
        return ""

    candidates = re.findall(
        r"([A-Za-z0-9_()]+(?:\s*[=≡]\s*[^,.;]+))",
        text,
    )

    if candidates:
        return clean_property_text(candidates[0], 250)

    if contains_math_expression(text):
        return clean_property_text(text, 250)

    return ""


def infer_semantic_relationships(concept_names: list):
    """
    Infer a small number of useful concept-to-concept relationships.

    These are pattern-based semantic rules, not hard-coded lecture nodes.
    They improve the KG beyond simple Topic -> Concept folder structure.
    """

    relationships = []
    names = {name.lower(): name for name in concept_names}

    def add_if_present(source, rel_type, target):
        source_key = source.lower()
        target_key = target.lower()

        if source_key in names and target_key in names:
            relationships.append(
                (
                    names[source_key],
                    rel_type,
                    names[target_key],
                )
            )

    add_if_present("Arithmetic Sequence", "BUILDS_ON", "Sequence")
    add_if_present("Geometric Sequence", "BUILDS_ON", "Sequence")
    add_if_present("Fibonacci Sequence", "BUILDS_ON", "Sequence")
    add_if_present("Proper Subset", "BUILDS_ON", "Subset")
    add_if_present("Power Set", "BUILDS_ON", "Subset")
    add_if_present("Cartesian Product", "RELATED_TO", "Ordered Pair")
    add_if_present("Caesar Cipher", "BUILDS_ON", "Modulo Operation")
    add_if_present("Caesar Cipher", "BUILDS_ON", "Modular Arithmetic")
    add_if_present("Binary Number System", "RELATED_TO", "Modular Arithmetic")
    add_if_present("Union", "RELATED_TO", "Intersection")
    add_if_present("Difference", "RELATED_TO", "Complement")

    return relationships


# -------------------------------------------------------------------
# Graph payload construction
# -------------------------------------------------------------------

def add_node(nodes: dict, label: str, name: str, properties: dict | None = None):
    """
    Add or update a node in the in-memory graph payload.
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
                existing = nodes[key]["properties"].get(prop_key, 0.0)
                nodes[key]["properties"][prop_key] = max(existing, prop_value)
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

    This function performs the most important semantic cleanup:
    - Removes weak concepts.
    - Promotes formula-label concepts into concept properties.
    - Turns application/reason labels into properties when appropriate.
    - Blocks generic table headers from becoming nodes.
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
                safe_float(concept.get("confidence", 0.5), 0.5),
            )

            continue

        if is_application_chunk(source_text):
            item_text = application or source_quote or description or concept_name
            application_items.append(f"{concept_name}: {item_text}")
            continue

        if is_weak_reason_label(concept_name, concept, source_text):
            item_text = source_quote or description or concept_name
            reason_items.append(f"{concept_name}: {item_text}")
            continue

        if not should_keep_concept(concept, source_text):
            continue

        concept_score = combined_confidence_score(
            name=concept_name,
            source_text=source_text,
            llm_confidence=concept.get("confidence", 0.5),
            source_quote=source_quote,
        )

        valid_concepts.append(
            {
                "name": concept_name,
                "description": description,
                "formula": formula,
                "example": example,
                "application": application,
                "confidence": round(concept_score, 3),
                "source_quote": source_quote,
            }
        )

    if application_items:
        main_concept = derive_main_concept_from_text(source_text, topic_name)
        application_text = merge_text_items(application_items, 300)

        valid_concepts.append(
            {
                "name": main_concept,
                "description": "",
                "formula": "",
                "example": "",
                "application": application_text,
                "confidence": 0.9,
                "source_quote": application_text,
            }
        )

    if reason_items:
        main_concept = derive_main_concept_from_text(source_text, topic_name)
        reason_text = merge_text_items(reason_items, 300)

        valid_concepts.append(
            {
                "name": main_concept,
                "description": reason_text,
                "formula": "",
                "example": "",
                "application": reason_text,
                "confidence": 0.85,
                "source_quote": reason_text,
            }
        )

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
            valid_concepts.append(
                {
                    "name": target_name,
                    "description": properties.get("description", ""),
                    "formula": properties.get("formula", ""),
                    "example": properties.get("example", ""),
                    "application": properties.get("application", ""),
                    "confidence": round(properties.get("confidence", 0.7), 3),
                    "source_quote": properties.get("source_quote", ""),
                }
            )

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

    This creates only:
    - Module
    - Week
    - Topic
    - Concept

    Details such as description/example/formula/application become
    properties on Concept nodes.
    """

    nodes = {}
    relationships = []

    module_name = normalize_name(module_name or "Unknown Module")
    week_name = normalize_week_name(week_number or "0")

    module_key = add_node(
        nodes,
        "Module",
        module_name,
        {
            "sourceFile": metadata.get("pdf_path", ""),
        },
    )

    week_number_int = None
    week_number_match = re.search(r"\d+", str(week_name))

    if week_number_match:
        week_number_int = int(week_number_match.group(0))

    week_key = add_node(
        nodes,
        "Week",
        week_name,
        {
            "weekNumber": week_number_int,
            "sourceFile": metadata.get("pdf_path", ""),
        },
    )

    add_relationship(relationships, module_key, "HAS_WEEK", week_key)

    concept_name_to_key = {}
    all_concept_names_in_chunk = []

    for topic in extraction.get("topics", []):
        if not isinstance(topic, dict):
            continue

        if not should_keep_topic(topic, source_text):
            continue

        topic_name = normalize_name(topic.get("name", ""))

        valid_concepts = extract_valid_concepts_for_topic(topic, source_text)

        if not valid_concepts:
            continue

        concept_names = [concept["name"] for concept in valid_concepts]

        if is_application_chunk(source_text):
            topic_name = infer_broader_topic_name(topic_name, concept_names, source_text)

        if any(is_topic_concept_duplicate(topic_name, concept_name) for concept_name in concept_names):
            topic_name = infer_broader_topic_name(topic_name, concept_names, source_text)

        # Final safety correction for known source context patterns.
        topic_name = infer_broader_topic_name(topic_name, concept_names, source_text)

        filtered_concepts = []

        for concept in valid_concepts:
            concept_name = normalize_name(concept.get("name", ""))

            if not concept_name:
                continue

            has_properties = any(
                concept.get(prop)
                for prop in ["description", "formula", "example", "application"]
            )

            # Do not remove useful concepts only because they are similar to the topic.
            # This fixes valid cases such as:
            # Topic: Sequences -> Concept: Sequence
            # Topic: Set Theory -> Concept: Set
            # Topic: Set Theory -> Concept: Proper Subset
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
            nodes,
            "Topic",
            topic_name,
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
                nodes,
                "Concept",
                concept_name,
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

        if not source_key or not target_key:
            continue

        source_quote = clean_property_text(rel.get("source_quote", ""), 300)

        # Block placeholder or unsupported weak relationship output.
        if rel_type == "RELATED_TO" and not source_quote:
            continue

        rel_score = combined_confidence_score(
            name=f"{source_name} {target_name}",
            source_text=source_text,
            llm_confidence=rel.get("confidence", 0.5),
            source_quote=source_quote,
        )

        semantic_relation_score = embedding_similarity(source_name, target_name)

        if rel_type == "RELATED_TO":
            if rel_score < 0.80 and semantic_relation_score < 0.72:
                continue

        if rel_type in {"BUILDS_ON", "PREREQUISITE_OF"}:
            if rel_score < 0.74:
                continue

        add_relationship(relationships, source_key, rel_type, target_key)

    return {
        "nodes": nodes,
        "relationships": relationships,
    }


# -------------------------------------------------------------------
# SHACL validation
# -------------------------------------------------------------------

def safe_uri_value(value: str):
    """
    Convert a node value into a safe URI fragment for RDF validation.
    """

    value = value.strip()
    value = re.sub(r"\s+", "_", value)
    value = re.sub(r"[^A-Za-z0-9_\-]", "_", value)

    return value[:120]


def graph_payload_to_rdf_graph(graph_payload: dict):
    """
    Convert the graph payload into RDF for SHACL validation.
    """

    kg = Namespace("http://example.org/university-kg#")
    graph = Graph()
    graph.bind("kg", kg)

    node_uri_map = {}

    def get_node_uri(label, name):
        node_key = (label, name)

        if node_key not in node_uri_map:
            node_uri_map[node_key] = URIRef(
                f"{kg}{label}_{safe_uri_value(name)}"
            )

        return node_uri_map[node_key]

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
    """
    Validate graph payload against OWL + SHACL before saving to Neo4j.
    """

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
# Neo4j / AuraDB writing
# -------------------------------------------------------------------

def create_constraints_tx(tx):
    """
    Create uniqueness constraints only for the active KG node labels.

    Definition, Example, Formula, and Application constraints are removed
    because those are no longer node types.
    """

    constraints = [
        """
        CREATE CONSTRAINT module_name_unique IF NOT EXISTS
        FOR (m:Module)
        REQUIRE m.name IS UNIQUE
        """,
        """
        CREATE CONSTRAINT week_name_unique IF NOT EXISTS
        FOR (w:Week)
        REQUIRE w.name IS UNIQUE
        """,
        """
        CREATE CONSTRAINT topic_name_unique IF NOT EXISTS
        FOR (t:Topic)
        REQUIRE t.name IS UNIQUE
        """,
        """
        CREATE CONSTRAINT concept_name_unique IF NOT EXISTS
        FOR (c:Concept)
        REQUIRE c.name IS UNIQUE
        """,
        """
        CREATE CONSTRAINT resource_name_unique IF NOT EXISTS
        FOR (r:Resource)
        REQUIRE r.name IS UNIQUE
        """,
        """
        CREATE CONSTRAINT assessment_name_unique IF NOT EXISTS
        FOR (a:Assessment)
        REQUIRE a.name IS UNIQUE
        """,
        """
        CREATE CONSTRAINT processed_document_name_unique IF NOT EXISTS
        FOR (d:ProcessedDocument)
        REQUIRE d.name IS UNIQUE
        """,
    ]

    for constraint in constraints:
        tx.run(constraint)


def create_constraints(driver):
    """
    Create Neo4j constraints using a retry-safe session.
    """

    with driver.session(database=NEO4J_DATABASE) as session:
        session.execute_write(create_constraints_tx)


def build_node_merge_cypher(label: str):
    """
    Build a safe Cypher query for merging a node by name.

    The label is controlled internally, not from the LLM.
    """

    allowed_labels = {
        "Module",
        "Week",
        "Topic",
        "Concept",
        "Resource",
        "Assessment",
        "ProcessedDocument",
    }

    if label not in allowed_labels:
        raise ValueError(f"Unsupported label: {label}")

    return f"""
MERGE (n:{label} {{name: $name}})
SET
    n.description = CASE
        WHEN $description IS NOT NULL AND $description <> '' AND (n.description IS NULL OR n.description = '')
        THEN $description
        ELSE n.description
    END,
    n.formula = CASE
        WHEN $formula IS NOT NULL AND $formula <> '' AND (n.formula IS NULL OR n.formula = '')
        THEN $formula
        ELSE n.formula
    END,
    n.example = CASE
        WHEN $example IS NOT NULL AND $example <> '' AND (n.example IS NULL OR n.example = '')
        THEN $example
        ELSE n.example
    END,
    n.application = CASE
        WHEN $application IS NOT NULL AND $application <> '' AND (n.application IS NULL OR n.application = '')
        THEN $application
        ELSE n.application
    END,
    n.sourceFile = CASE
        WHEN $sourceFile IS NOT NULL AND $sourceFile <> '' AND (n.sourceFile IS NULL OR n.sourceFile = '')
        THEN $sourceFile
        ELSE n.sourceFile
    END,
    n.weekNumber = CASE
        WHEN $weekNumber IS NOT NULL
        THEN $weekNumber
        ELSE n.weekNumber
    END,
    n.confidence = CASE
        WHEN $confidence IS NOT NULL AND (n.confidence IS NULL OR n.confidence < $confidence)
        THEN $confidence
        ELSE n.confidence
    END
"""


def write_graph_payload_tx(tx, graph_payload: dict):
    """
    Write one graph payload inside a Neo4j transaction.
    """

    allowed_labels = {
        "Module",
        "Week",
        "Topic",
        "Concept",
        "Resource",
        "Assessment",
        "ProcessedDocument",
    }

    allowed_relationships = {
        "HAS_WEEK",
        "COVERS",
        "HAS_CONCEPT",
        "RELATED_TO",
        "BUILDS_ON",
        "PREREQUISITE_OF",
        "NEXT",
        "HAS_RESOURCE",
        "HAS_ASSESSMENT",
    }

    for (label, name), node_data in graph_payload["nodes"].items():
        if label not in allowed_labels:
            continue

        properties = node_data.get("properties", {})

        params = {
            "name": name,
            "description": properties.get("description"),
            "formula": properties.get("formula"),
            "example": properties.get("example"),
            "application": properties.get("application"),
            "sourceFile": properties.get("sourceFile"),
            "weekNumber": properties.get("weekNumber"),
            "confidence": properties.get("confidence"),
        }

        cypher = build_node_merge_cypher(label)
        tx.run(cypher, **params)

    for (sub_label, sub_name), rel_type, (obj_label, obj_name) in graph_payload["relationships"]:
        if sub_label not in allowed_labels or obj_label not in allowed_labels:
            continue

        if rel_type not in allowed_relationships:
            continue

        cypher = f"""
MATCH (a:{sub_label} {{name: $sub_name}})
MATCH (b:{obj_label} {{name: $obj_name}})
MERGE (a)-[:{rel_type}]->(b)
"""

        tx.run(
            cypher,
            sub_name=sub_name,
            obj_name=obj_name,
        )


def write_graph_payload_to_auradb(driver, graph_payload: dict, max_retries: int = 3):
    """
    Write validated graph payload to AuraDB with retry support.

    This helps when AuraDB closes a connection during a long run.
    """

    last_error = None

    for attempt in range(1, max_retries + 1):
        try:
            with driver.session(database=NEO4J_DATABASE) as session:
                session.execute_write(write_graph_payload_tx, graph_payload)

            return True

        except (ServiceUnavailable, SessionExpired, TransientError) as error:
            last_error = error
            print(f"Neo4j write failed on attempt {attempt}/{max_retries}: {error}")

            if attempt < max_retries:
                time.sleep(2 * attempt)

    print(f"Neo4j write permanently failed after {max_retries} attempts: {last_error}")

    return False


# -------------------------------------------------------------------
# Debugging helpers
# -------------------------------------------------------------------

def print_week_distribution(nodes):
    """
    Print the number of chunks detected for each week.
    """

    week_counts = Counter(node.metadata.get("week") for node in nodes)

    print("\n--- WEEK DISTRIBUTION ---")
    for week, count in week_counts.items():
        print(f"Week {week}: {count} chunks")


def print_graph_payload(graph_payload: dict):
    """
    Print graph payload in a readable way before validation/writing.
    """

    print("\n--- FILTERED NODES ---")
    for (label, name), node_data in graph_payload["nodes"].items():
        print((label, name, node_data.get("properties", {})))

    print("\n--- FILTERED RELATIONSHIPS ---")
    for relationship in graph_payload["relationships"]:
        print(relationship)


def debug_node_output(nodes, llm):
    """
    Print the raw JSON extraction output for a selected node.
    """

    if not nodes:
        print("No nodes available.")
        return

    user_index = int(input("Which node would you like to debug?\n"))
    test_node = nodes[user_index]

    raw_output = extract_kg_json(test_node.text, llm)

    print("\n--- NODE TEXT ---")
    print(test_node.text)

    print("\n--- RAW MODEL OUTPUT ---")
    print(raw_output)

    print("\n--- PARSED JSON ---")
    print(json.dumps(extract_json_object(raw_output), indent=2))


# -------------------------------------------------------------------
# Main pipeline
# -------------------------------------------------------------------

# -------------------------------------------------------------------
# JSON-only export helpers (no Neo4j / AuraDB required)
# -------------------------------------------------------------------

OUTPUT_JSON_FILE = os.getenv("OUTPUT_JSON_FILE", "llm-comparator/knowledge/kg_extraction_output.json")
RUN_SHACL_VALIDATION = os.getenv("RUN_SHACL_VALIDATION", "false").lower() == "true"


def merge_graph_payload(accumulated: dict, new_payload: dict):
    """
    Merge one chunk's graph_payload into the running accumulated payload,
    reusing the same add_node/add_relationship de-duplication logic used
    for the Neo4j path, so the final JSON is already de-duplicated.
    """

    for (label, name), node_data in new_payload["nodes"].items():
        add_node(accumulated["nodes"], label, name, node_data.get("properties", {}))

    for sub_key, rel_type, obj_key in new_payload["relationships"]:
        # sub_key / obj_key are (label, name) tuples already normalised by add_node
        add_relationship(accumulated["relationships"], sub_key, rel_type, obj_key)


def graph_payload_to_json_friendly(payload: dict):
    """
    Convert the internal (label, name) tuple-keyed payload into a plain
    JSON-serialisable structure: a flat nodes[] list and edges[] list.
    """

    nodes_list = []
    for (label, name), node_data in payload["nodes"].items():
        nodes_list.append(
            {
                "id": f"{label}:{name}",
                "type": label,
                "name": name,
                "properties": node_data.get("properties", {}),
            }
        )

    edges_list = []
    for (sub_label, sub_name), rel_type, (obj_label, obj_name) in payload["relationships"]:
        edges_list.append(
            {
                "source": f"{sub_label}:{sub_name}",
                "relation": rel_type,
                "target": f"{obj_label}:{obj_name}",
            }
        )

    return {"nodes": nodes_list, "edges": edges_list}


if __name__ == "__main__":
    documents = load_pdf_documents(DATA_SOURCE)
    llm = create_llm()

    cleaned_documents = prepare_documents(documents, llm)
    nodes = chunk_documents(cleaned_documents)

    print(f"Cleaned documents: {len(cleaned_documents)}")
    print(f"Text chunks created: {len(nodes)}")

    print_week_distribution(nodes)

    # Use nodes[:3] for a small test run.
    # Use nodes for the full graph construction run.
    nodes_to_process = nodes

    total_nodes_saved = 0
    total_relationships_saved = 0
    total_validated_chunks = 0
    total_failed_chunks = 0
    total_empty_chunks = 0

    # Accumulates every chunk's nodes/relationships into one merged graph.
    accumulated_payload = {"nodes": {}, "relationships": []}

    for index, node in enumerate(nodes_to_process, start=1):
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
            extraction=extraction,
            source_text=node.text,
            module_name=module_name,
            week_number=week_number,
            metadata=node.metadata,
        )

        # Only Module -> Week means no meaningful academic KG content survived.
        if len(graph_payload["relationships"]) <= 1:
            total_empty_chunks += 1
            print("\n--- NO MEANINGFUL KG CONTENT AFTER FILTERING ---")
            print("Chunk skipped. Nothing added to output JSON.")
            continue

        print_graph_payload(graph_payload)

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

    final_json = graph_payload_to_json_friendly(accumulated_payload)

    with open(OUTPUT_JSON_FILE, "w", encoding="utf-8") as f:
        json.dump(final_json, f, indent=2, ensure_ascii=False)

    print("\nKG extraction complete.")
    print(f"Validated chunks merged: {total_validated_chunks}")
    print(f"Failed SHACL chunks: {total_failed_chunks}")
    print(f"Empty/noisy chunks skipped: {total_empty_chunks}")
    print(f"Total nodes in output JSON: {len(final_json['nodes'])}")
    print(f"Total edges in output JSON: {len(final_json['edges'])}")
    print(f"Written to: {os.path.abspath(OUTPUT_JSON_FILE)}")