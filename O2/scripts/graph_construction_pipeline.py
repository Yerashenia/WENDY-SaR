import os
import re
import json
import time
import logging
from pathlib import Path
from collections import Counter
from difflib import SequenceMatcher

from dotenv import load_dotenv
from neo4j import GraphDatabase
from neo4j.exceptions import ServiceUnavailable, SessionExpired, TransientError

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

ONTOLOGY_FILE = os.getenv("ONTOLOGY_FILE", "validation_schema/kg_ontology.owl")
SHACL_FILE = os.getenv("SHACL_FILE", "validation_schema/kg_validation_shapes.ttl")


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

    if symbol_ratio > 0.45:
        return True

    words = re.findall(r"[A-Za-z]{3,}", text)

    if len(words) < 8:
        return True

    return False


def is_overview_or_agenda_chunk(text: str) -> bool:
    """
    Detect title/agenda chunks that only list lecture sections.

    These chunks often cause wrong Topic -> Concept structures, for example:
    Topic: Sequences -> Concept: Number Formats

    We skip them unless they contain real explanatory/definition content.
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

IMPORTANT NODE QUALITY RULES:
- Extract only meaningful academic concepts.
- Do not extract single letters, variables, table column labels, activities, questions, answers, or raw formulas as concepts.
- Do not extract full sentences as concept names.
- Do not extract symbols or example sets as concept names.
- Do not create a concept named "Formula", "Equation", "Notation", or "Expression" unless it is a real taught concept.
- If a formula belongs to a concept, put it in that concept's "formula" field.
- If the text is about applications of a concept, create the main concept and put the applications in its "application" field.
- If the text is about reasons/benefits of a concept, create the main concept and put the reasons in its "application" or "description" field.
- If the chunk only lists lecture headings or revision topics, return empty topics.
- If there is no valid academic content, return:
{{"topics": [], "relationships": []}}

JSON FORMAT:
{{
  "topics": [
    {{
      "name": "Set Theory",
      "confidence": 0.92,
      "source_quote": "short exact quote from the text that supports this topic",
      "concepts": [
        {{
          "name": "Subset",
          "description": "A set A is a subset of set B if every element of A is also an element of B.",
          "formula": "A ⊆ B",
          "example": "A={{1,2}}, B={{1,2,3}}, so A⊆B",
          "application": "",
          "confidence": 0.95,
          "source_quote": "short exact quote from the text that supports this concept"
        }}
      ]
    }}
  ],
  "relationships": [
    {{
      "source": "Subset",
      "type": "RELATED_TO",
      "target": "Proper Subset",
      "confidence": 0.80,
      "source_quote": "short exact quote from the text supporting this relationship"
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
    value = value.strip(" .,:;")
    value = value.strip("\"'“”‘’")

    if not value:
        return ""

    lower_value = value.lower()

    small_synonym_map = {
        "sets": "Set",
        "natural numbers": "Natural Numbers",
        "natural number": "Natural Numbers",
        "integer": "Integers",
        "integers": "Integers",
        "prime number": "Prime Numbers",
        "prime numbers": "Prime Numbers",
        "modular mathematics": "Modular Arithmetic",
        "modular arithmetic": "Modular Arithmetic",
        "binary number system": "Binary Number System",
        "decimal number system": "Decimal Number System",
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
        else:
            fixed_words.append(word[:1].upper() + word[1:].lower())

    return " ".join(fixed_words)


def clean_property_text(value: str, max_length: int):
    """
    Clean optional concept property values.
    """

    if not value:
        return ""

    value = str(value)
    value = re.sub(r"\s+", " ", value).strip()

    bad_values = {
        "none",
        "n/a",
        "null",
        "not provided",
        "(not provided)",
        "unknown",
    }

    if value.lower() in bad_values:
        return ""

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
        "main", "basic", "basics", "introduction",
    }

    tokens = re.findall(r"[A-Za-z0-9]+", str(text).lower())

    return [
        token
        for token in tokens
        if len(token) >= 3 and token not in stopwords
    ]


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
    ]

    lower_text = str(text).lower()

    return any(symbol in lower_text for symbol in math_symbols)


def name_quality_score(name: str) -> float:
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

    symbol_count = len(re.findall(r"[^A-Za-z0-9\s\-]", raw))
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
    }

    if lower in generic_noise:
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

    quality = name_quality_score(name)
    support = source_support_score(name, source_text, source_quote)

    try:
        llm_confidence = float(llm_confidence)
    except (TypeError, ValueError):
        llm_confidence = 0.5

    llm_confidence = max(0.0, min(1.0, llm_confidence))

    final_score = (
        0.45 * quality +
        0.40 * support +
        0.15 * llm_confidence
    )

    return final_score


def is_application_chunk(source_text: str) -> bool:
    """
    Detect chunks mainly describing applications/use cases of a concept.
    """

    lower_text = source_text.lower()

    return (
        "applications of" in lower_text[:160]
        or "used in" in lower_text[:200]
        or "used for" in lower_text[:200]
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
        # Keep real equation concepts such as Linear Equations or Quadratic Equations.
        if lower_name in {"linear equations", "quadratic equations"}:
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
    ]

    if any(term in lower_name for term in technical_terms):
        return False

    if len(name.split()) <= 3:
        return True

    return False


def should_keep_topic(topic: dict, source_text: str) -> bool:
    """
    Decide whether a topic is meaningful enough to store.
    """

    name = normalize_name(topic.get("name", ""))
    quote = topic.get("source_quote", "")
    confidence = topic.get("confidence", 0.5)

    score = combined_confidence_score(
        name=name,
        source_text=source_text,
        llm_confidence=confidence,
        source_quote=quote,
    )

    return score >= 0.68


def should_keep_concept(concept: dict, source_text: str) -> bool:
    """
    Decide whether a concept is meaningful enough to store.
    """

    name = normalize_name(concept.get("name", ""))
    quote = concept.get("source_quote", "")
    confidence = concept.get("confidence", 0.5)

    if not name:
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

    return score >= 0.72


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

            if prop_key not in nodes[key]["properties"]:
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
                float(concept.get("confidence", 0.5)),
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

    # If the whole chunk is about applications, create only the main concept.
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

    # If the whole chunk is about reasons/benefits, attach reasons to main concept.
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

    # Add promoted formula properties.
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

    # Remove duplicates by concept name.
    deduped = {}
    for concept in valid_concepts:
        key = concept["name"].lower()

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

    for topic in extraction.get("topics", []):
        if not isinstance(topic, dict):
            continue

        if not should_keep_topic(topic, source_text):
            continue

        topic_name = normalize_name(topic.get("name", ""))

        if is_application_chunk(source_text):
            topic_name = derive_main_concept_from_text(source_text, topic_name)

        valid_concepts = extract_valid_concepts_for_topic(topic, source_text)

        # Critical fix:
        # Do not create a Topic unless at least one valid Concept survived.
        if not valid_concepts:
            continue

        topic_score = combined_confidence_score(
            name=topic_name,
            source_text=source_text,
            llm_confidence=topic.get("confidence", 0.5),
            source_quote=topic.get("source_quote", ""),
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

        for concept in valid_concepts:
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

            add_relationship(relationships, topic_key, "HAS_CONCEPT", concept_key)

    # Add concept-to-concept relationships only when both concepts survived filtering.
    for rel in extraction.get("relationships", []):
        if not isinstance(rel, dict):
            continue

        rel_type = normalize_relationship_type(rel.get("type", ""))

        if not rel_type:
            continue

        source_name = normalize_name(rel.get("source", ""))
        target_name = normalize_name(rel.get("target", ""))

        source_key = concept_name_to_key.get(source_name.lower())
        target_key = concept_name_to_key.get(target_name.lower())

        if not source_key or not target_key:
            continue

        rel_score = combined_confidence_score(
            name=f"{source_name} {target_name}",
            source_text=source_text,
            llm_confidence=rel.get("confidence", 0.5),
            source_quote=rel.get("source_quote", ""),
        )

        if rel_score < 0.76:
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

    total_relationships_saved = 0
    total_nodes_saved = 0
    total_validated_chunks = 0
    total_failed_chunks = 0
    total_empty_chunks = 0
    total_write_failed_chunks = 0

    driver = GraphDatabase.driver(
        NEO4J_URI,
        auth=(NEO4J_USERNAME, NEO4J_PASSWORD),
        max_connection_lifetime=300,
        connection_timeout=60,
    )

    try:
        create_constraints(driver)

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
                print("Chunk skipped. No data saved to AuraDB.")
                continue

            print_graph_payload(graph_payload)

            conforms, validation_report = validate_graph_payload_with_shacl(graph_payload)

            if not conforms:
                total_failed_chunks += 1

                print("\n--- SHACL VALIDATION FAILED ---")
                print(validation_report)
                print("Chunk skipped. No data saved to AuraDB.")

                continue

            print("\n--- SHACL VALIDATION PASSED ---")

            write_success = write_graph_payload_to_auradb(driver, graph_payload)

            if not write_success:
                total_write_failed_chunks += 1
                print("Chunk validated but could not be saved to AuraDB.")
                continue

            total_validated_chunks += 1
            total_nodes_saved += len(graph_payload["nodes"])
            total_relationships_saved += len(graph_payload["relationships"])

            print(
                f"Saved {len(graph_payload['nodes'])} nodes and "
                f"{len(graph_payload['relationships'])} relationships to AuraDB."
            )

    finally:
        driver.close()

    print("\nKG extraction complete.")
    print(f"Validated and saved chunks: {total_validated_chunks}")
    print(f"Failed SHACL chunks: {total_failed_chunks}")
    print(f"Empty/noisy chunks skipped: {total_empty_chunks}")
    print(f"Write failed chunks: {total_write_failed_chunks}")
    print(f"Total nodes attempted: {total_nodes_saved}")
    print(f"Total relationships attempted: {total_relationships_saved}")