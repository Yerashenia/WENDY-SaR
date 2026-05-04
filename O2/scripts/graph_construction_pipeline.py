import os
import re
import logging
from pathlib import Path
from collections import Counter

from dotenv import load_dotenv
from neo4j import GraphDatabase

from llama_index.core import Document
from llama_index.core.node_parser import SentenceSplitter
from llama_index.llms.ollama import Ollama
from llama_index.readers.file import PDFReader


# -------------------------------------------------------------------
# Logging configuration
# -------------------------------------------------------------------

logging.basicConfig(level=logging.ERROR)

logging.getLogger("pypdf").setLevel(logging.ERROR)
logging.getLogger("pypdf._reader").setLevel(logging.ERROR)
logging.getLogger("httpx").setLevel(logging.ERROR)
logging.getLogger("httpcore").setLevel(logging.ERROR)

# Hide Neo4j informational messages such as "constraint already exists".
logging.getLogger("neo4j").setLevel(logging.WARNING)


# -------------------------------------------------------------------
# Environment configuration
# -------------------------------------------------------------------

load_dotenv()

DATA_SOURCE = os.getenv("pdf_folder")
NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
NEO4J_DATABASE = os.getenv("NEO4J_DATABASE")


# -------------------------------------------------------------------
# PDF loading and preprocessing
# -------------------------------------------------------------------

def load_pdf_documents(pdf_folder: str):
    """
    Load all PDF files from the configured folder.

    Each page is loaded as a separate document and receives the source
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


def extract_pdf_metadata(pdf_path: str, llm):
    """
    Extract the module name and week number from the first page of a PDF.

    The extracted values are treated as trusted metadata. This prevents
    the LLM from assigning incorrect module or week values during graph
    extraction.
    """

    import pypdf

    reader = pypdf.PdfReader(pdf_path)
    first_page_text = reader.pages[0].extract_text()

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

    text = text.replace("\n", " ").replace("\t", " ")
    text = " ".join(text.split())
    text = re.sub(r"([a-z])([A-Z])", r"\1 \2", text)

    return text


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


def prepare_documents(documents, llm):
    """
    Clean loaded PDF pages and attach trusted module/week metadata.

    Metadata is extracted once per PDF and reused for all pages belonging
    to that PDF.
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

        cleaned_documents.append(
            Document(
                text=cleaned,
                metadata=doc.metadata,
            )
        )

    return cleaned_documents


def chunk_documents(documents, chunk_size=300, chunk_overlap=50):
    """
    Split cleaned documents into smaller text chunks for LLM extraction.
    """

    splitter = SentenceSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    return splitter.get_nodes_from_documents(documents)


# -------------------------------------------------------------------
# LLM setup and KG extraction
# -------------------------------------------------------------------

def create_llm():
    """
    Create the local Ollama LLM used for metadata and KG extraction.
    """

    return Ollama(
        model="llama3",
        temperature=0.0,
        request_timeout=300.0,
    )


def extract_kg_text_manually(text: str, llm):
    """
    Extract raw knowledge graph lines from one text chunk.

    The output format is intentionally simple so it can be parsed into
    Neo4j nodes and relationships.
    """

    prompt = f"""
You are extracting a knowledge graph from academic lecture text.

Return ONLY graph lines.
Do not explain.
Do not use markdown.
Do not include notes.
Do not write "Here are the extracted graph lines".
Do not copy any examples from this prompt into the output.
Only use names and facts that are present in the given TEXT.

TARGET SCHEMA:

Module
HAS_WEEK Week

Week
COVERS Topic

Topic
HAS_CONCEPT Concept

Concept
DEFINED_BY Definition
HAS_EXAMPLE Example
HAS_FORMULA Formula
USED_FOR Application
RELATED_TO Concept
BUILDS_ON Concept
PREREQUISITE_OF Concept

VALID NODE TYPES:
- Module
- Week
- Topic
- Concept
- Definition
- Example
- Formula
- Application

VALID RELATIONSHIPS:
- HAS_WEEK
- COVERS
- HAS_CONCEPT
- DEFINED_BY
- HAS_EXAMPLE
- HAS_FORMULA
- USED_FOR
- RELATED_TO
- BUILDS_ON
- PREREQUISITE_OF

OUTPUT FORMAT PATTERN:

Topic:<topic name from the text>
HAS_CONCEPT Concept:<concept name from the text>

Concept:<concept name from the text>
DEFINED_BY Definition:<definition sentence from the text>
HAS_EXAMPLE Example:<example from the text>
HAS_FORMULA Formula:<formula from the text>
USED_FOR Application:<application from the text>

IMPORTANT EXTRACTION RULES:
1. Extract only academic content.
2. Ignore staff names, emails, office hours, contact details, greetings, Blackboard links, and announcements.
3. Do not output placeholder names such as Topic, Concept, Module, Week, Definition, Example, Formula, or Application.
4. Do not invent missing information.
5. Prefer clear topic and concept names over long sentence-like concepts.
6. Use Definition only when the text clearly explains what something is.
7. Use Example only when the text gives a concrete example.
8. Use Formula only when the text contains a mathematical formula, equation, or notation.
9. Use Application only when the text clearly says where something is used.
10. If the text contains a heading followed by a colon and a list, the heading is usually a Topic and listed items are usually Concepts.
11. Do not turn headings such as "Arithmetic Skills" or "Set Theory" into Module nodes.
12. Do not mention Set Theory, Set, Empty Set, or the definition of a set unless the input text explicitly discusses sets.
13. Do not use generic fallback content from previous chunks or from this prompt.
14. Do not create Concept:Set unless the text explicitly contains the word "set" as an academic concept.
15. Do not create fake examples such as "unknown", "none", or "no example provided".
16. If a concept has a definition, formula, example, or application, first output a Topic line and then connect the concept using HAS_CONCEPT whenever possible.
17. Do not create concepts named Definition, Example, Task, Question, Answer, Property, Attribute, Element, Number, A, B, C, D, S, or T unless they are clearly academic concepts.
18. Single letters such as A, B, C, D, S, and T should be examples, not Concept nodes.
19. Expressions such as A∪B, A∩B, A×B, B∩C, A∖B, A⊂B, A⊆B, B/C, or A' should usually be Formula or Example nodes, not Concept nodes.
20. If the text is mainly an activity or worked answer, extract only the core academic concept and examples. Do not create separate concepts for every variable.
21. If no valid extraction exists, return nothing.

TEXT:
{text}
"""

    response = llm.complete(prompt)
    return response.text.strip()


# -------------------------------------------------------------------
# KG parsing and filtering
# -------------------------------------------------------------------

def normalize_week_name(week_number):
    """
    Convert week metadata into the standard Week node name.
    """

    week_text = str(week_number).strip()

    if not week_text.lower().startswith("week"):
        week_text = f"Week {week_text}"

    return week_text


def parse_labeled_object(text: str):
    """
    Parse an object written as Label:Value.

    Example:
    Concept:Empty Set -> ("Concept", "Empty Set")
    Definition:A set has no repeated elements. -> ("Definition", "A set has no repeated elements.")
    """

    if ":" not in text:
        return None, None

    label, value = text.split(":", 1)
    label = label.strip()
    value = value.strip()

    if not label or not value:
        return None, None

    return label, value


def is_noisy_topic(name: str) -> bool:
    """
    Identify topic names that are too generic or likely produced by noise.
    """

    if not name:
        return True

    normalized = name.strip().lower()

    blocked_exact = {
        "definition",
        "example",
        "examples",
        "task",
        "question",
        "answer",
        "answers",
        "activity",
        "property",
        "attribute",
        "note",
        "notes",
        "table",
        "result",
        "results",
    }

    return normalized in blocked_exact


def is_noisy_concept(name: str) -> bool:
    """
    Identify concept names that are usually too generic, example-specific,
    symbolic, or unsuitable as academic concept nodes.
    """

    if not name:
        return True

    normalized = name.strip().lower()

    blocked_exact = {
        "definition",
        "example",
        "examples",
        "task",
        "question",
        "answer",
        "answers",
        "property",
        "attribute",
        "note",
        "notes",
        "table",
        "result",
        "results",
        "activity",
        "element",
        "elements",
        "number",
        "computer",
        "s",
        "t",
        "a",
        "b",
        "c",
        "d",
        "a'",
        "a′",
        "ac",
        "b/c",
    }

    if normalized in blocked_exact:
        return True

    # Avoid numeric-only concepts, for example Concept:17.
    if normalized.isdigit():
        return True

    # Avoid storing example set names as concepts, such as Set A or Set B.
    if normalized.startswith("set ") and len(normalized.split()) == 2:
        return True

    # Avoid storing raw set-operation expressions as concepts.
    symbolic_markers = ["∪", "∩", "×", "∖", "⊂", "⊆", "/"]
    if any(marker in name for marker in symbolic_markers):
        return True

    # Avoid table-header style concepts.
    if "definition description" in normalized:
        return True

    return False


def is_noisy_detail(label: str, value: str) -> bool:
    """
    Filter weak detail nodes such as unknown examples, empty values,
    meaningless examples, and very short definitions.
    """

    if not value:
        return True

    normalized = value.strip().lower()

    blocked_values = {
        "unknown",
        "unknown (no example provided in the text)",
        "no example provided",
        "none",
        "n/a",
        "null",
        "e.g.",
        "example",
        "example:",
        "used in",
        "application",
    }

    if normalized in blocked_values:
        return True

    # Avoid fake examples.
    if label == "Example" and "no example" in normalized:
        return True

    # Avoid examples that are too vague to help QA.
    if label == "Example" and len(normalized) <= 3:
        return True

    # Avoid definitions that are too short to be meaningful.
    if label == "Definition" and len(normalized) <= 2:
        return True

    # Avoid empty formulas.
    if label == "Formula" and len(normalized) <= 1:
        return True

    # Avoid application nodes that are too vague.
    if label == "Application" and len(normalized) <= 3:
        return True

    return False

def parse_kg_output(raw_text: str, module_name=None, week_number=None):
    """
    Convert raw LLM graph lines into clean triples.

    The parser does not trust module/week values generated by the LLM.
    Instead, it uses trusted module/week metadata extracted from the PDF.
    The LLM output is mainly used for Topic, Concept, Definition, Example,
    Formula, and Application extraction.
    """

    triples = []
    lines = [line.strip() for line in raw_text.splitlines() if line.strip()]

    current_topic = None
    current_concept = None

    module_node = None
    week_node = None

    valid_node_labels = {
        "Module",
        "Week",
        "Topic",
        "Concept",
        "Definition",
        "Example",
        "Formula",
        "Application",
    }

    concept_detail_relations = {
        "DEFINED_BY": "Definition",
        "HAS_EXAMPLE": "Example",
        "HAS_FORMULA": "Formula",
        "USED_FOR": "Application",
    }

    concept_to_concept_relations = {
        "RELATED_TO",
        "BUILDS_ON",
        "PREREQUISITE_OF",
    }

    if module_name:
        module_node = ("Module", module_name.strip())

    if week_number:
        week_node = ("Week", normalize_week_name(week_number))

    if module_node and week_node:
        triples.append((module_node, "HAS_WEEK", week_node))

    for line in lines:
        # Ignore LLM-generated module/week hierarchy.
        if line.startswith("Module:"):
            continue

        if line.startswith("HAS_WEEK"):
            continue

        if line.startswith("Week:"):
            continue

        # Example: Topic:Set Theory
        if line.startswith("Topic:"):
            _, topic_name = parse_labeled_object(line)

            if topic_name and not is_noisy_topic(topic_name):
                current_topic = ("Topic", topic_name)
                current_concept = None

                if week_node:
                    triples.append((week_node, "COVERS", current_topic))

            continue

        # Example: Concept:Set
        if line.startswith("Concept:"):
            _, concept_name = parse_labeled_object(line)

            if concept_name and not is_noisy_concept(concept_name):
                current_concept = ("Concept", concept_name)

                if current_topic:
                    triples.append((current_topic, "HAS_CONCEPT", current_concept))

            continue

        # Split relationship lines, for example:
        # HAS_CONCEPT Concept:Empty Set
        # DEFINED_BY Definition:A set with no elements.
        parts = line.split(" ", 1)

        if len(parts) != 2:
            continue

        rel = parts[0].strip()
        obj_text = parts[1].strip()

        obj_label, obj_value = parse_labeled_object(obj_text)

        if obj_label not in valid_node_labels or not obj_value:
            continue

        obj_node = (obj_label, obj_value)

        # Example: COVERS Topic:Set Theory
        if rel == "COVERS" and obj_label == "Topic":
            if is_noisy_topic(obj_value):
                continue

            current_topic = obj_node
            current_concept = None

            if week_node:
                triples.append((week_node, "COVERS", current_topic))

            continue

        # Example: HAS_CONCEPT Concept:Empty Set
        if rel == "HAS_CONCEPT" and obj_label == "Concept":
            if is_noisy_concept(obj_value):
                continue

            current_concept = obj_node

            if current_topic:
                triples.append((current_topic, "HAS_CONCEPT", current_concept))

            continue

        # Example: DEFINED_BY Definition:A set is a collection of distinct elements.
        if rel in concept_detail_relations:
            expected_label = concept_detail_relations[rel]

            if obj_label == expected_label and current_concept:
                if is_noisy_detail(obj_label, obj_value):
                    continue

                triples.append((current_concept, rel, obj_node))

            continue

        # Example: RELATED_TO Concept:Integer
        if rel in concept_to_concept_relations:
            if obj_label == "Concept" and current_concept:
                if is_noisy_concept(obj_value):
                    continue

                triples.append((current_concept, rel, obj_node))

            continue

    return remove_duplicate_triples(triples)


def remove_duplicate_triples(triples):
    """
    Remove duplicate triples while preserving order.
    """

    unique_triples = []
    seen = set()

    for triple in triples:
        if triple not in seen:
            unique_triples.append(triple)
            seen.add(triple)

    return unique_triples


# -------------------------------------------------------------------
# Neo4j / AuraDB writing
# -------------------------------------------------------------------

def property_key_for_label(label: str):
    """
    Return the correct property key for a node label.

    Definition, Example, and Formula use `text`.
    Other labels use `name`.
    """

    if label in {"Definition", "Example", "Formula"}:
        return "text"

    return "name"


def triples_to_cypher(triples):
    """
    Convert parsed triples into Cypher MERGE statements.

    MERGE is used so rerunning the script reuses exact-match nodes and
    relationships instead of creating duplicates.
    """

    cypher_statements = []

    for (sub_label, sub_value), rel, (obj_label, obj_value) in triples:
        sub_key = property_key_for_label(sub_label)
        obj_key = property_key_for_label(obj_label)

        cypher = f"""
MERGE (a:{sub_label} {{{sub_key}: $sub_value}})
MERGE (b:{obj_label} {{{obj_key}: $obj_value}})
MERGE (a)-[:{rel}]->(b)
"""

        params = {
            "sub_value": sub_value,
            "obj_value": obj_value,
        }

        cypher_statements.append((cypher, params))

    return cypher_statements


def write_to_auradb(session, cypher_statements):
    """
    Write Cypher statements to AuraDB using an existing Neo4j session.
    """

    for cypher, params in cypher_statements:
        session.run(cypher, **params)


def create_constraints(session):
    """
    Create uniqueness constraints for the KG node labels.
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
        CREATE CONSTRAINT definition_text_unique IF NOT EXISTS
        FOR (d:Definition)
        REQUIRE d.text IS UNIQUE
        """,
        """
        CREATE CONSTRAINT example_text_unique IF NOT EXISTS
        FOR (e:Example)
        REQUIRE e.text IS UNIQUE
        """,
        """
        CREATE CONSTRAINT formula_text_unique IF NOT EXISTS
        FOR (f:Formula)
        REQUIRE f.text IS UNIQUE
        """,
        """
        CREATE CONSTRAINT application_name_unique IF NOT EXISTS
        FOR (a:Application)
        REQUIRE a.name IS UNIQUE
        """,
    ]

    for constraint in constraints:
        session.run(constraint)


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
        print(f"Week {week}: {count} nodes")


def debug_node_output(nodes, llm):
    """
    Print the raw LLM extraction output for a selected node.
    """

    if not nodes:
        print("No nodes available.")
        return

    user_index = int(input("Which node would you like to debug?\n"))
    test_node = nodes[user_index]

    raw_output = extract_kg_text_manually(test_node.text, llm)

    print("\n--- NODE TEXT ---")
    print(test_node.text)

    print("\n--- RAW MODEL OUTPUT ---")
    print(raw_output)


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

    total_triples = 0

    driver = GraphDatabase.driver(
        NEO4J_URI,
        auth=(NEO4J_USERNAME, NEO4J_PASSWORD),
    )

    try:
        with driver.session(database=NEO4J_DATABASE) as session:
            create_constraints(session)

            for index, node in enumerate(nodes_to_process, start=1):
                print(f"\n--- PROCESSING CHUNK {index}/{len(nodes_to_process)} ---")
                print(node.text[:300], "...")

                raw_output = extract_kg_text_manually(node.text, llm)

                print("\n--- RAW MODEL OUTPUT ---")
                print(raw_output)

                module_name = node.metadata.get("module")
                week_number = node.metadata.get("week")

                triples = parse_kg_output(
                    raw_output,
                    module_name=module_name,
                    week_number=week_number,
                )

                print("\n--- PARSED TRIPLES ---")
                for triple in triples:
                    print(triple)

                cypher_statements = triples_to_cypher(triples)

                if cypher_statements:
                    write_to_auradb(session, cypher_statements)
                    total_triples += len(cypher_statements)
                    print(f"Saved {len(cypher_statements)} relationships to AuraDB.")
                else:
                    print("No valid triples found for this chunk.")

    finally:
        driver.close()

    print("\nKG extraction complete.")
    print(f"Total relationships attempted: {total_triples}")