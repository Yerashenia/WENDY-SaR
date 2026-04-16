import os
from pathlib import Path
from typing import Literal
import re
import logging
from dotenv import load_dotenv
import os


# Reduce noisy library logs
logging.basicConfig(level=logging.ERROR)
logging.getLogger("pypdf").setLevel(logging.ERROR)
logging.getLogger("pypdf._reader").setLevel(logging.ERROR)
logging.getLogger("httpx").setLevel(logging.ERROR)
logging.getLogger("httpcore").setLevel(logging.ERROR)

from llama_index.core import SimpleDirectoryReader, Document, PromptTemplate
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.indices.property_graph import SchemaLLMPathExtractor
from llama_index.llms.ollama import Ollama


#importing data
load_dotenv()
DATA_SOURCE = os.getenv("pdf_folder")

def load_pdf_documents(pdf_folder: str):
    """Load PDF files from a folder."""
    reader = SimpleDirectoryReader(
        input_dir=pdf_folder,
        required_exts=[".pdf"]
    )
    return reader.load_data()


def clean_text(text: str):
    """Lightly clean extracted PDF text."""
    text = text.replace("\n", " ").replace("\t", " ")
    text = " ".join(text.split())
    text = re.sub(r"([a-z])([A-Z])", r"\1 \2", text)
    return text


def is_academic_content(text: str) -> bool:
    """Filter out Non-academic pages"""
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
        "module announcements"
    ]

    return not any(keyword in lower_text for keyword in unwanted_keywords)


def prepare_documents(documents):
    """Clean documents and keep only academic ones."""
    cleaned_documents = []

    for doc in documents:
        cleaned = clean_text(doc.text)

        if not is_academic_content(cleaned):
            continue

        cleaned_documents.append(
            Document(
                text=cleaned,
                metadata=doc.metadata
            )
        )

    return cleaned_documents


def chunk_documents(documents, chunk_size=300, chunk_overlap=50):
    """Split documents into smaller chunks."""
    splitter = SentenceSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    return splitter.get_nodes_from_documents(documents)


def create_llm():
    """Create Ollama LLM."""
    return Ollama(
        model="llama3",
        temperature=0.0,
        request_timeout=12.0
    )


def create_kg_extractor(llm):
    """Create schema-based KG extractor."""
    possible_entities = Literal[
        "Module",
        "Week",
        "Topic",
        "Concept"
    ]

    possible_relations = Literal[
        "HAS_WEEK",
        "NEXT",
        "COVERS",
        "HAS_CONCEPT",
        "PREREQUISITE_OF",
        "RELATED_TO",
        "BUILDS_ON",
        "USES",
        "USED_FOR",
        "DEFINED_BY",
        "EXAMPLE_OF",
        "SPECIAL_CASE_OF",
        "RESULTS_IN"
    ]

    kg_validation_schema = {
        "Module": ["HAS_WEEK"],
        "Week": ["NEXT", "COVERS"],
        "Topic": ["HAS_CONCEPT", "PREREQUISITE_OF", "RELATED_TO"],
        "Concept": [
            "RELATED_TO",
            "BUILDS_ON",
            "PREREQUISITE_OF",
            "USES",
            "USED_FOR",
            "DEFINED_BY",
            "EXAMPLE_OF",
            "SPECIAL_CASE_OF",
            "RESULTS_IN"
        ]
    }

    extract_prompt = PromptTemplate(
        """
You are extracting a knowledge graph from academic lecture text.

Return only valid graph content.
Do not include explanations.
Do not include markdown code fences.
Do not add text before or after the result.

NODE TYPES:
- Module
- Week
- Topic
- Concept

VALID RELATIONSHIPS:
- Module -> HAS_WEEK -> Week
- Week -> NEXT -> Week
- Week -> COVERS -> Topic
- Topic -> HAS_CONCEPT -> Concept
- Topic -> PREREQUISITE_OF -> Topic
- Topic -> RELATED_TO -> Topic
- Concept -> RELATED_TO -> Concept
- Concept -> BUILDS_ON -> Concept
- Concept -> PREREQUISITE_OF -> Concept
- Concept -> USES -> Concept
- Concept -> USED_FOR -> Concept
- Concept -> DEFINED_BY -> Concept
- Concept -> EXAMPLE_OF -> Concept
- Concept -> SPECIAL_CASE_OF -> Concept
- Concept -> RESULTS_IN -> Concept

RULES:
RULES:
1. Extract only meaningful academic content.
2. Ignore staff names, emails, office hours, contact details, greetings, welcome text, Blackboard links, and announcements.
3. Convert "Lecture 1" to "Week 1", "Lecture 2" to "Week 2", and so on.
4. Topics listed under a lecture belong to the Week using COVERS.
5. Do not connect Module directly to Topic using COVERS.
6. Do not invent placeholder names such as Topic 1, Topic 2, Concept A, or Concept B.
7. Use exact academic names from the text whenever possible.
8. If unsure, skip it.
9. Maximum {max_triplets_per_chunk} relationships.
10. Only extract Module if it is explicitly written in the text.
11. Only extract Week if the text explicitly mentions a lecture number or week number.
12. If Module or Week is missing, do not invent them.
13. If the chunk is a partial fragment without enough context, extract only clearly supported Topic and Concept relationships.
14. Never guess a module name such as Calculus or any other subject not present in the text.
15. Only create a Week node if the text explicitly mentions a lecture number or week number.
16. Do not assign topic groups to Week 1, Week 2, etc. unless the text explicitly says so.
17. Only create a Module node if the text explicitly identifies a module.
18. If Module or Week is missing, do not invent them.
19. When the text is a topic overview, extract only Topic and Concept relationships that are directly supported.
20. If the text is a revision list, activity, or topic summary without explicit module or week information, do not create Module or Week nodes.
21. In revision lists or topic summaries, treat headings as Topic nodes and listed sub-items as Concept nodes when clearly supported.
22. Do not output placeholder structure lines such as "Module HAS_WEEK Week", "Week COVERS Topic", or "Topic HAS_CONCEPT Concept".
23. Always include actual names when extracting, for example:
    - Topic: Algebra Foundations
    - Concept: Decimals
24. Do not use DEFINED_BY unless the text explicitly defines one concept using another.
25. Do not infer hierarchy from prior knowledge or assumptions. Use only what is explicitly supported by the given text.


TEXT:
{text}
"""
    )

    return SchemaLLMPathExtractor(
        llm=llm,
        possible_entities=possible_entities,
        possible_relations=possible_relations,
        kg_validation_schema=kg_validation_schema,
        extract_prompt=extract_prompt,
        strict=False,
        num_workers=1,
        max_triplets_per_chunk=3
    )


def debug_node_output(nodes, llm):
    """Show raw LLM output for the node."""
    if not nodes:
        print("No nodes available.")
        return

    user_index=int(input("Which node would you like to debug?\n"))
    test_node = nodes[user_index]

    prompt = f"""
    Extract a knowledge graph from the text below.

    Use only these entity types:
    - Module
    - Week
    - Topic
    - Concept

    Use only these relationship types:
    - HAS_WEEK
    - NEXT
    - COVERS
    - HAS_CONCEPT
    - PREREQUISITE_OF
    - RELATED_TO
    - BUILDS_ON
    - USES
    - USED_FOR
    - DEFINED_BY
    - EXAMPLE_OF
    - SPECIAL_CASE_OF
    - RESULTS_IN

    Rules:
    - Follow this hierarchy when possible:
      Module -> HAS_WEEK -> Week
      Week -> COVERS -> Topic
      Topic -> HAS_CONCEPT -> Concept
    - Convert "Lecture 1" to "Week 1", "Lecture 2" to "Week 2", etc.
    - If the text contains a lecture title followed by several listed items separated by periods, treat each listed item as a separate Topic covered by that Week.
    - Do not merge listed topics together.
    - Do not turn one listed topic into a concept of another unless the text clearly says so.
    - Do not invent information.
    - Do not say "Here is the extracted graph content" or anything similar.
    - Ignore administrative text.
    - Return only extracted graph content.
    - Return only the extracted graph lines.
    - Do not add explanations.
    - Do not add notes.
    - Do not add closing comments.
    - Do not say "Let me know if you have any further questions."
    - If no valid extraction exists, return nothing.
    - Only extract Module if it is explicitly written in the text.
    - Only extract Week if the text explicitly mentions a lecture number or week number.
    - If Module or Week is missing, do not invent them.
    - If the chunk is a partial fragment without enough context, extract only clearly supported Topic and Concept relationships.
    - Never guess a module name such as Calculus or any other subject not present in the text.
    - Only create a Week node if the text explicitly mentions a lecture number or week number.
    - Do not assign topic groups to Week 1, Week 2, etc. unless the text explicitly says so.
    - Only create a Module node if the text explicitly identifies a module.
    - If Module or Week is missing, do not invent them.
    - When the text is a topic overview, extract only Topic and Concept relationships that are directly supported.
    - If the text is a revision list, activity, or topic summary without explicit lecture/week mention:
    → Do NOT create Module or Week nodes.
    → Extract only Topic -> HAS_CONCEPT -> Concept relationships.

    - Do NOT output placeholder words like:
    "Topic", "Concept", "Module", "Week" without names.

    - Always include actual names:
        Topic:Algebra Foundations
        Concept:Decimals

    - Do NOT use DEFINED_BY unless explicitly stated in text.

    - If a topic heading is followed by a list, treat:
        Heading = Topic
        List items = Concepts 
    
    Text:
    {test_node.text}
    """

    response = llm.complete(prompt)

    print("\n--- FIRST NODE TEXT ---")
    print(test_node.text)

    print("\n--- RAW MODEL OUTPUT ---")
    print(response)
    print()




if __name__ == "__main__":
    documents = load_pdf_documents(DATA_SOURCE)
    cleaned_documents = prepare_documents(documents)
    nodes = chunk_documents(cleaned_documents)

    print(f"Cleaned documents: {len(cleaned_documents)}")
    print(f"Nodes created: {len(nodes)}")

    llm = create_llm()
    extractor = create_kg_extractor(llm)

    # For debugging purpose
    debug_node_output(nodes, llm)
