from dotenv import load_dotenv
load_dotenv()
from langchain_neo4j import Neo4jGraph
from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate
from langchain_neo4j import GraphCypherQAChain
import os

#Neo4j credentials
NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
NEO4J_DATABASE = os.getenv("NEO4J_DATABASE")

# Neo4j connection
graph = Neo4jGraph(url=NEO4J_URI,
                   username=NEO4J_USERNAME,
                   password=NEO4J_PASSWORD,
                   database=NEO4J_DATABASE,)


# LLM (Ollama - Llama3)
llm = OllamaLLM(model="llama3")

# Prompt template
cypher_prompt = PromptTemplate.from_template("""
You are an expert Neo4j developer.

Your task is to generate a valid Cypher query using ONLY the provided schema.

STRICT RULES:
- Use ONLY the given node labels, relationships, and properties.
- Do NOT invent labels, relationships, or properties.
- Return ONLY the Cypher query (no explanation, no markdown, no backticks).
- Use correct direction of relationships as defined.
- Use exact property names (`name` only).
- Always match string values EXACTLY (case-sensitive).

COUNTING RULES:
- If the question asks how many weeks a module has, use count(w), not invented variables.
- If no specific module name is provided, return the module name along with the count.
- Never invent placeholders like {{module_name}} or {{name}}.
- NEVER use curly-brace placeholders like {{}} inside Cypher queries.
- Only use Cypher parameters like $module_name if explicitly provided externally.

IMPORTANT DATA RULES:
- Week nodes use property `name` with exact values like "Week 1", "Week 2", etc.
- NEVER use "1", "week1", or "week 1" unless explicitly present.
- Always preserve exact spacing and capitalization.
- Topic, Module, and Concept nodes also use `name`.

QUERY RULES:
- Prefer simple and direct patterns (avoid unnecessary nodes).
- Use directed relationships exactly as defined (e.g., ->).
- Only use relationships that exist in the schema.
- If filtering by name, use exact match: {{name: "Week 1"}}
- Do NOT assume data exists beyond the schema.

OUTPUT FORMAT RULES:
- Output exactly one valid Cypher query.
- Do not add any introductory text.
- Do not say "Here is the Cypher query".
- Do not use markdown.
- The first word of the output must be a Cypher keyword such as MATCH, RETURN, CALL, or WITH.

Schema:

Node labels:
- Module(name)
- Week(name)
- Topic(name)
- Concept(name)

Relationships:
- (Module)-[:HAS_WEEK]->(Week)
- (Week)-[:NEXT]->(Week)
- (Week)-[:COVERS]->(Topic)
- (Topic)-[:HAS_CONCEPT]->(Concept)
- (Topic)-[:PREREQUISITE_OF]->(Topic)
- (Topic)-[:RELATED_TO]->(Topic)
- (Concept)-[:RELATED_TO]->(Concept)
- (Concept)-[:BUILDS_ON]->(Concept)
- (Concept)-[:PREREQUISITE_OF]->(Concept)
- (Concept)-[:USES]->(Concept)
- (Concept)-[:USED_FOR]->(Concept)
- (Concept)-[:DEFINED_BY]->(Concept)
- (Concept)-[:EXAMPLE_OF]->(Concept)
- (Concept)-[:SPECIAL_CASE_OF]->(Concept)
- (Concept)-[:RESULTS_IN]->(Concept)

Examples:

Question: What topics are covered in Week 1?
Cypher:
MATCH (w:Week {{name: "Week 1"}})-[:COVERS]->(t:Topic)
RETURN t.name

Question: What concepts are in a topic?
Cypher:
MATCH (t:Topic)-[:HAS_CONCEPT]->(c:Concept)
RETURN c.name

Question: How many weeks are in this module?
Cypher:
MATCH (m:Module)-[:HAS_WEEK]->(w:Week)
RETURN m.name, count(w) AS week_count

Question:
{question}
""")


answer_prompt = PromptTemplate.from_template("""
You are a helpful assistant answering questions based on Neo4j query results.

Question:
{question}

Cypher query:
{cypher}

Database result:
{context}

Give a short and clear answer.
If the result is empty, say you could not find matching data.
""")


# Function: sanitize Cypher
# Removes unwanted text before sending query to Neo4j
def sanitize_cypher(query: str) -> str:
    query = query.strip()
    query = query.replace("```cypher", "").replace("```", "").strip()
    prefixes = [
        "Here is the Cypher query:",
        "Cypher query:",
        "Here is the query:",
        "Query:"
    ]

    for prefix in prefixes:
        if query.startswith(prefix):
            query = query[len(prefix):].strip()
    return query

# Function: generate Cypher from the question
def generate_cypher(question: str) -> str:
    prompt_text = cypher_prompt.format(question=question)
    raw_cypher = llm.invoke(prompt_text)
    clean_cypher=sanitize_cypher(raw_cypher)
    return clean_cypher



# Function: generate final natural-language answer
def generate_answer(question: str, cypher: str, context) -> str:
    prompt_text = answer_prompt.format(
        question=question,
        cypher=cypher,
        context=context
    )
    answer = llm.invoke(prompt_text)
    return answer




'''
# This prebuilt GraphCypherQAChain class from LangChain was initially used but it produced inconsistent and sometimes invalid Cypher queries, causing execution errors. Therefore, a custom pipeline was implemented to control query generation, sanitisation, and execution. The original chain is kept as a comment in the code in case someone wants to try it.


chain = GraphCypherQAChain.from_llm(
    llm=llm,
    graph=graph,
    cypher_prompt=cypher_prompt,
    verbose=True,
    return_intermediate_steps=True,
    allow_dangerous_requests=True
)
'''

# Chat loop
while True:
    question = input("Ask your question (type 'exit' to quit): ")

    if question.lower() == "exit":
        break

    try:
        generated_cypher = generate_cypher(question)
        context = graph.query(generated_cypher)
        answer = generate_answer(question, generated_cypher, context)


        print("\n========================")
        print("Connected database:", NEO4J_DATABASE)
        print("Generated Cypher:")
        print(generated_cypher)
        print("\nFull Context:")
        print(context)
        print("\nAnswer:")
        print(answer)
        print("========================\n")

    except Exception as e:
        print("\n========================")
        print("Connected database:", NEO4J_DATABASE)
        print("An error occurred:")
        print(e)
        print("========================\n")