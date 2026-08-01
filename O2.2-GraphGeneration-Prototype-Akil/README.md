# 🚀 WENDY-SaR – Neo4j Knowledge Graph Question Answering Script

This script takes a user’s question in natural language and uses **Ollama (Llama3)** to generate a **Cypher query**.

It then connects to a **Neo4j Aura database** or a **local Neo4j database**, runs the generated Cypher query on the knowledge graph, retrieves the relevant data, and uses the LLM again to turn the database result into a short, clear answer for the user.

In short, the script follows this flow:

**User Question → Llama3 generates Cypher → Neo4j executes query → Results returned → Llama3 generates final answer**

---

## 📌 What This Script Does

- Accepts a user question from the terminal
- Uses **Ollama (Llama3)** to convert the question into a Cypher query
- Connects to **Neo4j Aura** or a **local Neo4j database**
- Runs the Cypher query on the knowledge graph
- Retrieves matching data from the database
- Uses the LLM again to generate a natural-language answer
- Displays the generated Cypher query, database result, and final answer

---

## 📌 Requirements

Make sure the following are installed before running the script:

- **Python 3.8 or higher**
- **Ollama**
- **Llama3 model in Ollama**
- **Neo4j Aura DB** or **local Neo4j database**
- Required Python libraries listed in `requirements.txt`

---

## 📌 Python Packages Used

This script uses the following main libraries:

- `python-dotenv`
- `langchain-neo4j`
- `langchain-ollama`
- `langchain-core`

---

## 📌 Environment Variables📌 Setup

Rename .env.example to `.env` file and add your Neo4j credentials:

```env
NEO4J_URI=your_neo4j_uri
NEO4J_USERNAME=your_username
NEO4J_PASSWORD=your_password
NEO4J_DATABASE=your_database_name

##📌 Setup & Run

Once Neo4j database and Ollama are running, follow these steps:

1️⃣ Navigate to the project folder

cd WENDY-SaR/O2

2️⃣ Run setup script (installs required environment and dependencies)

python setup.py

3️⃣ Run the script

python run.py

After running, a prompt will appear:

Ask your question (type 'exit' to quit):