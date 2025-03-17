# Egyptian Smart Legal Advisor

The **Egyptian Smart Legal Advisor** is an AI-powered system designed to assist users in navigating Egyptian legal documents, providing semantic search capabilities, and generating legal advice based on Egyptian law sources. This project combines advanced Natural Language Processing (NLP), machine learning, and graph database technologies to deliver accurate and context-aware legal insights tailored for Arabic legal text.

----

## Table of Contents

1. [Overview](#overview)
2. [Key Features](#key-features)
3. [System Architecture](#system-architecture)
4. [Workflow](#workflow)
5. [Technical Stack](#technical-stack)
6. [Getting Started](#getting-started)
   - [Prerequisites](#prerequisites)
   - [Installation](#installation)
   - [Running the System](#running-the-system)
7. [Files and Components](#files-and-components)
8. [Example Use Cases](#example-use-cases)
9. [Future Enhancements](#future-enhancements)
10. [License](#license)
11. [Contact](#contact)
12. [Acknowledgments](#acknowledgments)

---

## Overview

The Egyptian Smart Legal Advisor is a comprehensive tool for legal professionals, researchers, and the general public to interact with Egyptian legal documents. It provides:

- **Semantic Search**: Find relevant legal articles using natural language queries.
- **Legal Advice**: Generate context-aware legal advice based on Egyptian law.
- **Document Analysis**: Extract and analyze key legal terms and relationships.

The system is designed to handle Arabic legal text, addressing the unique challenges of the Arabic language and legal domain.

---

## Key Features

1. **Arabic Text Processing**:
   - Normalization of Arabic text (e.g., removing diacritics, standardizing characters).
   - Extraction of key legal terms using predefined patterns and stemming.

2. **Document Processing**:
   - Extraction of text from PDF documents (e.g., `الدستور المصري المعدل 2019.pdf`, `المدني.pdf`, `قانون_الاجراءات_الجنائية.pdf`).
   - Splitting of legal text into structured chunks for processing.

3. **Semantic Search**:
   - Use of FAISS (Facebook AI Similarity Search) for efficient vector-based semantic search.
   - Arabic-optimized embeddings for accurate retrieval of relevant legal documents.

4. **Graph-Based Search**:
   - Integration with Neo4j for structured legal knowledge representation.
   - Retrieval of articles based on legal terms and their relationships in the knowledge graph.

5. **Hybrid Retrieval**:
   - Combination of semantic search (FAISS) and graph-based search (Neo4j) for comprehensive results.
   - Contextual expansion to include related articles for better coverage.

6. **Query Classification**:
   - Classification of user queries into legal or general categories using zero-shot classification.
   - Support for Arabic queries with a multilingual model (`xlm-roberta-large-xnli`).

7. **Legal Advice Generation**:
   - Integration with large language models (LLMs) like Google's Gemini for generating legal advice.
   - Structured prompts to ensure accurate and context-aware responses.

8. **User Interaction**:
   - Support for Arabic conversational queries.
   - Clear and formatted output for easy consumption by users or LLMs.

---

## System Architecture

The system is built using a modular architecture, with the following key components:

1. **Arabic Text Processor**:
   - Handles text normalization, stemming, and legal term extraction.
   - Ensures consistency in Arabic text processing.

2. **PDF Processor**:
   - Extracts text from PDF documents.
   - Splits text into manageable chunks for further processing.

3. **FAISS Vector Database**:
   - Stores vector embeddings of legal text chunks.
   - Enables fast and accurate semantic search.

4. **Neo4j Graph Database**:
   - Represents legal knowledge as a graph (e.g., articles, terms, relationships).
   - Supports structured search and contextual retrieval.

5. **Hybrid Retrieval Engine**:
   - Combines results from FAISS and Neo4j.
   - Ranks results based on a weighted combination of semantic and graph-based relevance.

6. **Query Classifier**:
   - Classifies user queries into legal or general categories.
   - Routes queries to the appropriate processing pipeline.

7. **LLM Integration**:
   - Uses Google's Gemini model for generating legal advice.
   - Structured prompts ensure accurate and context-aware responses.

---

## Workflow

1. **Input**:
   - The user submits a query in Arabic (e.g., "ما هي حقوق المرأة في الدستور المصري؟").

2. **Query Processing**:
   - The query is normalized and key legal terms are extracted.
   - The query is classified as legal or general.

3. **Document Retrieval**:
   - Semantic search is performed using FAISS to find relevant documents.
   - Graph-based search is performed using Neo4j to find articles related to the extracted legal terms.

4. **Result Combination**:
   - Results from FAISS and Neo4j are combined and ranked.
   - Contextual expansion is performed to include related articles.

5. **Output**:
   - The top results are formatted and presented to the user or LLM.

---

## Technical Stack

- **Programming Language**: Python
- **NLP Libraries**: Transformers, Tashaphyne, LangChain
- **Machine Learning**: Hugging Face models, FAISS
- **Databases**: Neo4j (graph database)
- **LLM Integration**: Google Gemini, Groq
- **PDF Processing**: pdfplumber
- **Arabic Text Handling**: arabic-reshaper, bidi.algorithm

---

## Getting Started

### Prerequisites

- Python 3.8 or higher.
- Neo4j database (with credentials).
- Google API key for Gemini integration.
- Required Python libraries (install via `pip install -r requirements.txt`).

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/MahmoudSaad21/Egyptian-Smart-Legal-Advisor.git
   cd egyptian-smart-legal-advisor
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Set up Neo4j:
   - Create a Neo4j instance and configure the connection details in `neo4j_config.json`.

4. Set up Google API key:
   - Add your Google API key to the environment variables:
     ```bash
     export GOOGLE_API_KEY="your-api-key"
     ```

### Running the System

1. **Notebook**:
   - Open the `Egyptian Smart Legal Advisor.ipynb` notebook in Jupyter or any compatible environment.
   - Follow the instructions in the notebook to process legal documents and run queries.

2. **Streamlit App**:
   - Run the Streamlit app using the following command:
     ```bash
     streamlit run app.py
     ```
   - Access the app in your browser at `http://localhost:8501`.

---
### Streamlit App Video Demo

Watch the video below to see the **Egyptian Smart Legal Advisor** in action!


https://github.com/user-attachments/assets/f5f55acc-62bf-408b-a6c4-b20fa42e4f5b


---

## Files and Components

- **Notebook**: `Egyptian Smart Legal Advisor.ipynb`
  - Contains the core logic for document processing, semantic search, and legal advice generation.

- **Streamlit App**: `app.py`
  - Provides a user-friendly interface for interacting with the system.

- **Legal Documents**:
- The Egyptian Constitution (الدستور المصري المعدل 2019.pdf).
- The Egyptian Civil Code (المدني.pdf).
- The Egyptian Criminal Procedures Code (قانون_الاجراءات_الجنائية.pdf)


https://github.com/user-attachments/assets/246fae9d-ef7c-444b-818f-aa3bd449052b



https://github.com/user-attachments/assets/016ca5a1-221c-4b46-b188-c1bc650051aa


---

## Example Use Cases

1. **Legal Research**:
   - Retrieve relevant articles from the Egyptian Constitution or Civil Code.
   - Find contextually related articles for comprehensive research.

2. **Legal Advice**:
   - Generate accurate legal advice based on Egyptian law.
   - Provide citations and references to relevant legal texts.

3. **Document Analysis**:
   - Extract and analyze key legal terms from legal documents.
   - Identify relationships between legal concepts.

---

## Future Enhancements

1. **Support for More Legal Sources**:
   - Expand the system to include additional legal documents (e.g., criminal code, labor law).

2. **Multilingual Support**:
   - Extend the system to handle queries in multiple languages (e.g., English, French).

3. **User Interface**:
   - Develop a web-based interface for easier interaction.

4. **Advanced Query Handling**:
   - Support for complex legal queries (e.g., multi-part questions, hypothetical scenarios).

5. **Integration with Legal Databases**:
   - Connect to external legal databases for real-time updates.

---

## Contact

For questions or feedback, please contact:

- **Name**: Mahmoud Saad Mahmoud
- **Email**: mahmoud.saad.mahmoud.11@gmail.com 

---

## Acknowledgments

- **Hugging Face** for providing pre-trained NLP models.
- **Neo4j** for the graph database technology.
- **Google** for the Gemini LLM integration.
- **FAISS** for efficient vector search capabilities.
