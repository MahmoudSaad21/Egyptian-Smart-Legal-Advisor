# Imports arranged by category
import re
import time
import json
import os
from typing import List, Dict, Any, Literal

# Data processing
import torch
import numpy as np
import arabic_reshaper
import unicodedata
from bidi.algorithm import get_display
import pdfplumber
from tashaphyne.stemming import ArabicLightStemmer

# NLP and ML libraries
from transformers import pipeline

# Database
from neo4j import GraphDatabase

# Vector stores and embeddings
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter

# LLM chains and components
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableBranch, RunnableLambda
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI

# Frontend
import streamlit as st
from streamlit_chat import message


# Apply RTL for Arabic UI
st.set_page_config(page_title="المستشار القانوني الذكي", layout="wide")
st.markdown("""
    <style>
        body { direction: rtl; text-align: right; }
        .stTextInput > div > div > input { text-align: right; }
        .st-emotion-cache-16txtl3 { padding: 1rem 1rem 1rem 10rem; }
        .st-emotion-cache-13ejsyy { direction: rtl; }
    </style>
""", unsafe_allow_html=True)

# Arabic Text Processor for Normalization
class ArabicTextProcessor:
    def __init__(self):
        self.stopwords = self.load_arabic_stopwords()
        self.stemmer = ArabicLightStemmer()  # Light stemming for Arabic

    def load_arabic_stopwords(self):
        """Loads common Arabic stopwords."""
        return set(["في", "من", "على", "و", "ال", "الى", "إلى", "عن", "مع", "هذا", "هذه", "ذلك", "تلك", "أو", "ثم", "لكن", "و", "ف", "ب", "ل"])

    def normalize_arabic_text(self, text):
        """Normalize Arabic text by removing diacritics, normalizing letters, and stemming."""
        text = re.sub(r'[\u064B-\u065F\u0670]', '', text)  # Remove diacritics
        text = re.sub(r'[إأآا]', 'ا', text)  # Normalize alef
        text = text.replace('ة', 'ه')  # Normalize teh marbuta
        text = text.replace('ى', 'ي')  # Normalize yeh
        text = text.replace('ـ', '')  # Remove kashida

        return text

    def extract_key_legal_terms(self, text):
        """Extract key legal terms from a scenario to improve retrieval"""
        legal_term_patterns = [
            r'(جريمة|تهمة|قضية|دعوى|محكمة|قانون|حكم|عقوبة|سجن|غرامة|تعويض|مخالفة|جنحة|جناية)',
            r'(متهم|شاهد|محام|قاضي|مدعي|مدعى عليه|نيابة|تحقيق|محضر|أدلة|إثبات|براءة|إدانة)',
            r'(إجراءات|استئناف|نقض|طعن|دفاع|اتهام|محاكمة|جلسة|حبس|توقيف|احتجاز|تفتيش|ضبط)',
            r'(تقادم|تقادم مكسب|تقادم مسقط|مدة التقادم|مرور الزمن|انقضاء الدعوى|سقوط الحق بالتقادم)',
            r'(استعمال الحق|سوء استعمال الحق|تجاوز حدود الحق|حق مشروع|حق غير مشروع|إساءة استعمال الحق)',
            r'(جريمة|دعوى|محكمة|حكم|قانون|التزام|مسؤولية|عقد|التزام قانوني|إجراءات|حقوق|مخالفة)',
            r'(مدعي|مدعى عليه|محام|قاضي|تحقيق|إثبات|براءة|إدانة|إجراءات قانونية|عقوبة|دعوى مدنية)',
            r'(حقوق المرأة|المساواة بين الجنسين|تمكين المرأة|حماية المرأة|التمثيل السياسي للمرأة)',
            r'(عدم التمييز|الحقوق الدستورية|المشاركة السياسية|العمل|الأجور|الأحوال الشخصية)',
            r'(الدستور المصري|المجلس القومي للمرأة|القانون المدني|القانون الجنائي|الأحوال الشخصية)',
            r'(استعمال الحق|سوء استعمال الحق|تجاوز حدود الحق|حق مشروع|حق غير مشروع)',
            r'(القانون المدني|المسؤولية التقصيرية|المسؤولية العقدية|التعويض المدني)',
            r'(العقد|الالتزام|الإرادة المنفردة|الإثراء بلا سبب|الفعل الضار|الضرر)'
        ]

        key_terms = []
        for pattern in legal_term_patterns:
            matches = re.findall(pattern, text)
            key_terms.extend([self.stemmer.light_stem(word) for word in matches])

        return list(set(key_terms))  # Return unique terms

# Function to extract text from Arabic PDF using pdfplumber
def extract_text_from_pdf(pdf_path):
    """Extracts Arabic text from a PDF file and fixes reversed text issues"""
    extracted_text = ""

    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                reshaped_text = arabic_reshaper.reshape(text)  # Fix Arabic shaping
                fixed_text = get_display(reshaped_text)  # Correct text order
                extracted_text += fixed_text + "\n"

    if not extracted_text.strip():
        st.warning(f"⚠️ لا يوجد نص مستخرج من {pdf_path}! تحقق ما إذا كان ملف PDF ممسوحًا ضوئيًا أو يستخدم خطوطًا مضمنة.")

    normalized_text = unicodedata.normalize("NFKC", extracted_text)
    normalized_text = re.sub(r'\bلا(?=[ا-ي])', 'ال', normalized_text)  # Beginning of word
    normalized_text = re.sub(r'(?<=[ا-ي])لا\b', 'ال', normalized_text)  # End of word
    normalized_text = re.sub(r"\)([^\)]+)\(", r"(\1)", normalized_text)
    return normalized_text

# Function to split legal text based on article numbers and law source
def split_legal_text(text, source_name):
    """Splits legal text into structured chunks using Arabic article numbers."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=[".", "؟", "\n\n"]
    )

    pages = splitter.split_text(text)
    # Rebuild chunks, ensuring each article stays with its content
    chunks = []
    for i, page in enumerate(pages, start=1):  # Start chunk numbering from 1
        chunks.append({
            "content": page.strip(),
            "source": source_name,
            "article_id": str(i)  # Using chunk number as article_id
        })

    return chunks

# Function to load and process multiple Arabic PDFs
def load_and_process_pdfs(legal_files, progress_bar=None):
    """Loads and processes multiple Arabic PDFs for retrieval"""
    all_chunks = []
    total_files = len(legal_files)

    for idx, (pdf_path, source_name) in enumerate(legal_files):
        if progress_bar:
            progress_bar.progress((idx) / total_files, text=f"جاري معالجة {source_name}...")

        # Extract text using pdfplumber
        extracted_text = extract_text_from_pdf(pdf_path)

        if not extracted_text:
            if progress_bar:
                progress_bar.warning(f"❌ فشل في استخراج النص من {pdf_path}")
            continue

        # Split text into structured chunks with source information
        chunks = split_legal_text(extracted_text, source_name)
        all_chunks.extend(chunks)

        if progress_bar:
            progress_bar.progress((idx + 1) / total_files, text=f"تم معالجة {source_name} بنجاح!")

    if not all_chunks:
        if progress_bar:
            progress_bar.error("❌ لم يتم العثور على أي أجزاء صالحة من أي مستند!")
        return []

    return all_chunks

# Neo4j Graph Database Handler
class LegalGraphDB:
    def __init__(self, uri, username, password):
        self.driver = GraphDatabase.driver(uri, auth=(username, password))
        self.setup_schema()

    def close(self):
        self.driver.close()

    def setup_schema(self):
        """Setup Neo4j schema with constraints and indexes"""
        with self.driver.session() as session:
            # Create constraints for uniqueness
            session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (l:LegalArticle) REQUIRE l.id IS UNIQUE")
            session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (s:Source) REQUIRE s.name IS UNIQUE")
            session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (t:Term) REQUIRE t.name IS UNIQUE")

            # Create indexes for faster retrieval
            session.run("CREATE INDEX IF NOT EXISTS FOR (l:LegalArticle) ON (l.article_id)")
            session.run("CREATE INDEX IF NOT EXISTS FOR (t:Term) ON (t.name)")

    def add_legal_chunks(self, chunks, progress_bar=None):
        """Add legal chunks to Neo4j with metadata and relationships"""
        total_chunks = len(chunks)

        with self.driver.session() as session:
            for i, chunk in enumerate(chunks):
                if progress_bar and i % 10 == 0:  # Update progress every 10 chunks
                    progress_bar.progress((i) / total_chunks, text=f"إضافة البيانات إلى Neo4j... {i}/{total_chunks}")

                # Generate a unique ID for the chunk
                chunk_id = f"{chunk['source']}-{chunk['article_id']}-{hash(chunk['content'])}"

                # Create the legal article node with all metadata
                session.run("""
                    MERGE (s:Source {name: $source})
                    MERGE (l:LegalArticle {
                        id: $id,
                        content: $content,
                        article_id: $article_id
                    })
                    MERGE (l)-[:FROM_SOURCE]->(s)
                """, {
                    "id": chunk_id,
                    "content": chunk["content"],
                    "article_id": chunk["article_id"],
                    "source": chunk["source"]
                })

                # Extract key terms and create relationships
                processor = ArabicTextProcessor()
                key_terms = processor.extract_key_legal_terms(chunk["content"])

                for term in key_terms:
                    session.run("""
                        MATCH (l:LegalArticle {id: $chunk_id})
                        MERGE (t:Term {name: $term})
                        CREATE (l)-[:MENTIONS]->(t)
                    """, {
                        "chunk_id": chunk_id,
                        "term": term
                    })

            if progress_bar:
                progress_bar.progress(1.0, text="تم إنشاء قاعدة بيانات Neo4j بنجاح!")

    def search_related_articles(self, terms, limit=5):
        """Search for related articles based on legal terms"""
        with self.driver.session() as session:
            result = session.run("""
                UNWIND $terms AS term
                MATCH (t:Term {name: term})<-[:MENTIONS]-(l:LegalArticle)-[:FROM_SOURCE]->(s:Source)
                RETURN l.id AS id, l.content AS content, l.article_id AS article_id,
                       s.name AS source, count(t) AS relevance
                ORDER BY relevance DESC
                LIMIT $limit
            """, {"terms": terms, "limit": limit})

            return [record.data() for record in result]

    def search_contextual_articles(self, article_id, source, limit=3):
        """Find contextually related articles based on legal hierarchy"""
        with self.driver.session() as session:
            result = session.run("""
                MATCH (l:LegalArticle)-[:FROM_SOURCE]->(s:Source {name: $source})
                WHERE abs(toInteger(l.article_id) - toInteger($article_id)) <= 5
                AND l.article_id <> $article_id
                RETURN l.id AS id, l.content AS content, l.article_id AS article_id,
                      s.name AS source
                ORDER BY abs(toInteger(l.article_id) - toInteger($article_id))
                LIMIT $limit
            """, {"article_id": article_id, "source": source, "limit": limit})

            return [record.data() for record in result]

    def check_database_populated(self):
        """Check if the database has any data"""
        with self.driver.session() as session:
            result = session.run("MATCH (l:LegalArticle) RETURN count(l) as count")
            record = result.single()
            if record and record["count"] > 0:
                return True
            return False

# Function to create vector embeddings using FAISS with an Arabic model
def create_faiss_embeddings(chunks, progress_bar=None):
    """Creates FAISS vector embeddings for the text using an Arabic-optimized model"""
    if not chunks:
        if progress_bar:
            progress_bar.warning("⚠️ لا توجد أجزاء نصية للتضمين. تخطي عملية التضمين!")
        return None

    if progress_bar:
        progress_bar.progress(0.3, text="جاري تحميل نموذج التضمين...")

    # Extract just the content field for embedding
    texts = [chunk["content"] for chunk in chunks]
    metadatas = [{"source": chunk["source"], "article_id": chunk["article_id"]} for chunk in chunks]

    embedding_model = HuggingFaceEmbeddings(
        model_name="silma-ai/silma-embeddding-sts-v0.1",  # Arabic-optimized embedding model
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )

    if progress_bar:
        progress_bar.progress(0.6, text="جاري إنشاء تضمينات المتجهات...")

    vectordb = FAISS.from_texts(
        texts=texts,
        embedding=embedding_model,
        metadatas=metadatas
    )

    # Save the FAISS index for future use
    vectordb.save_local("./arabic_law_faiss")

    if progress_bar:
        progress_bar.progress(1.0, text="تم إنشاء قاعدة بيانات FAISS بنجاح!")

    return vectordb

# Setup query classifier using a more robust approach
def setup_query_classifier():
    """Setup a classifier for legal query types"""
    # Using Arabic-compatible multilingual model
    classifier = pipeline(
        "zero-shot-classification",
        model="joeddav/xlm-roberta-large-xnli",
    )

    def classify_query(query):
        # Define classes in Arabic
        classes = [
            "سؤال عن قانون محدد",     # Question about specific law
            "استفسار عن حالة قانونية", # Legal scenario inquiry
            "طلب استشارة قانونية",     # Request for legal advice
            "سؤال عام",                # General question
            "دردشة عامة"               # General chat
        ]

        result = classifier(
            query,
            candidate_labels=classes,
            hypothesis_template="هذا النص يتعلق بـ {}."
        )

        # Get the top class and confidence
        top_class = result["labels"][0]
        confidence = result["scores"][0]

        # Map to simplified categories
        if top_class in ["سؤال عن قانون محدد", "استفسار عن حالة قانونية", "طلب استشارة قانونية"]:
            query_type = "legal_query"
        else:
            query_type = "general_query"

        return {
            "detailed_type": top_class,
            "query_type": query_type,
            "confidence": confidence
        }

    return classify_query

# Enhanced retrieval function integrating FAISS and Neo4j
def hybrid_retrieval(query, faiss_db, graph_db, top_k=10):
    """Perform hybrid retrieval combining vector search and graph traversal"""
    processor = ArabicTextProcessor()

    # Step 1: Extract key legal terms for Neo4j search
    legal_terms = processor.extract_key_legal_terms(query)

    # Step 2: Perform FAISS vector similarity search
    normalized_query = processor.normalize_arabic_text(query)
    vector_results = faiss_db.similarity_search_with_score(query, k=top_k)

    # Step 3: Perform Neo4j graph-based search
    graph_results = []
    if legal_terms:
        graph_results = graph_db.search_related_articles(legal_terms, limit=top_k)

    # Step 4: Combine and rank results
    combined_results = []

    # Process vector results
    for doc, score in vector_results:
        retrieved_doc = {
            "content": doc.page_content,
            "source": doc.metadata.get("source", "Unknown"),
            "article_id": doc.metadata.get("article_id", "Unknown"),
            "vector_score": score,
            "graph_score": 0,
            "total_score": 1 / (1 + score)  # Convert distance to similarity score
        }
        combined_results.append(retrieved_doc)

    # Process graph results and merge with vector results if they exist
    for result in graph_results:
        # Check if this result already exists in combined_results
        existing = next((item for item in combined_results if
                         item["content"] == result["content"] and
                         item["source"] == result["source"]), None)

        if existing:
            # Update the existing entry with graph score
            existing["graph_score"] = result["relevance"]
            existing["total_score"] = existing["total_score"] + (result["relevance"] * 0.2)  # Weight graph results
        else:
            # Add new entry
            retrieved_doc = {
                "content": result["content"],
                "source": result["source"],
                "article_id": result["article_id"],
                "vector_score": 0,
                "graph_score": result["relevance"],
                "total_score": result["relevance"] * 0.2  # Weight graph results
            }
            combined_results.append(retrieved_doc)

    # Step 5: Sort by total score and get top results
    combined_results.sort(key=lambda x: x["total_score"], reverse=True)

    # Step 6: Context expansion - for top results, find contextually related articles
    top_results = combined_results[:min(5, len(combined_results))]
    expanded_results = list(top_results)

    for result in top_results:
        if result["article_id"] != "غير محدد":
            related = graph_db.search_contextual_articles(
                result["article_id"],
                result["source"],
                limit=2
            )

            for rel in related:
                # Check if already in results
                if not any(r["content"] == rel["content"] for r in expanded_results):
                    expanded_results.append({
                        "content": rel["content"],
                        "source": rel["source"],
                        "article_id": rel["article_id"],
                        "vector_score": 0,
                        "graph_score": 0.3,
                        "total_score": 0.3,
                        "context_relation": f"Related to Article {result['article_id']}"
                    })

    # Get final top results (limit to reasonable number)
    final_results = expanded_results[:min(10, len(expanded_results))]

    # Format for LLM
    formatted_results = []
    for i, res in enumerate(final_results):
        formatted_results.append(
            f"【{i+1}】 المصدر: {res['source']} | المادة: {res['article_id']}\n{res['content']}\n"
        )

    return "\n\n".join(formatted_results)

# Function to build legal advisor chain with context handling
def build_legal_advisor_chain(faiss_db, graph_db):
    """Builds an enhanced legal advisor chain with context handling"""
    # Set Google API key
    os.environ["GOOGLE_API_KEY"] = "AIzaSyAmzcdK1AXIndg6LTR0QXhV4mNy4hMbkqY"  # Replace with your actual API key

    # Create LLM
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        google_api_key=os.environ["GOOGLE_API_KEY"],
        temperature=0.2
    )

    # Create enhanced prompt template with history and context
    prompt_template = """
    أنت مستشار قانوني مصري متخصص في القانون المصري، مهمتك تقديم مشورة قانونية دقيقة استناداً إلى النصوص القانونية.

    # الاستفسار الحالي:
    {query}

    # النصوص القانونية ذات الصلة:
    {context}

    # التاريخ السابق للاستفسارات (إن وجد):
    {history}

    # مهمتك:
    1. حلل الاستفسار بدقة وحدد العناصر القانونية الرئيسية فيه.
    2. استخرج المواد القانونية المناسبة من النصوص المقدمة.
    3. اشرح كيف تنطبق هذه المواد على الحالة المعروضة.
    4. قدم المشورة القانونية بناءً على النصوص القانونية فقط، وليس رأيك الشخصي.
    5. اذكر المصدر القانوني والمواد التي استندت إليها بوضوح.
    6. تجنب الاستنتاجات غير المدعمة بنصوص قانونية.

    # المشورة القانونية:
    """

    # Function to retrieve documents
    def retrieve_docs(query):
        return hybrid_retrieval(query, faiss_db, graph_db)

    # Query history handler
    def process_with_history(query):
        # Add to history (limiting to last 3 queries for context)
        if 'query_history' not in st.session_state:
            st.session_state.query_history = []

        st.session_state.query_history.append(query)
        if len(st.session_state.query_history) > 3:
            st.session_state.query_history.pop(0)

        # Format history
        history_text = ""
        if len(st.session_state.query_history) > 1:
            history_text = "الاستفسارات السابقة:\n" + "\n".join([
                f"- {q}" for q in st.session_state.query_history[:-1]
            ])

        # Get context from retrieval
        context = retrieve_docs(query)

        return {
            "query": query,
            "context": context,
            "history": history_text
        }

    # Create the chain
    legal_chain = (
        RunnableLambda(process_with_history)
        | PromptTemplate.from_template(prompt_template)
        | llm
        | StrOutputParser()
    )

    return legal_chain

# Function to build general conversation chain
def build_general_chain():
    """Builds a chain for general conversation"""
    # Set Google API key
    os.environ["GOOGLE_API_KEY"] = "Your API key"  # Replace with your actual API key

    # Create LLM
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        google_api_key=os.environ["GOOGLE_API_KEY"],
        temperature=0.7
    )

    prompt = PromptTemplate.from_template("""
    أنت مساعد ودود يتحدث العربية بطلاقة. يرجى الرد على رسالة المستخدم بشكل طبيعي ولطيف:

    المستخدم: {query}
    """)

    return prompt | llm | StrOutputParser()

# Function to build complete system with routing
def initialize_system():
    """Initialize the legal advisory system once and store in session state"""
    if 'system_initialized' in st.session_state and st.session_state.system_initialized:
        return True

    # Define legal files to process
    legal_files = [
        ("/content/الدستور المصري المعدل 2019.pdf", "الدستور"),
        ("/content/المدني.pdf", "القانون المدني"),
        ("/content/قانون_الاجراءات_الجنائية.pdf", "الإجراءات الجنائية")
    ]

    # Neo4j configuration
    neo4j_config = {
        "uri": "Your URI",
        "username": "neo4j",
        "password": "Your_password"
    }

    # Create progress container
    progress_container = st.empty()

    with progress_container.container():
        st.markdown("## تهيئة نظام المستشار القانوني")
        progress_bar = st.progress(0.0, text="بدء التهيئة...")

        # Step 1: Connect to Neo4j and check if database already populated
        progress_bar.progress(0.1, text="الاتصال بقاعدة بيانات Neo4j...")
        graph_db = LegalGraphDB(
            neo4j_config["uri"],
            neo4j_config["username"],
            neo4j_config["password"]
        )

        # Check if database already populated
        db_populated = graph_db.check_database_populated()

        # Step 2: Load and process data if needed
        if db_populated and os.path.exists("./arabic_law_faiss"):
            progress_bar.progress(0.5, text="قاعدة البيانات موجودة بالفعل، جاري تحميل النماذج...")
            st.success("تم اكتشاف قاعدة بيانات موجودة. جاري استخدامها...")

            # Load existing FAISS database
            progress_bar.progress(0.8, text="جاري تحميل قاعدة بيانات FAISS...")
            embedding_model = HuggingFaceEmbeddings(
                model_name="silma-ai/silma-embeddding-sts-v0.1",
                model_kwargs={'device': 'cpu'},
                encode_kwargs={'normalize_embeddings': True}
            )
            faiss_db = FAISS.load_local("./arabic_law_faiss", embedding_model, allow_dangerous_deserialization=True)


        else:
            st.info("لم يتم العثور على قاعدة بيانات موجودة. جاري إنشاء قاعدة بيانات جديدة...")

            # Process PDFs
            progress_bar.progress(0.2, text="جاري معالجة ملفات PDF...")
            all_chunks = load_and_process_pdfs(legal_files, progress_bar)

            if not all_chunks:
                st.error("❌ فشل في معالجة النصوص القانونية!")
                return False

            # Create FAISS vector database
            progress_bar.progress(0.5, text="جاري إنشاء قاعدة بيانات FAISS...")
            faiss_db = create_faiss_embeddings(all_chunks, progress_bar)

            # Add chunks to Neo4j
            progress_bar.progress(0.7, text="جاري إضافة البيانات إلى Neo4j...")
            graph_db.add_legal_chunks(all_chunks, progress_bar)

        # Step 3: Set up query classifier
        progress_bar.progress(0.9, text="جاري إعداد مصنف الاستعلامات...")
        query_classifier = setup_query_classifier()

        # Step 4: Build advisor chains
        progress_bar.progress(0.95, text="جاري إنشاء سلاسل المعالجة...")
        legal_chain = build_legal_advisor_chain(faiss_db, graph_db)
        general_chain = build_general_chain()

        # Store components in session state
        st.session_state.faiss_db = faiss_db
        st.session_state.graph_db = graph_db
        st.session_state.query_classifier = query_classifier
        st.session_state.legal_chain = legal_chain
        st.session_state.general_chain = general_chain
        st.session_state.system_initialized = True

        progress_bar.progress(1.0, text="تم إنشاء النظام بنجاح!")
        time.sleep(1)  # Short pause to show completion

    progress_container.empty()  # Clear the progress interface after completion
    return True

# Function to route queries to the right chain
def route_query(query_text):
    """Routes a query to either legal or general chain based on classification"""
    # Classify query
    classification = st.session_state.query_classifier(query_text)

    # Route to appropriate chain
    if classification["query_type"] == "legal_query":
        return {
            "chain": st.session_state.legal_chain,
            "query_type": "legal_query",
            "detailed_type": classification["detailed_type"]
        }
    else:
        return {
            "chain": st.session_state.general_chain,
            "query_type": "general_query",
            "detailed_type": classification["detailed_type"]
        }


# Main UI function
def main():
    st.title("المستشار القانوني الذكي 🧑‍⚖️")
    st.markdown("""
    مرحبًا بك في المستشار القانوني الذكي، مساعدك الشخصي للإجابة على استفساراتك القانونية
    وفقًا للقوانين المصرية. يمكنك طرح أسئلتك حول:

    - الدستور المصري
    - القانون المدني
    - قانون الإجراءات الجنائية
    """)

    # Initialize the system once
    system_ready = initialize_system()

    if not system_ready:
        st.error("❌ فشل في تهيئة النظام. يرجى المحاولة مرة أخرى.")
        return

    # Initialize chat history
    if 'messages' not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "مرحبًا بك! أنا المستشار القانوني الذكي. كيف يمكنني مساعدتك اليوم في الاستفسارات القانونية المتعلقة بالقانون المصري؟"}
        ]

    # Display chat messages - Fixed method for displaying chat messages
    for i, msg in enumerate(st.session_state.messages):
        if msg["role"] == "user":
            st.chat_message("user").write(msg["content"])
        else:
            st.chat_message("assistant").write(msg["content"])

    # Chat input
    if prompt := st.chat_input("اكتب استفسارك القانوني هنا..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        message(prompt, is_user=True, key=str(hash(prompt)))

        # Show thinking indicator
        with st.status("جاري التفكير...", expanded=True) as status:
            # Route the query
            router_result = route_query(prompt)

            # Display query classification
            query_type_arabic = "استفسار قانوني" if router_result["query_type"] == "legal_query" else "استفسار عام"
            st.write(f"نوع الاستفسار: {query_type_arabic} ({router_result['detailed_type']})")

            # Process with appropriate chain
            try:
                response = router_result["chain"].invoke(prompt)
                status.update(label="تم!", state="complete", expanded=False)
            except Exception as e:
                st.error(f"حدث خطأ أثناء معالجة استفسارك: {str(e)}")
                response = "عذرًا، حدث خطأ أثناء معالجة استفسارك. يرجى المحاولة مرة أخرى أو إعادة صياغة سؤالك."

        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})
        message(response, is_user=False)

# Add feedback mechanism
def add_feedback_section():
    st.sidebar.markdown("---")
    st.sidebar.markdown("### تقييم الإجابة")
    col1, col2 = st.sidebar.columns(2)

    with col1:
        if st.button("👍 مفيد"):
            st.success("شكرًا على تقييمك الإيجابي!")

    with col2:
        if st.button("👎 غير مفيد"):
            feedback = st.text_area("ما الذي يمكننا تحسينه؟")
            if st.button("إرسال"):
                st.success("شكرًا على ملاحظاتك!")

# Add sidebar information
def add_sidebar():
    st.sidebar.title("حول المستشار القانوني الذكي")
    st.sidebar.markdown("""
    **المستشار القانوني الذكي** هو نظام ذكاء اصطناعي متخصص في القانون المصري، يساعدك في:

    - فهم النصوص القانونية
    - الإجابة على الاستفسارات القانونية
    - توفير مراجع للمواد القانونية ذات الصلة

    💡 **ملاحظة هامة**: هذا النظام مصمم للمساعدة فقط ولا يغني عن استشارة محامٍ مؤهل.
    """)

    # Add databases info
    st.sidebar.markdown("### مصادر البيانات")
    st.sidebar.markdown("""
    - الدستور المصري المعدل 2019
    - القانون المدني المصري
    - قانون الإجراءات الجنائية
    """)

    # Add feedback section
    add_feedback_section()

# Run the application
if __name__ == "__main__":
    add_sidebar()
    main()
