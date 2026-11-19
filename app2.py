
import os
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI


# --------------------------------------------------
# 1. Configuration
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
DOCS_DIR = BASE_DIR / "mydocs"

load_dotenv(BASE_DIR / ".env")

NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY")

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CHAT_MODEL = "nvidia/nemotron-3.5-lightning-30b-a3b"

MAX_CONTEXT_CHARS = 60000


# --------------------------------------------------
# 2. Load and extract every PDF
# --------------------------------------------------

def load_all_pdfs():

    if not DOCS_DIR.exists():
        raise FileNotFoundError(
            f"PDF directory not found: {DOCS_DIR}"
        )

    pdf_files = sorted(DOCS_DIR.glob("*.pdf"))

    if not pdf_files:
        raise ValueError(
            f"No PDF files found in {DOCS_DIR}"
        )

    all_pages = []
    inventory = []

    for pdf_file in pdf_files:

        try:
            loader = PyPDFLoader(str(pdf_file))
            pages = loader.load()

            extracted_chars = sum(
                len(page.page_content.strip())
                for page in pages
            )

            inventory.append({
                "filename": pdf_file.name,
                "pages": len(pages),
                "characters": extracted_chars,
                "status": (
                    "Text extracted"
                    if extracted_chars > 0
                    else "No text extracted"
                )
            })

            for page in pages:

                page.metadata["source_file"] = pdf_file.name

                # Convert page number to human-readable numbering
                page.metadata["page_number"] = (
                    page.metadata.get("page", 0) + 1
                )

                all_pages.append(page)

        except Exception as e:

            inventory.append({
                "filename": pdf_file.name,
                "pages": 0,
                "characters": 0,
                "status": f"Error: {str(e)}"
            })

    if not all_pages:
        raise ValueError(
            "No PDF pages could be extracted."
        )

    return all_pages, inventory


# --------------------------------------------------
# 3. Split extracted PDF text into chunks
# --------------------------------------------------

def split_documents(pages):

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", ". ", " ", ""]
    )

    chunks = splitter.split_documents(pages)

    if not chunks:
        raise ValueError(
            "No text chunks were created. Check PDF extraction."
        )

    return chunks


# --------------------------------------------------
# 4. Create embedding model and FAISS database
# --------------------------------------------------

@st.cache_resource
def load_embedding_model():

    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={
            "normalize_embeddings": True,
            "batch_size": 16
        }
    )


def build_vector_database(chunks):

    embeddings = load_embedding_model()

    vector_db = FAISS.from_documents(
        documents=chunks,
        embedding=embeddings
    )

    # Save a local copy of the FAISS index
    index_path = BASE_DIR / "faiss_index"

    vector_db.save_local(str(index_path))

    return vector_db


# --------------------------------------------------
# 5. Retrieve evidence from ALL documents
# --------------------------------------------------

def retrieve_evidence(question, vector_db):

    search_queries = [
        question,

        "Candidate resume professional summary, employment history, "
        "work experience, accomplishments, technical skills and education",

        "Job description required qualifications, preferred qualifications, "
        "responsibilities, technical skills and experience",

        "Candidate experience with AI, machine learning, cloud computing, "
        "solutions architecture, customer engineering and leadership"
    ]

    retrieved = []
    seen = set()

    for query in search_queries:

        results = vector_db.similarity_search(
            query,
            k=8
        )

        for doc in results:

            key = (
                doc.metadata.get("source_file", ""),
                doc.metadata.get("page_number", 0),
                doc.page_content[:200]
            )

            if key not in seen:
                seen.add(key)
                retrieved.append(doc)

    return retrieved


# --------------------------------------------------
# 6. Prepare document context
# --------------------------------------------------

def format_documents(documents):

    formatted = []

    for i, doc in enumerate(documents, start=1):

        filename = doc.metadata.get(
            "source_file",
            "Unknown document"
        )

        page = doc.metadata.get(
            "page_number",
            "Unknown"
        )

        formatted.append(
            f"""
DOCUMENT EVIDENCE {i}
Source file: {filename}
Page: {page}

{doc.page_content}
"""
        )

    return "\n".join(formatted)


def prepare_context(all_chunks, retrieved_chunks):

    # For reasonably sized document collections, include all text.
    # This prevents relevant resume sections from being missed.
    all_text = format_documents(all_chunks)

    if len(all_text) <= MAX_CONTEXT_CHARS:
        return all_text, "All extracted document chunks"

    # For larger collections, use semantic retrieval.
    retrieved_text = format_documents(retrieved_chunks)

    return retrieved_text[:MAX_CONTEXT_CHARS], "Retrieved relevant chunks"


# --------------------------------------------------
# 7. NVIDIA Nemotron
# --------------------------------------------------

def get_llm():

    if not NVIDIA_API_KEY:
        raise ValueError(
            "NVIDIA_API_KEY is missing. Check your .env file."
        )

    return ChatOpenAI(
        model=CHAT_MODEL,
        openai_api_key=NVIDIA_API_KEY,
        openai_api_base="https://integrate.api.nvidia.com/v1",
        temperature=0.1,
        max_tokens=3000,
        timeout=180,
        max_retries=2
    )


# --------------------------------------------------
# 8. Evaluate candidate against job description
# --------------------------------------------------

def evaluate_candidate(question, context):

    llm = get_llm()

    prompt = f"""
You are an experienced technical recruiter and hiring manager
evaluating candidates for technical and customer-facing engineering roles.

The documents below may contain:
- One or more candidate resumes
- One or more job descriptions
- Supporting professional documents

IMPORTANT:
Do not rely on filenames to identify documents.
Read the actual content and determine which documents are resumes
and which documents are job descriptions.

USER QUESTION:
{question}

DOCUMENT CONTENT:
{context}

YOUR TASK:

1. Identify the candidate's name from the resume, if available.

2. Identify the target job title and employer from the job description,
   if available.

3. Compare the candidate's actual experience, skills, education,
   accomplishments and technical background against the job requirements.

4. Make a clear hiring-fit decision.

Use exactly one of these verdicts:

FIT
PARTIAL FIT
NOT A FIT
INSUFFICIENT EVIDENCE

5. Explain the reasons for your decision.

6. Evaluate these areas where relevant:
   - Technical skills
   - AI, ML and Generative AI experience
   - Cloud and infrastructure experience
   - Software engineering and architecture
   - Customer-facing engineering experience
   - Solutions architecture and technical leadership
   - Problem solving and execution
   - Required years of experience
   - Preferred qualifications

7. Provide a requirement-by-requirement comparison table.

8. Identify:
   - Strong matches
   - Partial matches
   - Missing or weakly demonstrated qualifications
   - Questions to clarify during an interview

9. Give an overall fit score from 0 to 100.

10. Recommend whether the candidate should be:
    - Advanced to interview
    - Considered for another role
    - Not advanced

IMPORTANT RULES:

- Base conclusions on the supplied document content.
- Do not invent experience, skills, employers or qualifications.
- Distinguish demonstrated experience from skills that are merely mentioned.
- A missing skill in the resume is not proof that the candidate lacks it.
- Do not confuse the job requirements with the candidate's experience.
- If the resume or job description cannot be identified, explain that clearly.
- Be direct and make an actual recommendation.

FORMAT YOUR ANSWER:

# Candidate Fit Assessment

## Candidate and Target Role

## Overall Verdict

## Overall Fit Score

## Requirement-by-Requirement Comparison

| Job Requirement | Resume Evidence | Assessment |
|---|---|---|

## Strong Matches

## Partial Matches

## Gaps or Unclear Areas

## Interview Questions

## Hiring Recommendation

"""

    response = llm.invoke(prompt)

    return response.content


# --------------------------------------------------
# 9. Streamlit interface
# --------------------------------------------------

st.set_page_config(
    page_title="AI Resume and Job Fit Evaluator",
    page_icon="📄",
    layout="wide"
)

st.title("📄 AI Resume and Job Fit Evaluator")

st.write(
    "Upload resumes, job descriptions and supporting PDFs "
    "into the mydocs folder. The application will analyze all PDFs "
    "without depending on their filenames."
)

st.info(
    "Place all PDF documents in the project's ./mydocs folder. "
    "The application searches across all documents."
)


# --------------------------------------------------
# 10. Load documents and build vector database
# --------------------------------------------------

if "vector_db" not in st.session_state:

    with st.spinner("Reading and extracting PDF documents..."):

        try:

            pages, inventory = load_all_pdfs()

            chunks = split_documents(pages)

            st.session_state["pages"] = pages
            st.session_state["chunks"] = chunks
            st.session_state["inventory"] = inventory

        except Exception as e:

            st.error(f"PDF loading error: {e}")
            st.stop()

    with st.spinner("Creating embeddings and FAISS vector database..."):

        try:

            vector_db = build_vector_database(chunks)

            st.session_state["vector_db"] = vector_db

        except Exception as e:

            st.error(f"Vector database error: {e}")
            st.stop()


# --------------------------------------------------
# 11. Document diagnostics
# --------------------------------------------------

st.subheader("1. Indexed PDF Documents")

inventory = st.session_state["inventory"]

st.dataframe(
    inventory,
    use_container_width=True,
    hide_index=True
)

total_pages = len(st.session_state["pages"])
total_chunks = len(st.session_state["chunks"])

col1, col2, col3 = st.columns(3)

col1.metric("PDF Files", len(inventory))
col2.metric("Extracted Pages", total_pages)
col3.metric("Text Chunks Embedded", total_chunks)

failed_files = [
    item for item in inventory
    if item["characters"] == 0
]

if failed_files:

    st.warning(
        "Some PDFs did not produce extractable text. "
        "If these are scanned PDFs, OCR may be required."
    )


# --------------------------------------------------
# 12. Show extracted PDF text
# --------------------------------------------------

with st.expander("View extracted PDF text"):

    for page in st.session_state["pages"]:

        filename = page.metadata.get(
            "source_file",
            "Unknown"
        )

        page_number = page.metadata.get(
            "page_number",
            "Unknown"
        )

        st.markdown(
            f"**{filename} — Page {page_number}**"
        )

        st.text(page.page_content[:4000])

        st.divider()


# --------------------------------------------------
# 13. Candidate evaluation
# --------------------------------------------------

st.subheader("2. Evaluate Candidate")

question = st.text_area(
    "Enter your evaluation question",
    value=(
        "Is Nitu Parimi a good fit for the AMD Solutions Engineer role? "
        "Compare the resume against the job description and provide "
        "a clear hiring recommendation."
    ),
    height=120
)

if st.button("Evaluate Candidate", type="primary"):

    if not question.strip():

        st.warning("Please enter an evaluation question.")

    else:

        with st.spinner(
            "Retrieving document evidence and evaluating candidate..."
        ):

            try:

                vector_db = st.session_state["vector_db"]
                chunks = st.session_state["chunks"]

                retrieved = retrieve_evidence(
                    question,
                    vector_db
                )

                context, context_source = prepare_context(
                    chunks,
                    retrieved
                )

                st.session_state["retrieved"] = retrieved

                answer = evaluate_candidate(
                    question,
                    context
                )

                st.session_state["answer"] = answer
                st.session_state["context_source"] = context_source

            except Exception as e:

                st.error(f"Evaluation error: {e}")


# --------------------------------------------------
# 14. Display assessment
# --------------------------------------------------

if "answer" in st.session_state:

    st.subheader("3. Candidate Assessment")

    st.caption(
        f"Context supplied to the model: "
        f"{st.session_state['context_source']}"
    )

    st.markdown(st.session_state["answer"])

    with st.expander("View retrieved evidence"):

        for i, doc in enumerate(
            st.session_state.get("retrieved", []),
            start=1
        ):

            filename = doc.metadata.get(
                "source_file",
                "Unknown"
            )

            page = doc.metadata.get(
                "page_number",
                "Unknown"
            )

            st.markdown(
                f"**Evidence {i}: {filename}, Page {page}**"
            )

            st.write(doc.page_content)

            st.divider()


# --------------------------------------------------
# 15. Sidebar controls
# --------------------------------------------------

with st.sidebar:

    st.header("Application Settings")

    st.write(f"PDF directory: `{DOCS_DIR}`")

    st.write(f"Embedding model: `{EMBEDDING_MODEL}`")

    st.write(f"LLM: `{CHAT_MODEL}`")

    st.divider()

    if st.button("Rebuild PDF Index"):

        st.cache_resource.clear()

        for key in [
            "vector_db",
            "pages",
            "chunks",
            "inventory",
            "answer",
            "retrieved",
            "context_source"
        ]:

            st.session_state.pop(key, None)

        st.rerun()
