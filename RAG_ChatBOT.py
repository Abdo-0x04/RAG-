# Run with:  streamlit run app.py
# install all needed packages with:
# pip install streamlit langchain langchain-community langchain-cohere langchain-openai langchain-text-splitters langchain-huggingface faiss-cpu pypdf sentence-transformers rank_bm25
import uuid
import os
import tempfile
import time
from dotenv import load_dotenv
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_cohere import ChatCohere, CohereRerank
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_classic.retrievers import ContextualCompressionRetriever, EnsembleRetriever
from langchain_openai import ChatOpenAI
load_dotenv()

st.set_page_config(page_title="Multi-PDF RAG Chatbot", page_icon="📚", layout="wide")
st.title("📚 Multi-PDF RAG Chatbot")

with st.sidebar:
    st.header("Setup")

    provider = st.selectbox(
        "LLM Provider",
        ["Cohere", "OpenAI-compatible (custom base URL)"],
        help=(
            "'OpenAI-compatible' works with OpenAI itself, and with any provider that "
            "exposes an OpenAI-style /v1/chat/completions endpoint — e.g. Groq, "
            "Together.ai, OpenRouter, Fireworks, or a local server like Ollama/LM Studio."
        ),
    )

    if provider == "Cohere":
        api_key_input = st.text_input(
            "Cohere API Key",
            type="password",
            value="",
            help="Get one at dashboard.cohere.com",
        )
        model_name = st.text_input("Model name", value="command-a-03-2025")
        base_url = None
        if api_key_input:
            os.environ["COHERE_API_KEY"] = api_key_input
        provider_ready = bool(api_key_input)

    else: 
        base_url = st.text_input(
            "Base URL",
            value=os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"),
            help="e.g. https://api.openai.com/v1, https://api.groq.com/openai/v1, "
                 "http://localhost:11434/v1 (Ollama)",
        )
        api_key_input = st.text_input(
            "API Key",
            type="password",
            value="",
            help="Some local servers (e.g. Ollama) accept any placeholder string here.",
        )
        model_name = st.text_input("Model name", value="gpt-4o-mini")
        if api_key_input:
            os.environ["OPENAI_API_KEY"] = api_key_input
        provider_ready = bool(api_key_input and base_url)

    st.divider()
    st.caption(
        "Rerank always uses Cohere (rerank-v3.5), independent of your chosen LLM provider. "
        "Add a Cohere key below to enable it; leave blank to skip reranking."
    )
    rerank_key_input = st.text_input(
        "Cohere API Key (for Rerank only)",
        type="password",
        value="",
        help="Only needed if you want the Rerank bonus feature; leave blank otherwise.",
    )
    if rerank_key_input:
        os.environ["COHERE_API_KEY"] = rerank_key_input
    rerank_available = bool(os.environ.get("COHERE_API_KEY"))

if not provider_ready:
    st.warning("Enter your provider's API key (and base URL, if applicable) in the sidebar to continue.")
    st.stop()

@st.cache_resource(show_spinner=False)
def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

@st.cache_resource(show_spinner=False)
def get_llm(provider: str, model_name: str, base_url: str | None, api_key: str):
    if provider == "Cohere":
        return ChatCohere(model=model_name, temperature=0, cohere_api_key=api_key)
    return ChatOpenAI(
        model=model_name,
        temperature=0,
        base_url=base_url,
        api_key=api_key,
    )

embeddings = get_embeddings()
llm = get_llm(provider, model_name, base_url, api_key_input)

if "messages" not in st.session_state:
    st.session_state.messages = [] 
if "all_chunks" not in st.session_state:
    st.session_state.all_chunks = [] 
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "indexed_files" not in st.session_state:
    st.session_state.indexed_files = set() 
if "global_bm25" not in st.session_state:
    st.session_state.global_bm25 = None 

with st.sidebar:
    st.header("1. Upload PDFs")
    uploaded_files = st.file_uploader(
        "Upload one or more PDFs", type="pdf", accept_multiple_files=True
    )

    st.header("2. Chunking")
    chunk_size = st.slider("Chunk size", 300, 1500, 800, step=100)
    chunk_overlap = st.slider("Chunk overlap", 0, 300, 100, step=25)

    process_clicked = st.button("Process / Re-index PDFs", type="primary")

def load_and_tag_pdf(file_bytes, filename):
    """Save an uploaded PDF to a temp file, load it, tag every page with its source."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name
    loader = PyPDFLoader(tmp_path)
    docs = loader.load()
    for d in docs:
        d.metadata["source_file"] = filename
        d.metadata["page_display"] = d.metadata.get("page", 0) + 1
    os.unlink(tmp_path)
    return docs

if process_clicked:
    if not uploaded_files:
        st.sidebar.error("Upload at least one PDF first.")
    else:
        with st.spinner("Loading and indexing PDFs..."):
            all_documents = []
            for uf in uploaded_files:
                docs = load_and_tag_pdf(uf.read(), uf.name)
                all_documents.extend(docs)
                st.session_state.indexed_files.add(uf.name)

            splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size, chunk_overlap=chunk_overlap
            )
            chunks = splitter.split_documents(all_documents)
            st.session_state.all_chunks = chunks

            # Build FAISS Vectorstore
            st.session_state.vectorstore = FAISS.from_documents(
                documents=chunks,
                embedding=embeddings
            )
            
            st.session_state.global_bm25 = BM25Retriever.from_documents(chunks)
            
        st.sidebar.success(
            f"Indexed {len(uploaded_files)} PDF(s), {len(chunks)} chunks total."
        )

if st.session_state.indexed_files:
    st.sidebar.caption("Indexed files: " + ", ".join(sorted(st.session_state.indexed_files)))

with st.sidebar:
    st.header("3. Retrieval settings")

    selected_files = st.multiselect(
        "Search only these PDFs (empty = search all)",
        options=sorted(st.session_state.indexed_files),
        default=[],
    )

    k = st.slider("k (chunks retrieved before rerank)", 2, 15, 6)
    final_k = st.slider("Top N kept after rerank", 1, k, min(4, k))

    use_hybrid = st.checkbox("Use hybrid retrieval (semantic + BM25 keyword)", value=True)
    use_rerank = st.checkbox(
        "Apply Cohere Rerank (Advanced)",
        value=False, 
        disabled=not rerank_available,
        help="Enables Cohere Rerank v3.5 to boost precision. Requires a Cohere API key."
        if rerank_available
        else "Add a Cohere API key in the Setup section to enable reranking.",
    )

    if use_hybrid:
        semantic_weight = st.slider("Semantic weight (vs keyword)", 0.0, 1.0, 0.6)

    if st.button("Clear conversation"):
        st.session_state.messages = []
        st.rerun()

def build_retriever():
    vs = st.session_state.vectorstore
    if vs is None:
        return None

    search_kwargs = {"k": k}
    if selected_files:
        search_kwargs["filter"] = lambda metadata: metadata.get("source_file") in selected_files

    semantic_retriever = vs.as_retriever(search_kwargs=search_kwargs)

    if use_hybrid:
        if selected_files:
            pool = [c for c in st.session_state.all_chunks if c.metadata.get("source_file") in selected_files]
            if not pool:
                bm25 = st.session_state.global_bm25
            else:
                bm25 = BM25Retriever.from_documents(pool)
        else:
            bm25 = st.session_state.global_bm25
            
        bm25.k = k

        base_retriever = EnsembleRetriever(
            retrievers=[semantic_retriever, bm25],
            weights=[semantic_weight, 1 - semantic_weight],
        )
    else:
        base_retriever = semantic_retriever

    if use_rerank:
        compressor = CohereRerank(
            model="rerank-v3.5", top_n=final_k, cohere_api_key=os.environ.get("COHERE_API_KEY")
        )
        return ContextualCompressionRetriever(
            base_compressor=compressor, base_retriever=base_retriever
        )

    return base_retriever

PROMPT = ChatPromptTemplate.from_template(
    """You are answering questions using ONLY the context below, which may come from
multiple different documents. Use the conversation history to resolve follow-up
questions (e.g. "what about the second one?") that refer back to earlier turns.

Only respond with "I don't know based on the provided documents." if the context
contains no relevant information at all to answer the question.

Conversation history:
{history}

Context:
{context}

Question: {question}

Answer:"""
)

def format_docs(docs):
    if not docs:
        return "No relevant context found."
    return "\n\n".join(
        f"[Source: {d.metadata.get('source_file')}, Page: {d.metadata.get('page_display')}]\n"
        f"{d.page_content}"
        for d in docs
    )

def format_history(messages, max_turns=4):
    recent = messages[-(max_turns * 2):]
    return "\n".join(f"{m['role'].capitalize()}: {m['content']}" for m in recent) or "(none)"

def ask(question):
    retriever = build_retriever()
    if retriever is None:
        return "Please upload and process at least one PDF first.", []

    start_time = time.time()
    docs = retriever.invoke(question)
    retrieval_time = time.time() - start_time
    print(f"\n--- TIMING REPORT ---")
    print(f"1. Document Retrieval took: {retrieval_time:.2f} seconds")

    context = format_docs(docs)
    history = format_history(st.session_state.messages)
    chain = PROMPT | llm
    
    start_time = time.time()
    response = chain.invoke({"context": context, "question": question, "history": history})
    llm_time = time.time() - start_time
    print(f"2. AI Text Generation took: {llm_time:.2f} seconds\n")
    
    return response.content, docs

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander("Sources"):
                for s in msg["sources"]:
                    st.caption(f"📄 {s['file']} — page {s['page']}")

question = st.chat_input("Ask a question about your uploaded PDFs...")

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            answer, docs = ask(question)
        st.markdown(answer)
        sources = []
        if docs:
            with st.expander("Sources"):
                for d in docs:
                    file_name = d.metadata.get("source_file", "unknown")
                    page = d.metadata.get("page_display", "?")
                    st.caption(f"📄 {file_name} — page {page}")
                    sources.append({"file": file_name, "page": page})

    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "sources": sources}
    )
