import os
import io
import time
from typing import List
import streamlit as st
import fitz  # PyMuPDF
from sentence_transformers import SentenceTransformer
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA, LLMChain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

# Load .env variables
load_dotenv()


GROQ_SUPPORTED = False
try:
    from langchain.chat_models import Groq  
    GROQ_SUPPORTED = True
except Exception:
    GROQ_SUPPORTED = False


try:
    from langchain.chat_models.openai import ChatOpenAI  
    OPENAI_WRAPPER_AVAILABLE = True
except Exception:
    OPENAI_WRAPPER_AVAILABLE = False


st.set_page_config(page_title="StudyMate AI", layout="wide")

# ---------- SIDEBAR UI
st.sidebar.image("https://r2.erweima.ai/i/9fE6iF-nSo6FO6kHFoD4Lw.png", width=150)
with st.sidebar:
    st.markdown("## StudyMate AI")
    st.caption("Your intelligent learning companion")

    
    st.markdown(
    """
    <div style="background-color: #f1f1f1; padding: 1rem; border-radius: 10px; max-width: 400px;">
        <p style="margin: 0; font-weight: bold;">Upload Documents</p>
    </div>
    """,
    unsafe_allow_html=True
)



# ---------- MAIN CONTENT UI
col_main, col_empty = st.columns([3, 1])

with col_main:
    
    st.markdown(
        """
        <div style="background-color: #f9f9f9; padding: 2rem; border-radius: 1rem; text-align: center;">
            <img src="https://r2.erweima.ai/i/9fE6iF-nSo6FO6kHFoD4Lw.png" 
                 alt="StudyMate" style="width: 300px; margin-bottom: 1rem;">
            <h2>Welcome to <span style="color:#5B2C6F;">StudyMate AI</span></h2>
            <p>Your intelligent learning companion powered by advanced AI. 
            Upload your study materials, ask questions, and get personalized explanations 
            tailored to your learning style.</p>
        </div>
        """,
        unsafe_allow_html=True
    )
uploaded_file = st.file_uploader(
    "Upload your study files",
    type=["pdf", "docx", "txt"],
    label_visibility="collapsed"  
)
    
st.markdown("---")
st.markdown("### Recent Conversations")
recent_chats = ["Quantum Physics Basics", "Linear Algebra Study Plan"]
for chat in recent_chats:
        st.button(chat, use_container_width=True)

st.markdown("---")
st.button("➕ New Conversation", use_container_width=True)

EMBED_MODELS = {
    "all-MiniLM-L6-v2": "all-MiniLM-L6-v2",
    "all-mpnet-base-v2": "all-mpnet-base-v2",
}
embed_choice = st.sidebar.selectbox("Sentence-transformer model", list(EMBED_MODELS.keys()), index=0)
EMBED_MODEL_NAME = EMBED_MODELS[embed_choice]


GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
if not GROQ_API_KEY:
    groq_input = st.sidebar.text_input("Groq API Key (or set in .env)", type="password")
    if groq_input:
        GROQ_API_KEY = groq_input
        os.environ["GROQ_API_KEY"] = groq_input

st.sidebar.markdown("---")
st.sidebar.markdown("Chunking settings")
chunk_size = st.sidebar.number_input("Chunk size (chars)", value=1000, min_value=200, max_value=5000, step=100)
chunk_overlap = st.sidebar.number_input("Chunk overlap (chars)", value=200, min_value=0, max_value=1000, step=50)


if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
    
if "chat_history" not in st.session_state:
    
    st.session_state.chat_history = []


SYSTEM_PROMPT = (
    "You are a helpful assistant that answers questions using the provided document context. "
    "If the context is insufficient, say you don't know and ask a clarifying question."
)

if "docs" not in st.session_state:
    st.session_state.docs = None

@st.cache_resource(show_spinner=False)
def init_hf_embeddings(model_name: str):
    return HuggingFaceEmbeddings(model_name=model_name)

def pdf_to_text(pdf_bytes: bytes) -> str:
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    pages = [p.get_text("text") for p in doc]
    return "\n\n".join(pages)

def chunk_text(text: str, chunk_size: int, chunk_overlap: int) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    pieces = splitter.split_text(text)
    return [Document(page_content=p, metadata={"chunk": i}) for i, p in enumerate(pieces)]

def build_inmemory_chroma(docs: List[Document], embeddings: HuggingFaceEmbeddings) -> Chroma:
    return Chroma.from_documents(documents=docs, embedding=embeddings, collection_name="memory_collection")
def trim_history(history: list, max_turns: int = 12):
    """
    Keep only the last `max_turns` user+assistant turns to limit prompt size.
    """
    return history[-max_turns:]

def history_to_text(history: list) -> str:
    """
    Convert chat_history list into a single textual block for prompt injection.
    Expects items with keys 'role' and 'content'.
    """
    return "\n".join(f"{h['role'].capitalize()}: {h['content']}" for h in history)


from langchain_groq import ChatGroq

def get_llm(model_name="gemma2-9b-it", temperature=0.2, max_tokens=512):
    if not os.environ.get("GROQ_API_KEY"):
        st.error("No GROQ_API_KEY set. Add it in .env or sidebar.")
        st.stop()

    llm = ChatGroq(
        groq_api_key=os.environ["GROQ_API_KEY"],
        model_name=model_name,
        temperature=temperature,
        max_tokens=max_tokens
    )
    return llm

if uploaded_file:
    pdf_bytes = uploaded_file.read()
    text = pdf_to_text(pdf_bytes)
    st.success(f"Extracted {len(text)} characters from PDF.")
    st.write(f"Preview (first 500 chars): {text[:500]}...")

    if st.button("Create / Rebuild in-memory index"):
        if text.strip():
            docs = chunk_text(text, chunk_size, chunk_overlap)
            embeddings = init_hf_embeddings(EMBED_MODEL_NAME)
            st.session_state.vectorstore = build_inmemory_chroma(docs, embeddings)
            st.session_state.docs = docs
            st.success(f"In-memory index created with {len(docs)} chunks.")
        else:
            st.error("No text extracted from PDF.")



if st.session_state.vectorstore:
    
    col_left, col_right = st.columns([1, 3])
    with col_left:
        max_history_turns = st.number_input("Max turns to keep", min_value=2, max_value=50, value=12, step=1)
        top_k = st.slider("Top-k chunks (retriever)", 1, 10, 4)
        temperature = st.slider("Model temperature", 0.0, 1.0, 0.2)
        if st.button("Clear chat"):
            st.session_state.chat_history = []
            st.experimental_rerun()
    with col_right:
        st.markdown("Ask follow-up questions — the chat maintains context across turns.")

    
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    
    user_input = st.chat_input("Ask anything about the document...")
    if user_input:
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        
        retriever = st.session_state.vectorstore.as_retriever(search_kwargs={"k": top_k})
        docs = retriever.get_relevant_documents(user_input)
        context_text = "\n\n".join(d.page_content for d in docs) if docs else ""

        
        trimmed = trim_history(st.session_state.chat_history, max_turns=max_history_turns)
        history_text = history_to_text(trimmed)

        
        prompt_template = PromptTemplate(
            input_variables=["system", "history", "context", "user_input"],
            template=(
                "{system}\n\n"
                "Conversation so far:\n{history}\n\n"
                "Relevant document context:\n{context}\n\n"
                "User: {user_input}\n"
                "Assistant:"
            )
        )

        
        llm = get_llm(temperature=temperature)
        chain = LLMChain(llm=llm, prompt=prompt_template)

        
        try:
            assistant_response = chain.run({
                "system": SYSTEM_PROMPT,
                "history": history_text,
                "context": context_text,
                "user_input": user_input
            })
        except Exception as e:
            assistant_response = f"Error calling LLM: {e}"

        st.session_state.chat_history.append({"role": "assistant", "content": assistant_response})
        with st.chat_message("assistant"):
            st.markdown(assistant_response)
else:
    st.info("Upload a PDF and create an index first.")
