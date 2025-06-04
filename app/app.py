import os
import time

import streamlit as st
from dotenv import load_dotenv

# ──────────────────────────────────────────────────────────────────────────────
# 0) Set Streamlit page configuration (MUST be first Streamlit command)
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Phama Punjabi Chat (FAISS Load)",
    layout="centered"
)

# ──────────────────────────────────────────────────────────────────────────────
# 1) Load environment variables (for Mistral API key)
# ──────────────────────────────────────────────────────────────────────────────
load_dotenv()  # looks for a .env file in this folder or parent

MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY", None)
if not MISTRAL_API_KEY:
    st.warning(
        "⚠️ MISTRAL_API_KEY not found in .env. "
        "Make sure to set it if required by LangChain-Mistral."
    )

# ──────────────────────────────────────────────────────────────────────────────
# 2) Load a pre-built FAISS index from disk (cached to run only once)
# ──────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading FAISS index from disk…")
def load_vectorstore():
    from langchain.embeddings import HuggingFaceEmbeddings
    from langchain.vectorstores import FAISS

    # 2.1) Same embedding model used during indexing, force CUDA if available
    EMBED_MODEL = "l3cube-pune/punjabi-sentence-similarity-sbert"
    model_kwargs = {"device": "cuda"} if os.getenv("CUDA_VISIBLE_DEVICES", None) or True else {}
    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL, model_kwargs=model_kwargs)

    # 2.2) Path to the saved FAISS index folder
    INDEX_DIR = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "faiss_index", "phama_faiss")
    )

    if not os.path.isdir(INDEX_DIR):
        st.error(
            "FAISS index not found on disk! Please run `python build_faiss_index.py` first "
            "to generate the index in 'faiss_index/phama_faiss/'."
        )
        return None

    # 2.3) Load the persisted FAISS index, allowing pickle deserialization
    vectorstore = FAISS.load_local(
        INDEX_DIR, embeddings, allow_dangerous_deserialization=True
    )
    return vectorstore

vectorstore = load_vectorstore()
if vectorstore is None:
    st.stop()

# ──────────────────────────────────────────────────────────────────────────────
# 3) Initialize the ChatMistralAI LLM and prompt template (cached once)
# ──────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading prompt template & LLM…")
def init_llm_and_prompt():
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_mistralai import ChatMistralAI

    # 3.1) Punjabi “system + human” prompt
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """ਤੁਸੀਂ ਕਾਲ ਸੈਂਟਰ ਵਿੱਚ ਇੱਕ ਕਰਮਚਾਰੀ ਹੋ।
                ਆਪਣੇ ਖੁਦ ਦੇ ਗਿਆਨ ਅਤੇ ਹੇਠਾਂ ਦਿੱਤੇ ਤਿੰਨ ਸੰਦਰਭਾਂ ਦੀ ਵਰਤੋਂ ਕਰਕੇ
                ਸਵਾਲ ਦਾ ਇੱਕ ਹੀ ਸੰਯੁਕਤ ਜਵਾਬ ਪ੍ਰਦਾਨ ਕਰੋ।
                ਸੰਦਰਭ:
                {context}
                """,
            ),
            ("human", "{question}"),
        ]
    )

    # 3.2) Use a valid Mistral model name (e.g., "mistral-large-latest")
    llm = ChatMistralAI(
        model="mistral-large-latest",
        temperature=0
    )

    return prompt, llm

prompt_obj, llm_obj = init_llm_and_prompt()

# ──────────────────────────────────────────────────────────────────────────────
# 4) Helper: “embed → retrieve → LLM → answer”
# ──────────────────────────────────────────────────────────────────────────────
def get_answer(question: str) -> str:
    from langchain_core.runnables import RunnableMap
    from langchain_core.output_parsers import StrOutputParser

    # 4.1) Retrieve top-3 chunks from FAISS
    docs = vectorstore.similarity_search(question, k=3)

    # 4.2) Format them as:
    #     “ਸੰਦਰਭ 1: <chunk_text>\n\n
    #      ਸੰਦਰਭ 2: <chunk_text>\n\n
    #      ਸੰਦਰਭ 3: <chunk_text>”
    formatted_contexts = "\n\n".join(
        [f"ਸੰਦਰਭ {i+1} ({d.metadata['source']}): {d.page_content}" for i, d in enumerate(docs)]
    )

    # 4.3) Build a RunnableMap to feed “context” & “question” into prompt → LLM
    rag_chain = (
        RunnableMap(
            {
                "context": lambda _: formatted_contexts,
                "question": lambda _: question,
            }
        )
        | prompt_obj
        | llm_obj
        | StrOutputParser()
    )

    # 4.4) Invoke synchronously and return the answer
    answer = rag_chain.invoke({})
    return answer

# ──────────────────────────────────────────────────────────────────────────────
# 5) Streamlit UI layout
# ──────────────────────────────────────────────────────────────────────────────
st.title("Punjabi AgroBot")
st.write(
    """
    ਆਪਣਾ ਪੰਜਾਬੀ ਸਵਾਲ ਦਿਓ, ਅਤੇ ਇਹ ਐਪ ਤਿੰਨਾਂ ਸਭ ਤੋਂ ਢੁਕਵੇਂ ਚੰਕਾਂ ਨੂੰ ਲੱਭ ਕੇ 
    ਇੱਕ ਸੰਯੁਕਤ ਜਵਾਬ ਵਾਪਸ ਕਰੇਗਾ (`Mistral-7B` ਵਰਤ ਕੇ)।
    """
)

# 5.1) Text area for user to type a Punjabi question
user_question = st.text_area(
    "ਆਪਣਾ ਪੰਜਾਬੀ ਸਵਾਲ ਇੱਥੇ ਲਿਖੋ:",
    height=150,
    placeholder="例: ਮੈਂ ਬੀਜ ਬਾਰੇ ਜਾਣਕਾਰੀ ਚਾਹੁੰਦਾ ਹਾਂ…",
)

# 5.2) “Get Answer” button
if st.button("ਜਵਾਬ ਪ੍ਰਾਪਤ ਕਰੋ") and user_question.strip():
    with st.spinner("ਸੋਚ ਰਹੇ ਹਾਂ…"):
        start_time = time.time()
        try:
            answer = get_answer(user_question)
        except Exception as e:
            answer = f"🚨 ਕੋਈ ਗਲਤੀ ਆਈ: {e}"
        elapsed = time.time() - start_time

    # 5.3) Show the final answer
    st.subheader("ਮਾਡਲ ਦਾ ਜਵਾਬ:")
    st.write(answer)
    st.caption(f"Response time: {elapsed:.2f}s")

# 5.4) Sidebar info
with st.sidebar:
    st.header("ਇਹ ਕੀ ਹੈ?")
    st.write(
        "ਇਹ Streamlit ਐਪ ਪਹਿਲਾਂ ਬਣਾਈ ਗਈ FAISS ਇੰਡੈਕਸ ਨੂੰ ਲੋਡ ਕਰਦਾ ਹੈ।\n\n"
        "1. **ਇੰਡੈਕਸ** ਫਾਇਲ ਤੋਂ ਤੁਰੰਤ FAISS ਵੇਕਟਰਸਟੋਰ ਲੋਡ ਹੋ ਜਾਂਦਾ ਹੈ।\n"
        "2. **Punjabi SBERT** ਵਰਤ ਕੇ ਤੁਹਾਡੇ ਸਵਾਲ ਨੂੰ embed ਕਰਦਾ ਹੈ (CUDA ਸਮਰਥਨ)।\n"
        "3. FAISS ਵਿੱਚੋਂ ਟਾਪ-3 ਚੰਕ ਲਿਆਉਂਦਾ ਹੈ।\n"
        "4. ਉਹ ਚੰਕ **ਅਤੇ** ਤੁਹਾਡਾ ਸਵਾਲ `Mistral-7B` ਨੂੰ ਭੇਜ ਕੇ ਜਵਾਬ ਲੈਂਦਾ ਹੈ।\n\n"
        "ਕਦੇ ਵੀ FAISS ਇੰਡੈਕਸ ਨਹੀਂ ਬਣਾਉਂਦਾ—ਸਿਰਫ਼ `build_faiss_index.py` ਚਲਾਉਣ ਨਾਲ ਪਹਿਲਾਂ ਹੀ ਇੰਡੈਕਸ ਬਣ ਚੁੱਕਾ ਹੁੰਦਾ ਹੈ।"
    )
    st.markdown("[GitHub: Punjabi Agrobot](https://github.com/WakeUpSidd/Punjabi-AgroBot)")
