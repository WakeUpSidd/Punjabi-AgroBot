import os
import glob
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain.schema import Document

# 1) Configuration
EMBED_MODEL = "l3cube-pune/punjabi-sentence-similarity-sbert"
VECTORSTORE_DIR = "faiss_index"  # folder where we will save FAISS files
VECTORSTORE_NAME = "phama_faiss" # you can pick any name

# 2) Load SBERT embeddings
embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL, model_kwargs={"device": "cuda"})

# 3) Read & chunk all .txt files under vectorDb/
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "vectorDb"))
pattern = os.path.join(base_dir, "*.txt")
filepaths = glob.glob(pattern)

if not filepaths:
    raise RuntimeError(f"No .txt files found under {base_dir}")

# Use a CharacterTextSplitter to chunk large files (~1,000 chars per chunk)
splitter = CharacterTextSplitter(separator="\n", chunk_size=1000, chunk_overlap=200)

docs = []
for fp in filepaths:
    with open(fp, "r", encoding="utf-8") as f:
        text = f.read().strip()
    if not text:
        continue

    chunks = splitter.split_text(text)
    for idx, chunk in enumerate(chunks):
        doc_id = os.path.basename(fp)
        chunk_id = f"{doc_id}__chunk_{idx+1}"
        docs.append(Document(page_content=chunk, metadata={"source": chunk_id}))

# 4) Build FAISS index from these Document chunks
print(f"Building FAISS index from {len(docs)} chunks ...")
vectorstore = FAISS.from_documents(docs, embeddings)
print("Done building index.")

# 5) Create the output folder if it doesn’t exist
os.makedirs(VECTORSTORE_DIR, exist_ok=True)

# 6) Save the FAISS index to disk
print(f"Saving FAISS index to {VECTORSTORE_DIR}/{VECTORSTORE_NAME} ...")
vectorstore.save_local(os.path.join(VECTORSTORE_DIR, VECTORSTORE_NAME))
print("Index saved successfully.")
