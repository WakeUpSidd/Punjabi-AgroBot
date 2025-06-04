# Punjabi-AgroBot 🌾

**Punjabi-AgroBot** is a Punjabi-language agricultural chatbot that answers user questions by retrieving context from Punjabi text documents using FAISS and generating a combined response using the Mistral-7B language model.

---

## 🗂️ Project Structure

```
Punjabi-AgroBot/
├── app/
│   └── app.py               ← Streamlit frontend for interaction
├── build_faiss_index.py     ← Script to create and save FAISS index from text files
├── requirements.txt         ← Python dependencies
├── .gitignore
```

---

## ⚙️ Setup Instructions

1. **Clone the repository**

```bash
git clone https://github.com/YourUsername/Punjabi-AgroBot.git
cd Punjabi-AgroBot
```

2. **Install dependencies**

```bash
pip install -r requirements.txt
```

3. **Set your environment variable**

Create a `.env` file in the root folder and add your Mistral API key:

```
MISTRAL_API_KEY=your_mistral_api_key_here
```

4. **Prepare text files**

Place `.txt` files containing Punjabi agricultural information into the `vectorDb/` folder.

5. **Build the FAISS Index**

```bash
python build_faiss_index.py
```

This script will:
- Load the SBERT model: `l3cube-pune/punjabi-sentence-similarity-sbert`
- Read all `.txt` files from the `vectorDb` folder
- Chunk the content into 1000-character chunks with 200-character overlaps
- Create a FAISS index and save it to `faiss_index/phama_faiss/`

6. **Run the chatbot interface**

```bash
streamlit run app/app.py
```

---

## 🚀 Step-by-Step Workflow

### `build_faiss_index.py`

1. Load the **Punjabi SBERT embedding model** with CUDA support.
2. Read `.txt` files from `vectorDb/`.
3. Chunk each file into overlapping segments.
4. Store each chunk in a LangChain `Document` with metadata.
5. Create a FAISS vectorstore from all chunks.
6. Save the vectorstore to `faiss_index/phama_faiss/`.

### `app/app.py`

1. Load environment variables using `dotenv`.
2. Load the FAISS index and embedding model (cached for performance).
3. Define a Punjabi prompt using LangChain's `ChatPromptTemplate`.
4. Load the Mistral model (`mistral-large-latest`) for inference.
5. On user input:
   - Embed the query
   - Perform FAISS similarity search (top-3 chunks)
   - Format results as context
   - Generate a response using the LLM
6. Display the final Punjabi answer using Streamlit.

---

## ❗ Note

- The chatbot only uses the **pre-built index**.
- If you add or update `.txt` files, rerun `build_faiss_index.py` to rebuild the index.
