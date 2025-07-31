from flask import Flask, request, jsonify
from dotenv import load_dotenv
import os
import requests
import json
import fitz  # PyMuPDF
import faiss
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
from concurrent.futures import ThreadPoolExecutor

load_dotenv()
app = Flask(__name__)

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
TEAM_TOKEN = "d1b791fa0ef5092d9cd051b2b09df2473d1e2ea07e09fe6c61abb5722dfbc7d3"

# ---------------- PDF Text Extraction ----------------
def extract_text_from_pdf_url(url):
    response = requests.get(url)
    doc = fitz.open(stream=response.content, filetype="pdf")
    return "\n".join(page.get_text() for page in doc)

# ---------------- Chunking & Embedding ----------------
model = SentenceTransformer("all-MiniLM-L6-v2")

def build_index(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_text(text)
    embeddings = model.encode(chunks)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    return index, chunks, embeddings

def retrieve_top_chunks(question, index, chunks, top_k=3):
    q_vector = model.encode([question])
    _, I = index.search(q_vector, top_k)
    return [chunks[i] for i in I[0]]

# ---------------- Groq Prompt ----------------
def build_prompt(question, context_chunks):
    context = "\n".join(context_chunks)
    return f"""
You are an expert insurance assistant designed to extract accurate, concise, and policy-compliant answers from the provided document.

Use only the factual information available in the context. Your response must reflect exact policy terms, durations, conditions, and legal phrasing wherever applicable.

### Instructions:
- Do not infer or assume anything.
- Answer **only** if the information is explicitly mentioned.
- Mention durations, limits, and conditions exactly as written.
- Use formal, policy-style language.
- If unsure or not found in the context, respond with: "Not specified in the document."

---

### 📘 Few-shot Examples:

**Context:**  
A grace period of thirty days shall be allowed for payment of each renewal premium.

**Question:**  
What is the grace period for premium payment?  

**Response:**  
{{ "answer": "A grace period of thirty days is allowed for premium payment of each renewal." }}

---

**Context:**  
Pre-existing diseases are covered after a continuous coverage of thirty-six (36) months from the policy inception date.

**Question:**  
When will pre-existing diseases be covered?

**Response:**  
{{ "answer": "There is a waiting period of thirty-six (36) months of continuous coverage from the policy inception for pre-existing diseases to be covered." }}

---

**Context:**  
If the information is not provided here, you must respond accordingly.

**Question:**  
What is the coverage amount for robotic surgery?

**Response:**  
{{ "answer": "Not specified in the document." }}

---

### 🔍 Actual Context:
{context}

### ❓Question:
{question}

### 💬 Respond in this exact JSON format:
{{ "answer": "..." }}
"""

def call_groq(prompt):
    url = "https://api.groq.com/openai/v1/chat/completions"
    payload = {
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "model": "meta-llama/llama-4-scout-17b-16e-instruct",
        "temperature": 0.2,
        "max_tokens": 1024,
        "top_p": 1,
        "stream": False
    }

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {GROQ_API_KEY}"
    }

    response = requests.post(url, headers=headers, json=payload)

    if response.status_code != 200:
        raise Exception(f"Groq API error: {response.text}")

    response_text = response.json()["choices"][0]["message"]["content"]
    json_start = response_text.find("{")
    json_end = response_text.rfind("}")
    parsed = json.loads(response_text[json_start:json_end+1])
    return parsed.get("answer", "Not specified in the document.")

# ---------------- Main Endpoint ----------------
@app.route("/api/v1/hackrx/run", methods=["POST"])
def run_submission():
    try:
        auth_header = request.headers.get("Authorization", "")
        if not auth_header.startswith("Bearer ") or auth_header.split(" ")[1] != TEAM_TOKEN:
            return jsonify({"error": "Unauthorized"}), 401

        data = request.get_json()
        document_url = data.get("documents")
        questions = data.get("questions")

        if not document_url or not questions:
            return jsonify({"error": "Missing 'documents' or 'questions'"}), 400

        # Step 1: Extract and preprocess
        text = extract_text_from_pdf_url(document_url)
        index, chunks, _ = build_index(text)

        # Step 2: Process each question in parallel
        def process_question(q):
            top_chunks = retrieve_top_chunks(q, index, chunks)
            prompt = build_prompt(q, top_chunks)
            return call_groq(prompt)

        with ThreadPoolExecutor(max_workers=5) as executor:
            answers = list(executor.map(process_question, questions))

        return jsonify({"answers": answers}), 200

    except Exception as e:
        return jsonify({"error": "Server error", "details": str(e)}), 500

# ---------------- Main ----------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)
