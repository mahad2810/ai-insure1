from flask import Flask, request, jsonify
from dotenv import load_dotenv
import os
import requests
import json
from concurrent.futures import ThreadPoolExecutor

load_dotenv()
app = Flask(__name__)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
TEAM_TOKEN = "d1b791fa0ef5092d9cd051b2b09df2473d1e2ea07e09fe6c61abb5722dfbc7d3"

# Build Gemini prompt
def build_prompt(document_url, questions):
    formatted = "\n".join([f"{i+1}. {q}" for i, q in enumerate(questions)])
    return f"""You are an intelligent assistant trained for **exact factual extraction** from insurance policy documents. Your job is to locate precise answers **verbatim** from the document and respond **only if the answer is directly and explicitly written.**

### Objective:
Read the policy document provided and return **exact phrases, values, durations, and conditions** found in the text as answers to the following questions. Do not infer or interpret. Do not generalize or rephrase. Use the document's own words wherever applicable.

### Instructions:
- Return your output as **valid JSON** only.
- Structure:
  {{
    "answers": [
      "<answer to Q1, copied or precisely paraphrased>",
      "<answer to Q2>",
      ...
    ]
  }}
- If the answer is **not present, unclear, or cannot be found**, reply with: `"Not specified in the document."`
- Never infer, guess, or assume anything not explicitly stated.
- Always prefer copying the original wording from the document over rephrasing.
- Pay close attention to time durations (e.g., 24 vs. 36 months), monetary caps, exclusions, and benefit conditions.
- Be especially cautious about exclusion clauses, eligibility criteria, and named plans (like Plan A, Plan B, etc.).

---

### Document to Analyze:
{document_url}

### Questions:
{formatted}
"""

def call_gemini(prompt):
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-pro:generateContent?key={GEMINI_API_KEY}"
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "response_mime_type": "application/json",
            "temperature": 0.2
        }
    }
    response = requests.post(url, headers={"Content-Type": "application/json"}, json=payload)
    if response.status_code != 200:
        raise Exception(f"Gemini API error: {response.text}")
    response_text = response.json()["candidates"][0]["content"]["parts"][0]["text"]
    json_start = response_text.find("{")
    json_end = response_text.rfind("}")
    return json.loads(response_text[json_start:json_end+1])

@app.route("/api/v1/hackrx/run", methods=["POST"])
def run_submission():
    try:
        # Auth check
        auth_header = request.headers.get("Authorization", "")
        if not auth_header.startswith("Bearer ") or auth_header.split(" ")[1] != TEAM_TOKEN:
            return jsonify({"error": "Unauthorized"}), 401

        data = request.get_json()
        document_url = data.get("documents")
        questions = data.get("questions")

        if not document_url or not questions:
            return jsonify({"error": "Missing 'documents' or 'questions'"}), 400

        # Split questions into ordered chunks
        chunk_size = 3  # Process 3 questions at a time
        question_chunks = [questions[i:i + chunk_size] for i in range(0, len(questions), chunk_size)]

        results = []
        # Execute in order using futures mapped to index
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = {
                executor.submit(call_gemini, build_prompt(document_url, chunk)): idx
                for idx, chunk in enumerate(question_chunks)
            }

            # Prepare ordered answers array
            ordered_results = [None] * len(question_chunks)

            for future in futures:
                idx = futures[future]
                result = future.result()
                ordered_results[idx] = result["answers"]

        # Flatten in correct order
        answers = [ans for chunk in ordered_results for ans in chunk]

        return jsonify({"answers": answers}), 200

    except Exception as e:
        return jsonify({"error": "Server error", "details": str(e)}), 500

if __name__ == "__main__":
    app.run(port=8000, debug=True)
