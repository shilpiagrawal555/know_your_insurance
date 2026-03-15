import os
import json
import pdfplumber
import anthropic
from flask import Flask, request, jsonify, render_template, stream_with_context, Response
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024  # 50MB max upload

ALLOWED_EXTENSIONS = {"pdf"}


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def extract_pdf_text(file_stream):
    """Extract raw text from PDF using pdfplumber. Returns list of (page_num, text) tuples."""
    pages = []
    with pdfplumber.open(file_stream) as pdf:
        for i, page in enumerate(pdf.pages, 1):
            text = page.extract_text()
            if text and text.strip():
                pages.append((i, text.strip()))
    return pages


def build_extraction_prompt(raw_text: str) -> str:
    return f"""You are a precise insurance document analyst. Your ONLY job is to extract information that is EXPLICITLY stated in the document text provided below.

CRITICAL RULES — you MUST follow these without exception:
1. NEVER infer, guess, assume, or extrapolate any information
2. NEVER fill in fields with "typical" or "standard" values
3. If a piece of information is NOT explicitly written in the document, set its value to null
4. Copy values EXACTLY as they appear in the document — do not rephrase, summarize, or interpret
5. For dollar amounts, copy the exact figure including the $ sign and any formatting
6. For dates, copy the exact date format used in the document
7. Only extract what you can directly quote from the text

Return a single JSON object with this exact structure. Use null for any field not found:

{{
  "policy_holder": {{
    "name": "exact name as written or null",
    "address": "exact address as written or null",
    "date_of_birth": "exact DOB as written or null",
    "member_id": "exact member/subscriber ID as written or null",
    "group_number": "exact group number as written or null"
  }},
  "insurer": {{
    "company_name": "exact company name as written or null",
    "phone": "exact phone number as written or null",
    "website": "exact website as written or null",
    "address": "exact address as written or null"
  }},
  "policy": {{
    "policy_number": "exact policy number as written or null",
    "policy_type": "exact policy type as written (e.g. Individual Health, Family, Auto, Home) or null",
    "effective_date": "exact start date as written or null",
    "expiration_date": "exact end date as written or null",
    "premium_amount": "exact premium amount and frequency as written or null",
    "payment_due_date": "exact due date as written or null"
  }},
  "coverage": {{
    "deductible_individual": "exact individual deductible as written or null",
    "deductible_family": "exact family deductible as written or null",
    "out_of_pocket_max_individual": "exact individual OOP max as written or null",
    "out_of_pocket_max_family": "exact family OOP max as written or null",
    "lifetime_maximum": "exact lifetime max as written or null",
    "network": "exact network name/type as written or null"
  }},
  "cost_sharing": {{
    "primary_care_copay": "exact copay as written or null",
    "specialist_copay": "exact specialist copay as written or null",
    "emergency_room_copay": "exact ER copay as written or null",
    "urgent_care_copay": "exact urgent care copay as written or null",
    "coinsurance": "exact coinsurance percentage as written or null",
    "prescription_tier1": "exact tier 1 Rx cost as written or null",
    "prescription_tier2": "exact tier 2 Rx cost as written or null",
    "prescription_tier3": "exact tier 3 Rx cost as written or null"
  }},
  "covered_benefits": [
    "list each covered benefit EXACTLY as stated in the document, or empty array if none found"
  ],
  "exclusions": [
    "list each exclusion EXACTLY as stated in the document, or empty array if none found"
  ],
  "pre_authorization": [
    "list each service requiring pre-authorization EXACTLY as stated, or empty array if none found"
  ],
  "additional_notes": [
    "list any other important policy details EXACTLY as stated that don't fit above categories, or empty array"
  ]
}}

DOCUMENT TEXT:
---
{raw_text}
---

Return ONLY the JSON object. No explanation, no markdown, no code blocks."""


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/parse", methods=["POST"])
def parse_policy():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    if not allowed_file(file.filename):
        return jsonify({"error": "Only PDF files are supported"}), 400

    # Extract raw text from PDF
    try:
        pages = extract_pdf_text(file.stream)
    except Exception as e:
        return jsonify({"error": f"Could not read PDF: {str(e)}"}), 400

    if not pages:
        return jsonify({"error": "No readable text found in PDF. The file may be scanned or image-based."}), 400

    # Combine all pages with page markers
    full_text = ""
    for page_num, text in pages:
        full_text += f"\n[Page {page_num}]\n{text}\n"

    total_chars = len(full_text)
    # Truncate if extremely long (>200k chars) to stay within context limits
    if total_chars > 200000:
        full_text = full_text[:200000]
        truncated = True
    else:
        truncated = False

    def generate():
        try:
            client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
            prompt = build_extraction_prompt(full_text)

            # Stream the response
            with client.messages.stream(
                model="claude-opus-4-6",
                max_tokens=4096,
                thinking={"type": "adaptive"},
                system=(
                    "You are a document extraction system. Extract ONLY information explicitly present in the document. "
                    "Never hallucinate, infer, or add information not directly stated. "
                    "Return valid JSON only."
                ),
                messages=[{"role": "user", "content": prompt}],
            ) as stream:
                full_response = ""
                for text in stream.text_stream:
                    full_response += text

            # Parse and validate the JSON
            # Strip any potential markdown code blocks if Claude adds them
            cleaned = full_response.strip()
            if cleaned.startswith("```"):
                lines = cleaned.split("\n")
                # Remove first and last lines if they are code fence markers
                if lines[0].startswith("```"):
                    lines = lines[1:]
                if lines and lines[-1].strip() == "```":
                    lines = lines[:-1]
                cleaned = "\n".join(lines)

            parsed = json.loads(cleaned)

            result = {
                "success": True,
                "data": parsed,
                "meta": {
                    "pages_extracted": len(pages),
                    "total_characters": total_chars,
                    "truncated": truncated,
                },
            }
            yield f"data: {json.dumps(result)}\n\n"

        except json.JSONDecodeError as e:
            yield f"data: {json.dumps({'success': False, 'error': 'Failed to parse structured data from document. The document may not be a standard insurance policy.'})}\n\n"
        except anthropic.AuthenticationError:
            yield f"data: {json.dumps({'success': False, 'error': 'Invalid API key. Please check your ANTHROPIC_API_KEY.'})}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'success': False, 'error': str(e)})}\n\n"

    return Response(
        stream_with_context(generate()),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


if __name__ == "__main__":
    app.run(debug=True, port=5000)
