from fastapi import FastAPI, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
import fitz  # PyMuPDF
import openai
import os

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

openai.api_key = os.getenv("OPENAI_API_KEY")

@app.post("/translate-pdf/")
async def translate_pdf(file: UploadFile, sourceLang: str = Form(...), targetLang: str = Form(...)):
    # Save uploaded file
    file_location = f"/tmp/{file.filename}"
    with open(file_location, "wb") as f:
        f.write(await file.read())

    # Extract and normalize text from PDF
    doc = fitz.open(file_location)
    full_text = ""
    for page in doc:
        text = page.get_text()
        cleaned = text.replace("\n", " ").replace("  ", " ").strip()
        full_text += cleaned + "\n\n"

    # Translate via GPT
    messages = [
        {"role": "system", "content": f"Prevedi iz {sourceLang} v {targetLang}."},
        {"role": "user", "content": full_text[:12000]}  # limit due to token size
    ]

    completion = openai.ChatCompletion.create(
        model="gpt-4",
        messages=messages,
        temperature=0.2
    )

    translated = completion.choices[0].message["content"]
    return {"translated_text": translated}
