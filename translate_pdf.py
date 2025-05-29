from fastapi import FastAPI, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
import fitz  # PyMuPDF
from openai import OpenAI
from fpdf import FPDF
import os

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

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

    completion = client.chat.completions.create(
        model="gpt-4",
        messages=messages,
        temperature=0.2
    )

    translated = completion.choices[0].message.content

    # Generate new PDF
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.multi_cell(0, 10, translated)

    translated_pdf_path = "/tmp/translated.pdf"
    pdf.output(translated_pdf_path)

    return FileResponse(path=translated_pdf_path, filename="translated.pdf")