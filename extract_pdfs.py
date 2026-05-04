import PyPDF2
import sys

def extract_pdf(pdf_path, txt_path):
    with open(pdf_path, 'rb') as f:
        reader = PyPDF2.PdfReader(f)
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"
    
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write(text)

extract_pdf("Second Hand Fashion Advisor PRD.pdf", "prd.txt")
extract_pdf("Second Hand Fashion Advisor DESIGN.pdf", "design.txt")
print("Extraction complete")
