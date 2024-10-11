import os
from PyPDF2 import PdfReader
import re

def check_pdf_exists(file_path: str) -> bool:
    '''
    Check if the file exists and is a pdf file
    '''
    if os.path.isfile(file_path) and file_path.lower().endswith('.pdf'):
        return True
    return False


def extract_pdf(file_path: str) -> str | None:
    '''
    Extract text from a pdf file
    '''
    if check_pdf_exists(file_path):
        with open(file_path, 'rb') as file:
            pdf = PdfReader(file)
            text = ''
            for page in pdf.pages:
                text += page.extract_text()
            return text
    return None

def clean_text(text: str) -> str:
    '''
    Clean the extracted text from a pdf document
    '''
    # Remove unwanted characters (e.g., non-alphanumeric except spaces)
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    
    # Convert text to lowercase
    text = text.lower()
    
    # Remove extra spaces and newlines
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text