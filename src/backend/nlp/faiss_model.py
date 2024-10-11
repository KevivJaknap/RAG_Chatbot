from langchain.text_splitter import RecursiveCharacterTextSplitter, SentenceTransformersTokenTextSplitter
from PyPDF2 import PdfReader
import os
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel, AutoModelForQuestionAnswering
import faiss
import torch

class FAISSModel:
    def __init__(self, model_name='sentence-transformers/all-MiniLM-L6-v2', file_paths=[]):
        self.model = AutoModel.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.split_texts = []
        self.data_process(file_paths)
        if file_paths:
            self.build_index(self.split_texts)

    def splitter(self, text):
        character_splitter = RecursiveCharacterTextSplitter(
            separators=["\n\n", "\n", ".", " ", ""],
            chunk_size=400,
            chunk_overlap=100
        )
        sentence_splitter = SentenceTransformersTokenTextSplitter(
            tokens_per_chunk=256,
            chunk_overlap=50
        )

        character_split_texts = character_splitter.split_text(text)
        token_split_texts = []
        for character_split_text in character_split_texts:
            token_split_texts.extend(sentence_splitter.split_text(character_split_text))
        
        self.split_texts = token_split_texts
    
    def data_process(self, file_paths):
        total_text = '\n\n'.join([self.extract_pdf(file_path) for file_path in file_paths])
        self.splitter(total_text)

    def build_index(self, split_texts):
        embeddings = self.generate_embeddings(split_texts)
        self.index = faiss.IndexFlatIP(self.model.config.hidden_size)
        self.index.add(embeddings)
    
    def get_best_match(self, query, k=1):
        query_embeddings = self.generate_embeddings([query])
        D, I = self.index.search(query_embeddings, k)
        best_match = self.split_texts[I[0][0]]

        return best_match
   
    def get_k_best_matches(self, query, k=1):
        query_embeddings = self.generate_embeddings([query])
        D, I = self.index.search(query_embeddings, k)
        best_matches = [self.split_texts[i] for i in I[0]]
        return best_matches
    
    def generate_embeddings(self, chunks):
        inputs = self.tokenizer(chunks, padding=True, truncation=True, return_tensors='pt')
        with torch.no_grad():
            outputs = self.model(**inputs)
        embeddings = outputs.last_hidden_state.mean(dim=1)
        return embeddings.numpy()
    
    def check_pdf_exists(self, file_path: str) -> bool:
        '''
        Check if the file exists and is a pdf file
        '''
        if os.path.isfile(file_path) and file_path.lower().endswith('.pdf'):
            return True
        return False


    def extract_pdf(self, file_path: str) -> str | None:
        '''
        Extract text from a pdf file
        '''
        if self.check_pdf_exists(file_path):
            with open(file_path, 'rb') as file:
                pdf = PdfReader(file)
                text = ''
                for page in pdf.pages:
                    text += page.extract_text()
                return text
        return None


class FAISS_V2:
    def __init__(self, model_name='sentence-transformers/all-MiniLM-L6-v2'):
        self.model = AutoModel.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.split_texts = []
        self.indexed_files = []
        self.index = None

    
    def splitter(self, text):
        character_splitter = RecursiveCharacterTextSplitter(
            separators=["\n\n", "\n", ".", " ", ""],
            chunk_size=400,
            chunk_overlap=100
        )
        sentence_splitter = SentenceTransformersTokenTextSplitter(
            tokens_per_chunk=256,
            chunk_overlap=50
        )

        character_split_texts = character_splitter.split_text(text)
        token_split_texts = []
        for character_split_text in character_split_texts:
            token_split_texts.extend(sentence_splitter.split_text(character_split_text))
        
        return token_split_texts
    
    def data_process(self, file_path):
        total_text = self.extract_pdf(file_path)
        return self.splitter(total_text)
    
    def get_embeddings(self, split_texts):
        inputs = self.tokenizer(split_texts, padding=True, truncation=True, return_tensors='pt')
        with torch.no_grad():
            outputs = self.model(**inputs)
        embeddings = outputs.last_hidden_state.mean(dim=1)
        return embeddings.numpy()

    def get_k_best_matches(self, query, k=1):
        query_embeddings = self.generate_embeddings([query])
        D, I = self.index.search(query_embeddings, k)
        best_matches = [self.split_texts[i] for i in I[0]]
        return best_matches
    
    def generate_embeddings(self, chunks):
        inputs = self.tokenizer(chunks, padding=True, truncation=True, return_tensors='pt')
        with torch.no_grad():
            outputs = self.model(**inputs)
        embeddings = outputs.last_hidden_state.mean(dim=1)
        return embeddings.numpy()
    
    def check_pdf_exists(self, file_path: str) -> bool:
        '''
        Check if the file exists and is a pdf file
        '''
        if os.path.isfile(file_path) and file_path.lower().endswith('.pdf'):
            return True
        return False
    
    def add_new_path(self, file_path: str) -> None:
        '''
        Add a new file path to the existing list of file paths
        '''
        if file_path in self.indexed_files:
            return {"message": "File already indexed"}
        try:
            current_chunks = self.data_process(file_path)
            self.split_texts.extend(current_chunks)
            current_embeddings = self.get_embeddings(current_chunks)

            if self.index is not None:
                self.index.add(current_embeddings)
            else:
                self.index = faiss.IndexFlatIP(self.model.config.hidden_size)
                self.index.add(current_embeddings)
            
            self.indexed_files.append(file_path)
            return {"message": "File added to index"}
        except Exception as e:
            return {"error": str(e)}
        
    def extract_pdf(self, file_path: str) -> str | None:
        '''
        Extract text from a pdf file
        '''
        if self.check_pdf_exists(file_path):
            with open(file_path, 'rb') as file:
                pdf = PdfReader(file)
                text = ''
                for page in pdf.pages:
                    text += page.extract_text()
                return text
        return None