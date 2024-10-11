from fastapi import FastAPI
from nlp.faiss_model import FAISS_V2
from nlp.qa_model import QA
from pydantic import BaseModel
from nlp.qa_model import RAG2
import os 

root_path = r'C:\Users\vivek_pankaj\Desktop\DAP\dap_app_v1.0\src\files'
file_paths = os.listdir(root_path)
file_paths = [os.path.join(root_path, file_path) for file_path in file_paths]

app = FastAPI()
rag = RAG2()

class QARequest(BaseModel):
    question: str


@app.get("/")
def root():
    return {"message": "Hello World"}

@app.post("/add")
def add_file_path(file_path: str):
    return rag.index.add_new_path(file_path)

@app.post("/qa")
def qa(request: QARequest):
    return {"answer": rag.response(request.question)}

@app.get("/delete")
def delete_index():
    pass

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
    



