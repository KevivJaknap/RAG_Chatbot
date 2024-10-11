from nlp.faiss_model import FAISSModel
from nlp.qa_model import QA_Neo
import os 

root_path = r'C:\Users\vivek_pankaj\Desktop\DAP\dap_app_v1.0\src\files'
file_paths = os.listdir(root_path)
file_paths = [os.path.join(root_path, file_path) for file_path in file_paths]

qa = QA_Neo()
faiss_model = FAISSModel(file_paths=file_paths)

query = "What is the capital of France?"

context = faiss_model.get_best_match(query)
print(f"Context: {context}")

answer = qa.answer_question(context, query)

print(f"Answer: {answer}")