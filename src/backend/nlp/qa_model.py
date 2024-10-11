from transformers import pipeline
from transformers import GPTNeoForCausalLM, GPT2Tokenizer
class QA:
    def __init__(self, model_name="bert-large-uncased-whole-word-masking-finetuned-squad"):
        self.qa_pipeline = pipeline("question-answering", model=model_name)
    
    def answer_question(self, context: str, question: str) -> str:
        result = self.qa_pipeline(question=question, context=context)
        return result['answer']

class QA_Neo:
    def __init__(self):
        self.tokenizer = GPT2Tokenizer.from_pretrained("EleutherAI/gpt-neo-2.7B")
        self.model = GPTNeoForCausalLM.from_pretrained("EleutherAI/gpt-neo-2.7B")
    
    def answer_question(self, context: str, question: str) -> str:
        prompt = f'''
        Context: {context}
        Answer the following question using the context above.
        Question: {question}
        Answer:
        '''
        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids
        output = self.model.generate(input_ids, do_sample=True, max_length=400)
        return self.tokenizer.decode(output[0], skip_special_tokens=True)

from together import Together
import requests
import json
import os
from collections import Counter
from nlp.faiss_model import FAISS_V2

class RAG2:
    def __init__(self):
        self.index = FAISS_V2()

    def query_transformation(self, query):
        prompt = f"""
        Rephrase the question to better structure so that it returns better answers. Return two queries in json format. Do not use abbreviations or acronyms. Do not change the meaning of the question.
        At the start of the json, provide <JSON> and at the end provide </JSON>
        \n\n\n
        Question: {query}
        \n\n\n
        Answer:
        """
        system_msg = "You are a useful AI assistant"
        return self.arli_pipeline(user_input=prompt, system_msg=system_msg)
    
    def arli_pipeline(self, user_input: str, system_msg: str = "You are a useful AI assistant") -> str:
        url = "https://api.arliai.com/v1/completions"

        payload = json.dumps({
        "model": "Meta-Llama-3.1-8B-Instruct",
        "prompt": self.generate_prompt(user_input=user_input, system_msg=system_msg),
        "repetition_penalty": 1.1,
        "temperature": 0.7,
        "top_p": 0.9,
        "top_k": 40,
        "max_tokens": 1024,
        "stream": True
        })
        headers = {
        'Content-Type': 'application/json',
        'Authorization': f"Bearer {os.environ['ARLI_API_KEY']}"
        }

        response = requests.request("POST", url, headers=headers, data=payload)
        return self.process_arli_response(response.text)
    
    def process_arli_response(self, response: str) -> list:
        k = list(response.split('data: '))
        k = filter(lambda x: x != '', k)
        arr = []
        for obj in k:
            try:
                arr.append(json.loads(obj))
            except json.JSONDecodeError:
                continue
        return ''.join([obj['choices'][0]['text'] for obj in arr])
    
    def parse_query_transformation(self, response: str) -> dict:
        start_tag = "<JSON>"
        end_tag = "</JSON>"
        response = response.strip()
        start = response.find(start_tag)
        end = response.find(end_tag)
        if start == -1 or end == -1:
            return {}
        substring = response[start + len(start_tag):end].strip()
        return json.loads(substring)
    
    def statement_assumption(self, query: str) -> str:
        prompt = f"""
        Generate a hypothetical answer to the following query in one or two sentences.
        \n\n
        Query: {query}
        \n\n
        Answer:
        """
        system_msg = "You are a useful AI assistant"
        return self.arli_pipeline(user_input=prompt, system_msg=system_msg)
    
    def get_big_context(self, queries: list[str]) -> str:
        context_lst = []
        for query in queries:
            context_lst.extend(self.index.get_k_best_matches(query, k=1))

        counter = Counter(context_lst)
        ret_context = ''
        best_n_context = counter.most_common(5)
        for context, _ in best_n_context:
            ret_context += context + '\n\n'
        return ret_context
    
    def advanced_rag_pipeline(self, query: str) -> str:
        transformed_queries = self.query_transformation(query)
        queries = self.parse_query_transformation(transformed_queries)
        statement_assumption_query = self.statement_assumption(query)
        queries['query3'] = statement_assumption_query
        big_context = self.get_big_context(list(queries.values()))
        return self.together_rag_pipeline(query=query, context=big_context)
    
    def generate_prompt(self, user_input: str, system_msg: str) -> str:
        return f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{system_msg}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{user_input}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
    
    def together_rag_pipeline(self, query: str, context: str) -> str:
        client = Together()

        prompt = f"""
        Answer the following question based on the given context:
        {context}
        Question: {query}
        Answer:
        """
        completion = client.chat.completions.create(
            model="meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
            messages=[{"role": "system", "content": "You are a useful AI assistant"},
                    {"role": "user", "content": prompt}]
        )

        return completion.choices[0].message.content
    
    def response(self, query: str) -> str:
        return self.advanced_rag_pipeline(query)