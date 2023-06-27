import openai
import numpy as np
import tiktoken

import os
from dotenv import load_dotenv

# .env.local 파일 로드
load_dotenv('.env.local')

# 환경 변수 값 가져오기
openai.api_key  = os.getenv('OPENAI_API_KEY')

print (openai.api_key)

COMPLETIONS_API_PARAMS = {
    "temperature": 0,
    "max_tokens": 300,
    "model": "text-davinci-003",
}
SEPARATOR = "\n* "
ENCODING = "cl100k_base"  # encoding for text-embedding-ada-002
MAX_SECTION_LEN = 500
encoding = tiktoken.get_encoding(ENCODING)
separator_len = len(encoding.encode(SEPARATOR))

def get_embedding(text: str, model:str= "text-embedding-ada-002"):
    result = openai.Embedding.create(
      model=model,
      input=text
    )
    return result["data"][0]["embedding"]

def vector_similarity(x,y):
    return np.dot(np.array(x), np.array(y))

def order_document_sections_by_query_similarity(query: str, contexts):
    query_embedding = get_embedding(query)
    document_similarities = sorted([
        (vector_similarity(query_embedding, doc_embedding), doc_index) for doc_index, doc_embedding in contexts.items()
    ], reverse=True)
    return document_similarities

def construct_prompt(question: str, context_embeddings: dict, df) -> str:
    most_relevant_document_sections = order_document_sections_by_query_similarity(question, context_embeddings)
    chosen_sections = []
    chosen_sections_len = 0
    chosen_sections_indexes = []
    for _, section_index in most_relevant_document_sections:
        document_section = df.loc[section_index]
        chosen_sections_len += document_section.tokens + separator_len
        if chosen_sections_len > MAX_SECTION_LEN:
            break
        chosen_sections.append(SEPARATOR + document_section.quote.replace("\n", " "))
        chosen_sections_indexes.append(str(section_index))
    header = """Answer the question as truthfully as possible using the provided context, and if the answer is not contained within the text below, say "I don't know."\n\nContext:\n"""
    return header + "".join(chosen_sections) + "\n\n Q: " + question + "\n A:"

def answer_query_with_context(query: str, df, document_embeddings, show_prompt: bool = False) -> str:
    prompt = construct_prompt(
        query,
        document_embeddings,
        df
    )
    if show_prompt:
        print(prompt)
    response = openai.Completion.create(
                prompt=prompt,
                **COMPLETIONS_API_PARAMS
            )
    return response["choices"][0]["text"].strip(" \n")
