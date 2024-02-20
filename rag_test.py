from langchain.chains import RetrievalQA

import retriever
from retriever import get_retriever
from utils import Models

# MODEL = Models.FLAN_LARGE
MODEL = Models.MISTRAL
chain_type = "stuff"  # Options: stuff, map_reduce, refine, map_rerank


laptop_qa = RetrievalQA.from_chain_type(
    llm=Models.get_llm(MODEL), chain_type="stuff", retriever=retriever.get_retriever(issues=True)
)

laptop_name = laptop_qa.run("What is the full name of the laptop?") # str

print(laptop_name)

