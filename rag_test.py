from langchain.chains import RetrievalQA

from utils import CACHE_DIR, CHROMA_DIR,Models

# MODEL = Models.FLAN_LARGE
MODEL = Models.MISTRAL


print('******chroma done*****')

chain_type = "stuff"  # Options: stuff, map_reduce, refine, map_rerank
laptop_qa = RetrievalQA.from_chain_type(
    llm=Models.get_llm(MODEL), chain_type="stuff", retriever=retriever
)

laptop_name = laptop_qa.run("What is the full name of the laptop?") # str

print(laptop_name)

