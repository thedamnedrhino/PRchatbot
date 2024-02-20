from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.docstore.document import Document

from utils import CACHE_DIR, CHROMA_DIR,Models

# MODEL = Models.FLAN_LARGE
MODEL = Models.MISTRAL

# document = 'the first part of the name of this laptop is john' * 100 + 'the second part is atkins' * 100
# document = 'the name of the laptop is john doe'
documents = [
    'the first name of the laptop is john',
    'the last name of the laptop is doe',
    'the full brand of the laptop is macbook buddy',
]
# First we split the data into manageable chunks to store as vectors. There isn't an exact way to do this, more chunks means more detailed context, but will increase the size of our vectorstore.
text_splitter = CharacterTextSplitter(chunk_size=250, chunk_overlap=10)
texts = text_splitter.split_documents([Document(d) for d in documents])
# Now we'll create embeddings for our document so we can store it in a vector store and feed the data into an LLM. We'll use the sentence-transformers model for out embeddings. https://www.sbert.net/docs/pretrained_models.html#sentence-embedding-models/
model_name = "sentence-transformers/all-MiniLM-L6-v2"
embeddings = HuggingFaceEmbeddings(
    model_name=model_name, cache_folder=CACHE_DIR
)  # Use a pre-cached model
# Finally we make our Index using chromadb and the embeddings LLM
chromadb_index = Chroma.from_documents(
    texts, embeddings, # persist_directory=CHROMA_DIR, -- this is the persistent version
)

retriever = chromadb_index.as_retriever()

print('******chroma done*****')

chain_type = "stuff"  # Options: stuff, map_reduce, refine, map_rerank
laptop_qa = RetrievalQA.from_chain_type(
    llm=Models.get_llm(MODEL), chain_type="stuff", retriever=retriever
)

laptop_name = laptop_qa.run("What is the full name of the laptop?") # str

print(laptop_name)

