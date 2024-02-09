from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.llms import HuggingFacePipeline
from langchain.docstore.document import Document
from langchain.llms import LlamaCpp
from langchain.embeddings import LlamaCppEmbeddings

CACHE_DIR = './cache'
CHROMA_DIR = './chroma'
# document = 'the first part of the name of this laptop is john' * 100 + 'the second part is atkins' * 100
document = 'the name of the laptop is john doe'

# First we split the data into manageable chunks to store as vectors. There isn't an exact way to do this, more chunks means more detailed context, but will increase the size of our vectorstore.
text_splitter = CharacterTextSplitter(chunk_size=250, chunk_overlap=10)
texts = text_splitter.split_documents([Document(document, metadata={'desc': 'first_test_doc'})])
# Now we'll create embeddings for our document so we can store it in a vector store and feed the data into an LLM. We'll use the sentence-transformers model for out embeddings. https://www.sbert.net/docs/pretrained_models.html#sentence-embedding-models/
model_name = "sentence-transformers/all-MiniLM-L6-v2"
embeddings = HuggingFaceEmbeddings(
    model_name=model_name, cache_folder=CACHE_DIR
)  # Use a pre-cached model
# Finally we make our Index using chromadb and the embeddings LLM
chromadb_index = Chroma.from_documents(
    texts, embeddings, persist_directory=CHROMA_DIR
)

retriever = chromadb_index.as_retriever()


chain_type = "stuff"  # Options: stuff, map_reduce, refine, map_rerank
laptop_qa = RetrievalQA.from_chain_type(
    llm=hf_llm, chain_type="stuff", retriever=retriever
)

laptop_name = laptop_qa.run("What is the full name of the laptop?") # str

print(laptop_name)

def get_embed_and_llm():
    if MODEL == 'google':
        model = HuggingFacePipeline.from_model_id(
            model_id="google/flan-t5-large",
            task="text2text-generation",
            model_kwargs={
                "temperature": 0,
                "max_length": 128,
                "cache_dir": CACHE_DIR,
            },
        )
    elif MODEL == 'astral':
        # model code from:
        # https://huggingface.co/TheBloke/MPT-7B-Instruct-GGML/discussions/2
        model_id = "TheBloke/Mistral-7B-OpenOrca-GGUF",
        llm = LlamaCpp
        model = HuggingFacePipeline(pipeline=pipe)
        embeddings = LlamaCppEmbeddings(model_path="models/llama-7b.ggmlv3.q4_0.bin")
