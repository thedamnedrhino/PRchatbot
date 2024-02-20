import json

from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.docstore.document import Document

from utils import CACHE_DIR

test_documents = [
    'the first name of the laptop is john',
    'the last name of the laptop is doe',
    'the full brand of the laptop is macbook buddy',
]


def get_retriever(issues: bool = True):
    if issues:
        documents = get_issues()
    else:
        raise Exception('some items must be enabled')
    text_splitter = CharacterTextSplitter(chunk_size=250, chunk_overlap=10)
    texts = text_splitter.split_documents(documents)
    # Now we'll create embeddings for our document so we can store it in a vector store and feed the data into an LLM. We'll use the sentence-transformers model for out embeddings. https://www.sbert.net/docs/pretrained_models.html#sentence-embedding-models/
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    embeddings = HuggingFaceEmbeddings(
        model_name=model_name, cache_folder=CACHE_DIR
    )  # Use a pre-cached model
    # Finally we make our Index using chromadb and the embeddings LLM
    chromadb_index = Chroma.from_documents(
        texts, embeddings,
        # persist_directory=CHROMA_DIR, -- this is the persistent version
    )

    retriever = chromadb_index.as_retriever()

    return retriever


def get_issues() -> list[Document]:
    issues = get_issues_json()
    documents = []
    for issue in issues:
        # -- just search the issues by title
        documents.append(Document(issue['title'],
                                  # -- give the full text to the llm
                                  metadata={'formatted': format_issue(issue),
                                            'item_type': 'ISSUE'}))
    return documents


def get_issues_json() -> list[dict]:
    with open('issues.json', 'r') as file:
        docs = json.load(file)
    return docs


def format_issue(issue: dict) -> str:
    """
    Converts an issue in dict format to an str for the LLM.
    """
    return f"""Item type: ISSUE
Number: #{issue['number']}
State: {issue['state']}, 
Labels: {', '.join([l['name'] for l in issue['labels']])},
 
Title: {issue['title']}
Body: {issue['body']}
"""
