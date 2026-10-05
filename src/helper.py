from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from typing import List
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
import os


#extract the text from the pdf files
def load_pdf_file(data):
    loader = directory_loader = DirectoryLoader(
        data, 
        glob="*.pdf",
        loader_cls=PyPDFLoader
         )
    documents = loader.load()
    return documents

def filter_to_minimal_docs(docs: List[Document]) -> List[Document]:
    """
    Filter the list of documents to only include those with  'source in metadata and a orignal page_content .


    """
    minimal_docs :List[Document] = []
    for doc in docs:
        src =doc.metadata.get("source")
        minimal_docs.append(Document(page_content=doc.page_content,
         metadata={"source": src}))
    return minimal_docs

#split the document into smaller chunks
def text_split(minimal_docs):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=20,
    )
    texts_chunk = text_splitter.split_documents(minimal_docs)
    return texts_chunk


def download_hugging_face_embeddings():
    '''
    Download the embeddings model from HuggingFace and return the embeddings object.
    '''
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
    )
    return embeddings

embeddings = download_hugging_face_embeddings() 