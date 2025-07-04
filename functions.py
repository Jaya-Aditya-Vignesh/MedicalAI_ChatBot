import tempfile
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader, UnstructuredURLLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Pinecone


def data_load_pdf(file):
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_file_path = f"{temp_dir}/{file.name}"
        with open(temp_file_path, "wb") as temp_file:
            temp_file.write(file.read())

        loader = DirectoryLoader(temp_dir, glob="*.pdf", loader_cls=PyPDFLoader)
        data = loader.load()
        return data


def data_load_url(url):
    loader = UnstructuredURLLoader(url)
    data = loader.load()
    return data


def text_split(data, chunk_size, chunk_overlap):
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    text_chunks = splitter.split_documents(data)
    return text_chunks


def huggingface_embeddings():
    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    return embeddings


