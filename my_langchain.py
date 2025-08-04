from langchain_community.document_loaders import Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 700,
    chunk_overlap = 100
)

loader = Docx2txtLoader('./tax_with_table.docx')
document_list = loader.load_and_split(text_splitter)

embedding = HuggingFaceEmbeddings(model_name='intfloat/multilingual-e5-large-instruct')

collection_name = 'tax-table-index'
database = Chroma.from_documents(
    documents=document_list,
    embedding=embedding,
    collection_name=collection_name,
    persist_directory='./chroma_huggingface'
)


