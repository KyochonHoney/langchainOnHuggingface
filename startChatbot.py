from transformers import BitsAndBytesConfig
from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain import hub
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
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype="float16",
    bnb_4bit_use_double_quant=True,
)

chat_model = HuggingFacePipeline.from_model_id(
    model_id='LGAI-EXAONE/EXAONE-3.0-7.8B-Instruct',
    task='text-generation',
    pipeline_kwargs=dict(
        max_new_tokens=1024,
        do_sample=False,
        repetition_penalty=1.03
    ),
    model_kwargs={
        'quantization_config': quantization_config,
        'trust_remote_code': True
    }
)

llm = ChatHuggingFace(llm=chat_model)
retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")
retriever = database.as_retriever(search_kwargs={"k": 1})
query = "연봉 5천만원인 거주자의 소득세는?"
retrieved_docs = retriever.invoke(query)

combine_docs_chain = create_stuff_documents_chain(
    llm, retrieval_qa_chat_prompt
)
retrieval_chain = create_retrieval_chain(retriever, combine_docs_chain)

ai_message = retrieval_chain.invoke({"input": query})
print(ai_message)
