import os
import time
import torch
import transformers
from dotenv import load_dotenv
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain_community.llms import HuggingFacePipeline
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
# pip install accelerate

load_dotenv()

HF_TOKEN = os.getenv('HUGGINGFACE_API_TOKEN')
if HF_TOKEN is None:
    raise ValueError("Access token not found. Please set the HUGGINGFACE_API_TOKEN in the .env file.")
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# set quantization configuration to load large model with less GPU memory
# this requires the `bitsandbytes` library
bnb_config = transformers.BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type='nf4',
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16
)

tokenizer = transformers.AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct", token = HF_TOKEN)
model = transformers.AutoModelForCausalLM.from_pretrained(
    "meta-llama/Meta-Llama-3-8B-Instruct",
    torch_dtype=torch.bfloat16,
    device_map="auto",
    quantization_config=bnb_config,
    token = HF_TOKEN
)

query_pipeline = transformers.pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        torch_dtype=torch.float16,
        max_length=1024,
        device_map="auto",)

llm = HuggingFacePipeline(pipeline=query_pipeline)

# This is just a test
# time_start = time.time()
# question = "Please explain what EU AI Act is."
# response = llm(prompt=question)
# time_end = time.time()
# total_time = f"{round(time_end-time_start, 3)} sec."
# response =  f"Question: {question}\nAnswer: {response}\nTotal time: {total_time}"
# print(response)


# Rag pipeline starts from here
loader = PyPDFLoader("NVIDIAAn.pdf")
documents = loader.load()
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2", model_kwargs = {"device": "cuda"})
print(embeddings)

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
all_splits = text_splitter.split_documents(documents)

vectordb = Chroma.from_documents(documents=all_splits, embedding=embeddings, persist_directory="chroma_db")

retriever = vectordb.as_retriever()

qa = RetrievalQA.from_chain_type(
    llm=llm, 
    chain_type="stuff", 
    retriever=retriever, 
    verbose=True
)

# Test RAG pipeline
# time_start = time()
# response = qa.run(query)
# time_end = time()
# total_time = f"{round(time_end-time_start, 3)} sec."

# full_response =  f"Question: {query}\nAnswer: {response}\nTotal time: {total_time}"
# print(full_response)


# Client side

query = "What are the operational obligations of notified bodies?"
qa.run(query)

docs = vectordb.similarity_search(query)
print(f"Query: {query}")
print(f"Retrieved documents: {len(docs)}")
for doc in docs:
    doc_details = doc.to_json()['kwargs']
    print("Source: ", doc_details['metadata']['source'])
    print("Text: ", doc_details['page_content'], "\n")