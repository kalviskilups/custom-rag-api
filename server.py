from flask import Flask, request, jsonify
import os
import torch
import transformers
from typing import List
from dotenv import load_dotenv
from langchain_core.runnables import chain
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFacePipeline
from langchain_community.vectorstores import Chroma
from langchain_huggingface.embeddings import HuggingFaceEmbeddings

from utils import custom_prompt_processing, extract_answer

app = Flask(__name__)

@chain
def retriever(query: str) -> List[Document]:
    docs, scores = zip(*vectordb.similarity_search_with_score(query, k=20))
    for doc, score in zip(docs, scores):
        doc.metadata["score"] = score
    return docs

def custom_retriever(query):
    results = retriever.invoke(query)
    results = list(results)
    filtered_results = [
        doc for doc in results
        if (1 - doc.metadata["score"]) >= 0.4
    ]
    results = tuple(filtered_results)

    return results

@app.route("/query", methods=["POST"])
def handle_query():
    data = request.json
    query = data.get("query")

    if not query:
        return jsonify({"error": "Query parameter is required"}), 400

    results = custom_retriever(query)

    formatted_context = "\n\n".join(
        doc.page_content.replace("\n", " ") for doc in results
    )
    formatted_query = prompt.format(context=formatted_context, question=query)
    response = llm.invoke(formatted_query)

    answer_only = extract_answer(response)

    return jsonify(
        {
            "answer": answer_only,
            "sources": [
                {
                    "source": idx + 1,
                    "page": doc.metadata["page"],
                    "content": doc.page_content.replace("\n", " ")[:100] + "...",
                }
                for idx, doc in enumerate(results)
            ],
        }
    )


if __name__ == "__main__":

    # Load the HF_API_TOKEN from .env file
    load_dotenv()

    # Safety check for the API token
    HF_TOKEN = os.getenv("HF_API_TOKEN")
    if HF_TOKEN is None:
        raise ValueError(
            "Access token not found. Please set the HF_API_TOKEN in the .env file."
        )

    # Set quantization configuration to load the llama model with less GPU memory
    bnb_config = transformers.BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    # Load the model, the tokenizer and set yp the pipeline
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        "meta-llama/Meta-Llama-3-8B-Instruct", token=HF_TOKEN
    )

    model = transformers.AutoModelForCausalLM.from_pretrained(
        "meta-llama/Meta-Llama-3-8B-Instruct",
        torch_dtype=torch.bfloat16,
        device_map="auto",
        quantization_config=bnb_config,
        token=HF_TOKEN,
    )

    query_pipeline = transformers.pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        torch_dtype=torch.float16,
        max_new_tokens=250,
        truncation=True,
        device_map="auto",
        )

    # Load PDF document
    loader = PyPDFLoader("NVIDIAAn.pdf")
    documents = loader.load()

    # Create embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2",
        model_kwargs={"device": "cuda"},
    )

    # Split the document into smaller chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    all_splits = text_splitter.split_documents(documents)

    # Create a vector store with Chroma
    vectordb = Chroma.from_documents(
        documents=all_splits,
        embedding=embeddings,
        collection_metadata={"hnsw:space": "cosine"},
    )

    # Define the prompt template

    prompt = custom_prompt_processing(
        template_str="""
        You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. 
        If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.

        Question: {question}
        Context: {context}
        Answer:
        """
    )

    llm = HuggingFacePipeline(pipeline=query_pipeline)


    app.run(debug=True, port=2026)
