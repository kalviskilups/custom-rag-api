from flask import Flask, request, jsonify
import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFacePipeline
from utils import (
    custom_prompt_processing,
    extract_answer,
    handle_missing_env_vars,
    load_model_tokenizer_pipeline,
    load_documents_and_split,
    custom_retriever,
    load_config,
)

app = Flask(__name__)


@app.route("/query", methods=["POST"])
def handle_query():
    """
    Handle a POST request to the /query endpoint. It processes the query, retrieves relevant documents,
    and returns the answer generated by the language model along with document sources.

    :return: A JSON response containing the answer and sources.
    """

    data = request.json
    query = data.get("query")

    if not query:
        return jsonify({"error": "Query parameter is required"}), 400

    results = custom_retriever(query, vectordb)

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
                    "score": round(1 - doc.metadata["score"], 2),
                    "content": doc.page_content.replace("\n", " ")[:100] + "...",
                }
                for idx, doc in enumerate(results)
            ],
        }
    )


@app.route("/health", methods=["GET"])
def health_check():
    """
    Health check endpoint to verify that the server is running.
    :return: A JSON response indicating the server status.
    """

    return jsonify({"status": "ok"}), 200


if __name__ == "__main__":

    # Load the HF_API_TOKEN from .env file
    load_dotenv()
    config = load_config(config_path="config.json")

    # Safety check for the API token
    HF_TOKEN = os.getenv("HF_API_TOKEN")
    handle_missing_env_vars(variable=HF_TOKEN, name="HF_API_TOKEN")

    # Load the model, tokenizer, and the pipeline
    tokenizer, model, query_pipeline = load_model_tokenizer_pipeline(
        model=config.get("model", ""), token=HF_TOKEN
    )

    # Load PDF document and split it into smaller chunks and load into vector store
    vectordb = load_documents_and_split(
        pdf_path=config.get("data", ""),
        embedding_model=config.get("embedding_model", ""),
        chunk_size=1000,
    )

    # Define the custom prompt template
    prompt = custom_prompt_processing(
        template_str=config.get("prompt_template", ""),
    )

    llm = HuggingFacePipeline(pipeline=query_pipeline)

    app.run(debug=False, port=config.get("port", 0))
