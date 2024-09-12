import re
import torch
import transformers
from typing import List, Tuple
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.prompts import (
    HumanMessagePromptTemplate,
    ChatPromptTemplate,
    PromptTemplate,
)
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_huggingface.embeddings import HuggingFaceEmbeddings


def custom_prompt_processing(template_str: str) -> ChatPromptTemplate:
    """
    Create a custom prompt template for a language model using the provided template string.

    :param template_str: The template string to use for generating the prompt.
    :return: An instance of `ChatPromptTemplate` configured with the provided template.
    """

    prompt_template = PromptTemplate(
        input_variables=["context", "question"], template=template_str
    )

    human_message_prompt = HumanMessagePromptTemplate(prompt=prompt_template)

    prompt = ChatPromptTemplate(
        input_variables=["context", "question"], messages=[human_message_prompt]
    )

    return prompt


def extract_answer(response_text: str) -> str:
    """
    Extract the answer from the response text by splitting and cleaning it.

    :param response_text: The text response from the language model that includes the answer.
    :return: A string containing the cleaned answer. If no answer is found, returns "No answer found."
    """

    parts = response_text.split("Answer:")
    if len(parts) > 1:
        output = parts[1].strip()
        cleaned_response = re.split(
            r"(Question:|Context:|You are an assistant|AI:|Robot:|Human:)", output
        )[0].strip()
        return f"\n\nAnswer: {cleaned_response}\n\n"
    return "No answer found."


def handle_missing_env_vars(variable: str, name: str) -> None:
    """
    Handle missing environment variables.

    :param variable: The environment variable to check.
    :param name: Name of the variable for the error message.
    """

    if variable is None:
        raise ValueError(f"{name} not found. Please set it in the .env file.")


def load_model_tokenizer_pipeline(model: str, token: str) -> Tuple[
    transformers.PreTrainedTokenizer,
    transformers.PreTrainedModel,
    transformers.Pipeline,
]:
    """
    Load a model, tokenizer, and text-generation pipeline from Hugging Face.

    :param model: The model identifier or path on Hugging Face.
    :param token: The Hugging Face API token.
    :return: A tuple containing the tokenizer, model, and the text-generation pipeline.
    """

    # Set quantization configuration to load the llama model with less GPU memory
    bnb_config = transformers.BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    tokenizer = transformers.AutoTokenizer.from_pretrained(model, token=token)

    model = transformers.AutoModelForCausalLM.from_pretrained(
        model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        quantization_config=bnb_config,
        token=token,
    )

    query_pipeline = transformers.pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        torch_dtype=torch.float16,
        max_new_tokens=150,
        truncation=True,
        device_map="auto",
    )

    return tokenizer, model, query_pipeline


def load_documents_and_split(pdf_path: str, chunk_size: int = 1000) -> List[Document]:
    """
    Load a PDF document, split it into chunks, and return the vector store.

    :param pdf_path: Path to the PDF file.
    :param chunk_size: Size of the chunk to split the document.
    :return: List of split documents.
    """

    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=100
    )

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2",
        model_kwargs={"device": "cuda"},
    )

    return Chroma.from_documents(
        documents=text_splitter.split_documents(documents),
        embedding=embeddings,
        collection_metadata={"hnsw:space": "cosine"},
    )


def custom_retriever(query: str, vectordb) -> List[Document]:
    """
    Custom retriever to fetch documents based on the query.

    :param query: Query string from the user.
    :param vectordb: The vector store to search in.
    :return: Filtered list of relevant documents.
    """

    docs, scores = zip(*vectordb.similarity_search_with_score(query, k=20))

    for doc, score in zip(docs, scores):
        doc.metadata["score"] = round(score, 2)

    filtered_results = [doc for doc in docs if (1 - doc.metadata["score"]) >= 0.4]
    return list(filtered_results)
