# Custom RAG API

## Overview

This repository contains a custom RAG API implementation using the Llama 3.1 8B Instruct model ([see here](https://huggingface.co/meta-llama/Meta-Llama-3.1-8B-Instruct)), this model was chosen mainly because it is fine-tuned on human instructions. If you lack the VRAM for a model of this size, llama.cpp ([see here](https://github.com/ggerganov/llama.cpp)) might be a better alternative. If you don't want to run the model locally, you can use OpenAI or other models that require credits or subscriptions. The RAG API is provided with a .pdf file, which allows it to take information from this file and reduce hallucinations about the questions provided regarding this topic.

## Setup Instructions

1. Execute ` pip install -r requirements.txt` to install the dependencies.

2. Get yourself a Huggingface API token and place it in the `.env` file.

3. Place a .pdf file in the `data` folder.

3. Run `python3 server.py`, which will download and initialize the model.

4. Run `python3 client.py` to interact with the model.


## Tests

Run tests by running `pytest` in the root directory.