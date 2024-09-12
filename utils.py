from typing import List
from langchain_core.prompts import HumanMessagePromptTemplate, ChatPromptTemplate, PromptTemplate

def custom_prompt_processing(template_str):

    prompt_template = PromptTemplate(
        input_variables=["context", "question"],
        template=template_str
    )

    human_message_prompt = HumanMessagePromptTemplate(prompt=prompt_template)

    # Create the ChatPromptTemplate instance
    prompt = ChatPromptTemplate(
        input_variables=["context", "question"],
        messages=[human_message_prompt]
        )

    return prompt

def extract_answer(response_text):
    parts = response_text.split("Answer:")
    if len(parts) > 1:
        return f"\n\nAnswer: {parts[1].strip()}\n\n"
    return "No answer found."