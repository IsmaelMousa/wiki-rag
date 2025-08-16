from transformers                         import AutoModelForCausalLM, AutoTokenizer
from langchain_huggingface                import HuggingFaceEmbeddings


def load_retriever(embedding: str):
    """
    Loads the given embedding model.

    :param embedding: the model id.
    :return: message and embedding model.
    """
    retriever = HuggingFaceEmbeddings(model_name=embedding)

    return {"message": "retriever successfully loaded!", "retriever": retriever}


def load_generator(pretrained_model: str):
    """
    Loads a tokenizer and model from a pretrained checkpoint.

    :param pretrained_model: the model id.
    :return: message, tokenizer and model.
    """
    tokenizer    = AutoTokenizer       .from_pretrained(pretrained_model_name_or_path=pretrained_model)
    model        = AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path=pretrained_model)

    return {"message": "tokenizer and model successfully loaded!", "tokenizer": tokenizer, "model": model}
