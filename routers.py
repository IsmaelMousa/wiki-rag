from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains                   import create_retrieval_chain
from langchain_core.prompts             import ChatPromptTemplate
from transformers                       import pipeline
from langchain_community.vectorstores   import FAISS
from langchain_huggingface              import HuggingFacePipeline
from pydantic                           import BaseModel
from fastapi                            import APIRouter
from config                             import get_config
from uitls                              import *


class Query(BaseModel):
    prompt  : str
    question: str
    note    : str = ""


config = get_config()
router = APIRouter()


@router.get(path="/", status_code=200)
def root():
    """
    Health check endpoint for the RAG system.

    :return: welcome message.
    """
    return {"message": "RAG system say hello!"}

@router.get("/documents", status_code=200)
def load(query: str):
    """
    Loads documents that match the given query.

    :param query: the query as a text.
    :return: message or total number of loaded documents.
    """
    documents = load_documents(query=query)

    return {"message": documents}

@router.get("/model", status_code=200)
def load():
    """
    Initializes and loads embedding, tokenizer and generative model.

    :return: embedding, tokenizer and generative model.
    """
    retriever = load_retriever(embedding=config.model.embedding)["message"]
    generator = load_generator(pretrained_model=config.model.pretrained_model)["message"]

    return {"message": {"retriever": retriever, "generator": generator}}

@router.post(path="/model", status_code=200)
def generate(query: Query):
    """
    Generates an answer based on the query using RAG (retrieval system + generative model).

    :param query: the query as an object.
    :return: context-aware answer.
    """
    documents = chunk_documents(**config.document.__dict__)

    embedding = load_retriever(embedding=config.model.embedding)["retriever"]

    vectordb  = FAISS.from_documents(documents=documents, embedding=embedding)

    retriever = vectordb.as_retriever(search_type=config.model.search_type, k=config.model.k)

    generator = load_generator(pretrained_model=config.model.pretrained_model)

    tokenizer = generator["tokenizer"]
    model     = generator["model"]

    pipe      = pipeline(task            ="text-generation",
                         model           =model,
                         tokenizer       =tokenizer,
                         max_new_tokens  =config.model.max_new_tokens,
                         return_full_text=False,
                         pad_token_id    =tokenizer.eos_token_id)
    llm       = HuggingFacePipeline(pipeline=pipe)

    prompt    = (query.prompt + " Based on the context:\n{context}\n\nQuestion: {input}").strip()
    prompt    = ChatPromptTemplate.from_messages([("human", prompt)])

    chain     = create_stuff_documents_chain(llm, prompt)
    chain     = create_retrieval_chain(retriever, chain)

    query     = (query.question + " " + query.note).strip()

    answer    = chain.invoke({"input": query})
    context   = answer["context"][0].page_content

    return {"message": {"answer": answer["answer"], "context": context}}
