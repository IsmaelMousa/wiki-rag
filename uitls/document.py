import os
from warnings                             import filterwarnings; filterwarnings("ignore")
from langchain_community.document_loaders import WikipediaLoader
from langchain.schema                     import Document
from langchain_text_splitters             import CharacterTextSplitter, TokenTextSplitter, RecursiveCharacterTextSplitter


def load_documents(query: str):
    """
    Load the wikipedia documents based on a query.

    :param query: search query (topic).
    :return: nothing, just save the documents in local directory.
    """
    path = "documents"

    if not os.path.exists(path):
        os.makedirs(path)

        documents = WikipediaLoader(query=query).load()

        for index, document in enumerate(documents):
            with open(os.path.join(path, f"{index}.txt"), mode="w", encoding="utf-8") as file:
                file.write(document.page_content)

        return len(documents)

    else:
        return "documents already loaded!"


def chunk_documents(level        : str = "token",
                    model_name   : str = "gpt2",
                    chunk_size   : int = 256,
                    chunk_overlap: int = 0,
                    separators   : list | None = None):
    """
    Splits loaded documents into smaller chunks based on the chosen level.

    :param level: the chunking level/strategy.
    :param model_name: the encoding model.
    :param chunk_size: the chunk size.
    :param chunk_overlap: the overlap between chunks.
    :param separators: the separators for custom chunking.
    :return: list of documents.
    """
    path      = "documents"
    documents = []

    if not os.path.exists(path):
        return "documents not loaded, call the `load` function first."

    for document in os.listdir(path=path):
        with open(os.path.join(path, document), mode="r", encoding="utf-8") as file:
            documents.append(file.read())

    documents = [Document(page_content=document) for document in documents]

    match level:
        case "character":
            splitter = CharacterTextSplitter(chunk_size            =chunk_size,
                                             chunk_overlap         =chunk_overlap)
        case "token":
            splitter = TokenTextSplitter(chunk_size                =chunk_size,
                                         chunk_overlap             =chunk_overlap,
                                         model_name                =model_name)
        case "custom":
            splitter = RecursiveCharacterTextSplitter(chunk_size   =chunk_size,
                                                      chunk_overlap=chunk_overlap,
                                                      separators   =separators)
        case _:
            splitter = TokenTextSplitter(chunk_size                =chunk_size,
                                         chunk_overlap             =chunk_overlap,
                                         model_name                =model_name)

    documents = splitter.split_documents(documents=documents)

    return documents
