import os
from dataclasses import dataclass
from functools   import lru_cache
from yaml        import safe_load


@dataclass(frozen=True)
class Document:
    """
    Configuration for document chunking.
    """
    level        : str
    model_name   : str
    chunk_size   : int
    chunk_overlap: int
    separators   : list


@dataclass(frozen=True)
class Model:
    """
    Configuration for embedding and generation models.
    """
    embedding       : str
    pretrained_model: str
    search_type     : str
    k               : int
    max_new_tokens  : int


@dataclass(frozen=True)
class App:
    """
    Configuration for the application settings.
    """

    host     : str
    port     : int
    path     : str
    version  : str


@dataclass(frozen=True)
class Config:
    """
    Top level configuration holding document, model and app settings.
    """
    app     : App
    document: Document
    model   : Model


@lru_cache(maxsize=1)
def get_config():
    """
    Loads configuration from the config.yml file and caches it.

    :return: configuration instance/object.
    """
    root = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(root, "./config.yml")

    with open(path, "r") as file:
        configurations = safe_load(file)
        document_cfg   = configurations.get("document", {})
        model_cfg      = configurations.get("model"   , {})
        app_cfg        = configurations.get("app"     , {})

        document = Document(level        =document_cfg.get("level"),
                            model_name   =document_cfg.get("model_name"),
                            chunk_size   =document_cfg.get("chunk_size"),
                            chunk_overlap=document_cfg.get("chunk_overlap"),
                            separators   =document_cfg.get("separators"))

        model    = Model(embedding       =model_cfg.get("embedding"),
                         pretrained_model=model_cfg.get("pretrained_model"),
                         search_type     =model_cfg.get("search_type"),
                         k               =model_cfg.get("k"),
                         max_new_tokens  =model_cfg.get("max_new_tokens"))

        app      = App(host              =app_cfg.get("host"),
                       port              =app_cfg.get("port"),
                       path              =app_cfg.get("path"),
                       version           =app_cfg.get("version"))


        config   = Config(document       =document,
                          model          =model,
                          app            =app)

    return config
