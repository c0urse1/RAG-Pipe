from dataclasses import dataclass


@dataclass(frozen=True)
class AppSettings:
    vector_dir: str = "var/vector_store/e5_large"
    embedding_model: str = "intfloat/multilingual-e5-large-instruct"
