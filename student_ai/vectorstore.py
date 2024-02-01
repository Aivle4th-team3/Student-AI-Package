from langchain_community.vectorstores import Chroma


def VectorStoreBuilder(embeddings_model):
    stores = {}

    def inner(tablename):
        if tablename not in stores:
            db = Chroma(
                persist_directory="./db/chromadb",
                embedding_function=embeddings_model,
                collection_name=tablename,
            )
            stores[tablename] = db
        return stores[tablename]
    return inner
