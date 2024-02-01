from langchain_community.vectorstores import Chroma, FAISS


def VectorStoreBuilder(vectorstore_provider, embeddings_model):
    stores = {}

    def inner(tablename):
        if tablename not in stores:
            if vectorstore_provider == "Chroma":
                db = Chroma(
                    persist_directory="./db/chromadb",
                    embedding_function=embeddings_model,
                    collection_name=tablename,
                )
            elif vectorstore_provider == "FAISS":
                try:
                    db = FAISS.load_local(
                        folder_path="./db/faiss",
                        index_name=tablename,
                        embeddings=embeddings_model,
                        allow_dangerous_deserialization=True,
                    )
                except Exception:
                    db = FAISS.from_texts([""], embeddings_model)
                    db.save_local(
                        folder_path="./db/faiss",
                        index_name=tablename
                    )

            stores[tablename] = db
        return stores[tablename]
    return inner
