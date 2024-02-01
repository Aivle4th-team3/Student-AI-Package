from langchain_community.vectorstores import Chroma, FAISS
from langchain_pinecone import Pinecone


def VectorStoreBuilder(provider, embeddings_model):
    stores = {}

    def inner(tablename):
        if tablename not in stores:
            if provider == "Chroma":
                db = Chroma(
                    persist_directory="./db/chromadb",
                    embedding_function=embeddings_model,
                    collection_name=tablename,
                )
            elif provider == "FAISS":
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
            elif provider == "PINECONE":
                try:
                    db = Pinecone(
                        index_name="ai-student",
                        embedding=embeddings_model,
                        namespace=tablename,
                    )
                except Exception:
                    db = Pinecone.from_texts(
                        texts=[""],
                        index_name="ai-student",
                        embedding=embeddings_model,
                        namespace=tablename,
                    )

            stores[tablename] = db
        return stores[tablename]
    return inner
