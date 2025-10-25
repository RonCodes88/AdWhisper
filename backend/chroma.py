import chromadb

class ChromaDB:
    _instance = None
    _client = None
    _collection = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ChromaDB, cls).__new__(cls)
            cls._client = chromadb.PersistentClient(path=".chroma_db")
            cls._collection = cls._client.get_or_create_collection(name="adwhisper_collection")
        return cls._instance

    @property
    def client(self):
        return self._client

    @property
    def collection(self):
        return self._collection

    def add_document(self, doc_id: str, content: str, metadata: dict):
        self._collection.add(
            documents=[content],
            metadatas=[metadata],
            ids=[doc_id]
        )

    def query_documents(self, query: str, n_results: int = 5):
        results = self._collection.query(
            query_texts=[query],
            n_results=n_results
        )
        return results

    def delete_document(self, doc_id: str):
        self._collection.delete(ids=[doc_id])