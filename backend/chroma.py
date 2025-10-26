import chromadb

class ChromaDB:
    _instance = None
    _client = None
    _collection = None
    _text_collection = None
    _visual_collection = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ChromaDB, cls).__new__(cls)
            cls._client = chromadb.PersistentClient(path=".chroma_db")
            # Create separate collections for different embedding dimensions
            cls._text_collection = cls._client.get_or_create_collection(name="adwhisper_text_embeddings")
            cls._visual_collection = cls._client.get_or_create_collection(name="adwhisper_visual_embeddings")
            # Keep legacy collection for backwards compatibility
            cls._collection = cls._text_collection
        return cls._instance

    @property
    def client(self):
        return self._client

    @property
    def collection(self):
        """Legacy property - returns text collection by default"""
        return self._collection

    @property
    def text_collection(self):
        """Collection for text embeddings (384-dim)"""
        return self._text_collection

    @property
    def visual_collection(self):
        """Collection for visual embeddings (512-dim)"""
        return self._visual_collection

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