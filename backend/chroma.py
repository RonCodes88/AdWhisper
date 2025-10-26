"""
ChromaDB Manager - Multi-Collection Vector Database for Ad Bias Detection

Manages four specialized collections:
1. ad_content - Complete ad submissions with combined embeddings
2. bias_patterns_text - Historical text bias patterns for RAG
3. bias_patterns_visual - Historical visual bias patterns for RAG
4. case_studies - Complete case studies with final scores for benchmarking
"""

import chromadb
from typing import Dict, Any, List, Optional
import os
from datetime import datetime, UTC


class ChromaDB:
    _instance = None
    _client = None
    _collections = {}

    # Collection names
    COLLECTION_AD_CONTENT = "ad_content"
    COLLECTION_TEXT_PATTERNS = "bias_patterns_text"
    COLLECTION_VISUAL_PATTERNS = "bias_patterns_visual"
    COLLECTION_CASE_STUDIES = "case_studies"

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ChromaDB, cls).__new__(cls)
            db_path = os.getenv("CHROMA_DB_PATH", ".chroma_db")
            cls._client = chromadb.PersistentClient(path=db_path)
            
            # Initialize all collections
            cls._collections[cls.COLLECTION_AD_CONTENT] = cls._client.get_or_create_collection(
                name=cls.COLLECTION_AD_CONTENT,
                metadata={"description": "Complete ad submissions with embeddings"}
            )
            cls._collections[cls.COLLECTION_TEXT_PATTERNS] = cls._client.get_or_create_collection(
                name=cls.COLLECTION_TEXT_PATTERNS,
                metadata={"description": "Historical text bias patterns"}
            )
            cls._collections[cls.COLLECTION_VISUAL_PATTERNS] = cls._client.get_or_create_collection(
                name=cls.COLLECTION_VISUAL_PATTERNS,
                metadata={"description": "Historical visual bias patterns"}
            )
            cls._collections[cls.COLLECTION_CASE_STUDIES] = cls._client.get_or_create_collection(
                name=cls.COLLECTION_CASE_STUDIES,
                metadata={"description": "Complete case studies with scores"}
            )
        return cls._instance

    @property
    def client(self):
        return self._client

    def get_collection(self, name: str):
        """Get a specific collection by name"""
        if name in self._collections:
            return self._collections[name]
        raise ValueError(f"Collection '{name}' not found")

    @property
    def collection(self):
        """Legacy compatibility - returns ad_content collection"""
        return self._collections[self.COLLECTION_AD_CONTENT]

    # ==================== Ad Content Storage ====================
    
    def store_ad_content(
        self,
        ad_id: str,
        text_embedding: Optional[List[float]] = None,
        visual_embedding: Optional[List[float]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Store complete ad content with embeddings.
        
        Args:
            ad_id: Unique identifier for the ad
            text_embedding: Text embedding vector
            visual_embedding: Visual embedding vector
            metadata: Additional metadata (content_type, timestamp, etc.)
        
        Returns:
            Collection ID where the ad was stored
        """
        collection = self._collections[self.COLLECTION_AD_CONTENT]
        
        # Combine embeddings if both exist, otherwise use whichever is available
        if text_embedding and visual_embedding:
            combined_embedding = text_embedding + visual_embedding
        elif text_embedding:
            combined_embedding = text_embedding
        elif visual_embedding:
            combined_embedding = visual_embedding
        else:
            raise ValueError("At least one embedding (text or visual) must be provided")
        
        # Prepare metadata
        meta = metadata or {}
        meta["stored_at"] = datetime.now(UTC).isoformat()
        meta["has_text_embedding"] = text_embedding is not None
        meta["has_visual_embedding"] = visual_embedding is not None
        
        collection.add(
            embeddings=[combined_embedding],
            metadatas=[meta],
            ids=[ad_id]
        )
        
        return self.COLLECTION_AD_CONTENT

    # ==================== Text Bias Patterns ====================
    
    def store_text_bias_pattern(
        self,
        pattern_id: str,
        embedding: List[float],
        bias_type: str,
        severity: str,
        context: str,
        examples: Optional[List[str]] = None
    ):
        """
        Store a text bias pattern for RAG retrieval.
        
        Args:
            pattern_id: Unique identifier for the pattern
            embedding: Text embedding vector
            bias_type: Type of bias (gender, racial, age, etc.)
            severity: Severity level (low, medium, high, critical)
            context: Contextual description
            examples: Example phrases showing this bias
        """
        collection = self._collections[self.COLLECTION_TEXT_PATTERNS]
        
        metadata = {
            "bias_type": bias_type,
            "severity": severity,
            "context": context,
            "examples": ",".join(examples) if examples else "",
            "stored_at": datetime.now(UTC).isoformat()
        }
        
        collection.add(
            embeddings=[embedding],
            metadatas=[metadata],
            ids=[pattern_id]
        )

    # ==================== Visual Bias Patterns ====================
    
    def store_visual_bias_pattern(
        self,
        pattern_id: str,
        embedding: List[float],
        bias_type: str,
        severity: str,
        visual_features: str,
        context: str
    ):
        """
        Store a visual bias pattern for RAG retrieval.
        
        Args:
            pattern_id: Unique identifier for the pattern
            embedding: Visual embedding vector (CLIP)
            bias_type: Type of visual bias
            severity: Severity level
            visual_features: Description of visual features
            context: Contextual description
        """
        collection = self._collections[self.COLLECTION_VISUAL_PATTERNS]
        
        metadata = {
            "bias_type": bias_type,
            "severity": severity,
            "visual_features": visual_features,
            "context": context,
            "stored_at": datetime.now(UTC).isoformat()
        }
        
        collection.add(
            embeddings=[embedding],
            metadatas=[metadata],
            ids=[pattern_id]
        )

    # ==================== Case Studies ====================
    
    def store_case_study(
        self,
        case_id: str,
        combined_embedding: List[float],
        final_score: float,
        bias_types: List[str],
        recommendations: List[str],
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Store a complete case study for benchmarking.
        
        Args:
            case_id: Unique identifier for the case
            combined_embedding: Combined text+visual embedding
            final_score: Final bias score (0-10)
            bias_types: List of detected bias types
            recommendations: List of recommendations
            metadata: Additional metadata
        """
        collection = self._collections[self.COLLECTION_CASE_STUDIES]
        
        meta = metadata or {}
        meta.update({
            "final_score": final_score,
            "bias_types": ",".join(bias_types),
            "num_recommendations": len(recommendations),
            "stored_at": datetime.now(UTC).isoformat()
        })
        
        collection.add(
            embeddings=[combined_embedding],
            metadatas=[meta],
            ids=[case_id]
        )

    # ==================== Query Methods ====================
    
    def query_by_embedding(
        self,
        collection_name: str,
        embedding: List[float],
        n_results: int = 5,
        where: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Semantic search using embedding similarity.
        
        Args:
            collection_name: Name of the collection to query
            embedding: Query embedding vector
            n_results: Number of results to return
            where: Optional metadata filters
        
        Returns:
            Query results with IDs, distances, metadatas
        """
        collection = self.get_collection(collection_name)
        
        results = collection.query(
            query_embeddings=[embedding],
            n_results=n_results,
            where=where
        )
        
        return results

    def query_by_metadata(
        self,
        collection_name: str,
        filters: Dict[str, Any],
        n_results: int = 10
    ) -> Dict[str, Any]:
        """
        Query by metadata filters.
        
        Args:
            collection_name: Name of the collection to query
            filters: Metadata filters (e.g., {"bias_type": "gender_bias"})
            n_results: Number of results to return
        
        Returns:
            Query results
        """
        collection = self.get_collection(collection_name)
        
        # Get with filters
        results = collection.get(
            where=filters,
            limit=n_results
        )
        
        return results

    # ==================== Legacy Methods (Compatibility) ====================
    
    def add_document(self, doc_id: str, content: str, metadata: dict):
        """Legacy method for compatibility"""
        self._collections[self.COLLECTION_AD_CONTENT].add(
            documents=[content],
            metadatas=[metadata],
            ids=[doc_id]
        )

    def query_documents(self, query: str, n_results: int = 5):
        """Legacy method for compatibility"""
        results = self._collections[self.COLLECTION_AD_CONTENT].query(
            query_texts=[query],
            n_results=n_results
        )
        return results

    def delete_document(self, doc_id: str):
        """Legacy method for compatibility"""
        self._collections[self.COLLECTION_AD_CONTENT].delete(ids=[doc_id])

    # ==================== Utility Methods ====================
    
    def get_collection_count(self, collection_name: str) -> int:
        """Get the number of items in a collection"""
        collection = self.get_collection(collection_name)
        return collection.count()

    def reset_collection(self, collection_name: str):
        """Reset (clear) a collection"""
        self._client.delete_collection(name=collection_name)
        self._collections[collection_name] = self._client.get_or_create_collection(
            name=collection_name
        )