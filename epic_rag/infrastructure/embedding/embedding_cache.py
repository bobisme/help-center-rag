"""Caching service for embeddings."""

import asyncio
import hashlib
import json
import logging
import sqlite3
from datetime import datetime, timedelta
from functools import lru_cache
from typing import Dict, List, Optional, Tuple, Any

from ...infrastructure.config.settings import Settings

logger = logging.getLogger(__name__)


class EmbeddingCache:
    """Cache for embeddings to avoid regenerating them repeatedly.
    
    Combines an in-memory LRU cache with a SQLite-based persistent cache.
    """

    def __init__(
        self,
        settings: Settings,
        db_path: Optional[str] = None,
        memory_cache_size: int = 1000,
        cache_expiration_days: int = 30,
    ):
        """Initialize the embedding cache.
        
        Args:
            settings: Application settings
            db_path: Path to SQLite database (defaults to main DB)
            memory_cache_size: Size of the in-memory LRU cache
            cache_expiration_days: Number of days before cache entries expire
        """
        self._settings = settings
        self._db_path = db_path or settings.database.path
        self._memory_cache_size = memory_cache_size
        self._cache_expiration_days = cache_expiration_days
        
        # Initialize the memory cache
        self._init_memory_cache(memory_cache_size)
        
        # Initialize the database cache
        self._init_db_cache()
        
        logger.info(f"Initialized embedding cache with memory size {memory_cache_size}")
        
    def _init_memory_cache(self, size: int) -> None:
        """Initialize the in-memory LRU cache.
        
        Args:
            size: Maximum number of entries in the memory cache
        """
        # Create LRU-cached getter function
        @lru_cache(maxsize=size)
        def _get_cached_vector(cache_key: str) -> Optional[Tuple[List[float], datetime]]:
            return None
        
        self._memory_get = _get_cached_vector
        self._memory_cache = {}  # Separate dict to support updates

    def _init_db_cache(self) -> None:
        """Initialize the SQLite database cache."""
        # Create cache table if it doesn't exist
        conn = sqlite3.connect(self._db_path)
        try:
            cursor = conn.cursor()
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS embedding_cache (
                    cache_key TEXT PRIMARY KEY,
                    provider TEXT NOT NULL,
                    model TEXT NOT NULL,
                    text_hash TEXT NOT NULL,
                    vector BLOB NOT NULL,
                    created_at TIMESTAMP NOT NULL,
                    last_accessed TIMESTAMP NOT NULL
                )
                """
            )
            
            # Create indexes
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_ec_text_hash ON embedding_cache(text_hash)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_ec_provider_model ON embedding_cache(provider, model)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_ec_last_accessed ON embedding_cache(last_accessed)")
            
            conn.commit()
        finally:
            conn.close()

    def _create_cache_key(
        self, text: str, provider: str, model: str, is_query: bool = False
    ) -> str:
        """Create a cache key for the given text and model.
        
        Args:
            text: Text to generate embedding for
            provider: Embedding provider (openai, huggingface, etc.)
            model: Model name
            is_query: Whether this is a query or passage
            
        Returns:
            Cache key string
        """
        # Hash the text for consistency
        text_hash = hashlib.md5(text.encode("utf-8")).hexdigest()
        
        # Create components for the key
        components = {
            "text_hash": text_hash,
            "provider": provider.lower(),
            "model": model,
            "type": "query" if is_query else "passage",
        }
        
        # Generate a consistent string representation
        key_str = json.dumps(components, sort_keys=True)
        
        # Hash the key string for storage efficiency
        return hashlib.sha1(key_str.encode("utf-8")).hexdigest()

    async def get(
        self, text: str, provider: str, model: str, is_query: bool = False
    ) -> Optional[List[float]]:
        """Get embedding from cache if available.
        
        Args:
            text: Text to get embedding for
            provider: Embedding provider
            model: Model name
            is_query: Whether this is a query embedding
            
        Returns:
            Cached embedding vector or None if not found
        """
        cache_key = self._create_cache_key(text, provider, model, is_query)
        text_hash = hashlib.md5(text.encode("utf-8")).hexdigest()
        
        # Try memory cache first (fastest)
        memory_result = self._memory_cache.get(cache_key)
        if memory_result is not None:
            logger.debug(f"Memory cache hit for {provider}/{model}")
            vector, _ = memory_result
            
            # Update access time in DB asynchronously
            asyncio.create_task(self._update_access_time(cache_key))
            
            return vector
            
        # Try database cache
        db_result = await self._get_from_db(cache_key, text_hash, provider, model)
        if db_result is not None:
            logger.debug(f"DB cache hit for {provider}/{model}")
            # Update memory cache
            self._memory_cache[cache_key] = db_result
            return db_result[0]
            
        logger.debug(f"Cache miss for {provider}/{model}")
        return None

    async def set(
        self, text: str, embedding: List[float], provider: str, model: str, is_query: bool = False
    ) -> None:
        """Store embedding in cache.
        
        Args:
            text: Text that was embedded
            embedding: Embedding vector
            provider: Embedding provider
            model: Model name
            is_query: Whether this is a query embedding
        """
        cache_key = self._create_cache_key(text, provider, model, is_query)
        text_hash = hashlib.md5(text.encode("utf-8")).hexdigest()
        now = datetime.now()
        
        # Update memory cache
        self._memory_cache[cache_key] = (embedding, now)
        
        # Update DB cache asynchronously
        asyncio.create_task(
            self._store_in_db(cache_key, text_hash, embedding, provider, model, now)
        )

    async def _get_from_db(
        self, cache_key: str, text_hash: str, provider: str, model: str
    ) -> Optional[Tuple[List[float], datetime]]:
        """Get embedding from the database cache.
        
        Args:
            cache_key: Cache key
            text_hash: MD5 hash of the text
            provider: Embedding provider
            model: Model name
            
        Returns:
            Tuple of (embedding vector, timestamp) if found, None otherwise
        """
        loop = asyncio.get_event_loop()
        
        # Run SQLite query in a thread pool
        try:
            result = await loop.run_in_executor(
                None,
                self._db_get_embedding,
                cache_key, 
                text_hash, 
                provider, 
                model
            )
            return result
        except Exception as e:
            logger.error(f"Error retrieving from cache DB: {e}")
            return None
            
    def _db_get_embedding(
        self, cache_key: str, text_hash: str, provider: str, model: str
    ) -> Optional[Tuple[List[float], datetime]]:
        """Synchronous database query for embeddings.
        
        Args:
            cache_key: Cache key
            text_hash: MD5 hash of the text
            provider: Embedding provider
            model: Model name
            
        Returns:
            Tuple of (embedding vector, timestamp) if found, None otherwise
        """
        # Check for expired entries
        expiration_date = datetime.now() - timedelta(days=self._cache_expiration_days)
        
        conn = sqlite3.connect(self._db_path)
        try:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            # Query for the embedding
            cursor.execute(
                """
                SELECT vector, created_at, last_accessed 
                FROM embedding_cache 
                WHERE cache_key = ? AND created_at > ?
                """,
                (cache_key, expiration_date),
            )
            
            row = cursor.fetchone()
            if row is None:
                return None
                
            # Parse the vector
            vector_blob = row["vector"]
            vector = json.loads(vector_blob)
            created_at = datetime.fromisoformat(row["created_at"])
            
            # Update last accessed time
            cursor.execute(
                "UPDATE embedding_cache SET last_accessed = ? WHERE cache_key = ?",
                (datetime.now().isoformat(), cache_key),
            )
            conn.commit()
            
            return (vector, created_at)
        finally:
            conn.close()

    async def _store_in_db(
        self, 
        cache_key: str, 
        text_hash: str,
        embedding: List[float],
        provider: str,
        model: str,
        timestamp: datetime,
    ) -> None:
        """Store embedding in the database.
        
        Args:
            cache_key: Cache key
            text_hash: MD5 hash of the text
            embedding: Embedding vector
            provider: Embedding provider
            model: Model name
            timestamp: Time when embedding was generated
        """
        loop = asyncio.get_event_loop()
        
        # Run SQLite query in a thread pool
        try:
            await loop.run_in_executor(
                None,
                self._db_store_embedding,
                cache_key, 
                text_hash, 
                embedding, 
                provider, 
                model,
                timestamp,
            )
        except Exception as e:
            logger.error(f"Error storing in cache DB: {e}")
    
    def _db_store_embedding(
        self,
        cache_key: str, 
        text_hash: str,
        embedding: List[float],
        provider: str,
        model: str,
        timestamp: datetime,
    ) -> None:
        """Synchronous database storage for embeddings.
        
        Args:
            cache_key: Cache key
            text_hash: MD5 hash of the text
            embedding: Embedding vector
            provider: Embedding provider
            model: Model name
            timestamp: Time when embedding was generated
        """
        # Serialize embedding to JSON
        vector_blob = json.dumps(embedding)
        timestamp_str = timestamp.isoformat()
        
        conn = sqlite3.connect(self._db_path)
        try:
            cursor = conn.cursor()
            
            # Insert or replace the embedding
            cursor.execute(
                """
                INSERT OR REPLACE INTO embedding_cache 
                (cache_key, provider, model, text_hash, vector, created_at, last_accessed)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    cache_key, 
                    provider, 
                    model, 
                    text_hash, 
                    vector_blob, 
                    timestamp_str, 
                    timestamp_str,
                ),
            )
            
            conn.commit()
        finally:
            conn.close()

    async def _update_access_time(self, cache_key: str) -> None:
        """Update the last accessed time for a cache entry.
        
        Args:
            cache_key: Cache key to update
        """
        loop = asyncio.get_event_loop()
        
        # Run SQLite query in a thread pool
        try:
            await loop.run_in_executor(
                None,
                self._db_update_access_time,
                cache_key,
            )
        except Exception as e:
            logger.error(f"Error updating access time: {e}")
    
    def _db_update_access_time(self, cache_key: str) -> None:
        """Synchronous database update for access time.
        
        Args:
            cache_key: Cache key to update
        """
        now = datetime.now().isoformat()
        
        conn = sqlite3.connect(self._db_path)
        try:
            cursor = conn.cursor()
            cursor.execute(
                "UPDATE embedding_cache SET last_accessed = ? WHERE cache_key = ?",
                (now, cache_key),
            )
            conn.commit()
        finally:
            conn.close()
            
    async def clear_old_entries(self) -> int:
        """Clear old entries from the cache.
        
        Returns:
            Number of entries cleared
        """
        expiration_date = datetime.now() - timedelta(days=self._cache_expiration_days)
        
        loop = asyncio.get_event_loop()
        
        # Run SQLite query in a thread pool
        try:
            return await loop.run_in_executor(
                None,
                self._db_clear_old_entries,
                expiration_date,
            )
        except Exception as e:
            logger.error(f"Error clearing old entries: {e}")
            return 0
    
    def _db_clear_old_entries(self, expiration_date: datetime) -> int:
        """Synchronous database operation to clear old entries.
        
        Args:
            expiration_date: Date before which entries should be cleared
            
        Returns:
            Number of entries cleared
        """
        conn = sqlite3.connect(self._db_path)
        try:
            cursor = conn.cursor()
            cursor.execute(
                "DELETE FROM embedding_cache WHERE created_at < ?",
                (expiration_date.isoformat(),),
            )
            deleted_count = cursor.rowcount
            conn.commit()
            return deleted_count
        finally:
            conn.close()

    async def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the cache.
        
        Returns:
            Dictionary of cache statistics
        """
        loop = asyncio.get_event_loop()
        
        # Run SQLite query in a thread pool
        try:
            return await loop.run_in_executor(
                None,
                self._db_get_stats,
            )
        except Exception as e:
            logger.error(f"Error getting cache stats: {e}")
            return {"error": str(e)}
    
    def _db_get_stats(self) -> Dict[str, Any]:
        """Synchronous database operation to get cache statistics.
        
        Returns:
            Dictionary of cache statistics
        """
        conn = sqlite3.connect(self._db_path)
        try:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            # Get total count
            cursor.execute("SELECT COUNT(*) as count FROM embedding_cache")
            total_count = cursor.fetchone()["count"]
            
            # Get counts by provider
            cursor.execute(
                "SELECT provider, COUNT(*) as count FROM embedding_cache GROUP BY provider"
            )
            provider_counts = {row["provider"]: row["count"] for row in cursor.fetchall()}
            
            # Get counts by model
            cursor.execute(
                "SELECT model, COUNT(*) as count FROM embedding_cache GROUP BY model"
            )
            model_counts = {row["model"]: row["count"] for row in cursor.fetchall()}
            
            # Get newest and oldest entries
            cursor.execute(
                "SELECT MIN(created_at) as oldest, MAX(created_at) as newest FROM embedding_cache"
            )
            dates = cursor.fetchone()
            
            # Get size approximation
            cursor.execute(
                "SELECT SUM(length(vector)) as total_size FROM embedding_cache"
            )
            size_row = cursor.fetchone()
            total_size = size_row["total_size"] if size_row and size_row["total_size"] else 0
            
            # Memory cache stats
            memory_size = len(self._memory_cache)
            
            return {
                "total_entries": total_count,
                "by_provider": provider_counts,
                "by_model": model_counts,
                "oldest_entry": dates["oldest"] if dates else None,
                "newest_entry": dates["newest"] if dates else None,
                "db_size_bytes": total_size,
                "memory_entries": memory_size,
                "memory_max_size": self._memory_cache_size,
            }
        finally:
            conn.close()