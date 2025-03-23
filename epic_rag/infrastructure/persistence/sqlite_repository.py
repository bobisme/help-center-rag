"""SQLite implementation of the document repository."""

import json
import asyncio
from datetime import datetime
from typing import List, Optional, Dict, Any, Tuple
import aiosqlite
from loguru import logger

from ...domain.models.document import Document, DocumentChunk
from ...domain.models.ident import new_id
from ...domain.repositories.document_repository import DocumentRepository


class SQLiteDocumentRepository(DocumentRepository):
    """SQLite implementation of the document repository."""

    def __init__(self, db_path: str, enable_json: bool = True):
        """Initialize the repository.

        Args:
            db_path: Path to the SQLite database file
            enable_json: Enable JSON extension support
        """
        self.db_path = db_path
        self.enable_json = enable_json
        logger.debug(f"Initializing SQLiteDocumentRepository with db_path: {db_path}")

        # Initialize database schema
        try:
            # With Python 3.10+, use a more future-proof approach
            try:
                # This is the modern way to get the running loop
                loop = asyncio.get_running_loop()
                # If we're in a running event loop, we need to use a different approach
                logger.debug("Using running event loop for database initialization")
                asyncio.create_task(self._initialize_db())
            except RuntimeError:
                # No running event loop
                logger.debug("Creating new event loop for database initialization")
                asyncio.run(self._initialize_db())
        except Exception as e:
            # Fall back if anything goes wrong
            logger.debug(f"Error in event loop handling: {str(e)}. Using asyncio.run()")
            asyncio.run(self._initialize_db())

    async def _initialize_db(self):
        """Initialize the database schema."""
        # Ensure parent directory exists
        import os

        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)

        async with aiosqlite.connect(self.db_path) as db:
            # Configure database settings
            if self.enable_json:
                # Load the JSON extension
                await db.execute(
                    "PRAGMA module_list;"
                )  # This forces SQLite to load extensions

            # Set pragmas for performance and reliability
            await db.execute("PRAGMA journal_mode=WAL;")  # Write-ahead logging
            await db.execute(
                "PRAGMA foreign_keys=ON;"
            )  # Enable foreign key constraints
            await db.execute(
                "PRAGMA synchronous=NORMAL;"
            )  # Balance between safety and speed
            await db.execute("PRAGMA temp_store=MEMORY;")  # Store temp tables in memory
            await db.execute(
                "PRAGMA cache_size=-10000;"
            )  # Use 10MB of memory for cache

            # Create documents table
            await db.execute(
                """
                CREATE TABLE IF NOT EXISTS documents (
                    id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    content TEXT NOT NULL,
                    metadata TEXT,
                    epic_page_id TEXT,
                    epic_path TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
            """
            )

            # Create chunks table
            await db.execute(
                """
                CREATE TABLE IF NOT EXISTS chunks (
                    id TEXT PRIMARY KEY,
                    document_id TEXT NOT NULL,
                    content TEXT NOT NULL,
                    metadata TEXT,
                    chunk_index INTEGER NOT NULL,
                    previous_chunk_id TEXT,
                    next_chunk_id TEXT,
                    relevance_score REAL,
                    vector_id TEXT,
                    FOREIGN KEY (document_id) REFERENCES documents(id) ON DELETE CASCADE
                );
            """
            )

            # Create query history table
            await db.execute(
                """
                CREATE TABLE IF NOT EXISTS query_history (
                    id TEXT PRIMARY KEY,
                    query_text TEXT NOT NULL,
                    transformed_query TEXT,
                    timestamp TEXT NOT NULL,
                    metadata TEXT
                );
            """
            )

            # Create statistics table
            await db.execute(
                """
                CREATE TABLE IF NOT EXISTS statistics (
                    key TEXT PRIMARY KEY,
                    value TEXT,
                    updated_at TEXT
                );
            """
            )

            # Create indices for performance
            await db.execute(
                "CREATE INDEX IF NOT EXISTS idx_chunks_document_id ON chunks(document_id);"
            )
            await db.execute(
                "CREATE INDEX IF NOT EXISTS idx_documents_epic_page_id ON documents(epic_page_id);"
            )
            await db.execute(
                "CREATE INDEX IF NOT EXISTS idx_document_title ON documents(title);"
            )
            await db.execute(
                "CREATE INDEX IF NOT EXISTS idx_query_timestamp ON query_history(timestamp);"
            )

            # Initialize statistics
            await db.execute(
                """
                INSERT OR IGNORE INTO statistics (key, value, updated_at)
                VALUES ('document_count', '0', datetime('now'))
            """
            )
            await db.execute(
                """
                INSERT OR IGNORE INTO statistics (key, value, updated_at)
                VALUES ('chunk_count', '0', datetime('now'))
            """
            )

            await db.commit()

    async def save_document(self, document: Document) -> Document:
        """Save a document to the repository."""
        async with aiosqlite.connect(self.db_path) as db:
            # Ensure timestamps are set
            if not document.created_at:
                document.created_at = datetime.now()
            document.updated_at = datetime.now()

            # Convert metadata to JSON string
            metadata_json = json.dumps(document.metadata)

            # Check if document exists
            is_new = False
            async with db.execute(
                "SELECT id FROM documents WHERE id = ?", (document.id,)
            ) as cursor:
                if await cursor.fetchone() is None:
                    is_new = True

            # Insert or replace document
            await db.execute(
                """
                INSERT OR REPLACE INTO documents
                (id, title, content, metadata, epic_page_id, epic_path, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    document.id,
                    document.title,
                    document.content,
                    metadata_json,
                    document.epic_page_id,
                    document.epic_path,
                    document.created_at.isoformat(),
                    document.updated_at.isoformat(),
                ),
            )

            # Update document count if it's a new document
            if is_new:
                await db.execute(
                    """
                    UPDATE statistics
                    SET value = CAST(CAST(value AS INTEGER) + 1 AS TEXT),
                        updated_at = datetime('now')
                    WHERE key = 'document_count'
                """
                )

            await db.commit()

            return document

    async def get_document(self, document_id: str) -> Optional[Document]:
        """Get a document by ID."""
        async with aiosqlite.connect(self.db_path) as db:
            # Get document
            async with db.execute(
                """
                SELECT id, title, content, metadata, epic_page_id, epic_path, created_at, updated_at
                FROM documents
                WHERE id = ?
            """,
                (document_id,),
            ) as cursor:
                row = await cursor.fetchone()

                if not row:
                    return None

                # Parse document
                document = Document(
                    id=row[0],
                    title=row[1],
                    content=row[2],
                    metadata=json.loads(row[3]) if row[3] else {},
                    epic_page_id=row[4],
                    epic_path=row[5],
                    created_at=datetime.fromisoformat(row[6]),
                    updated_at=datetime.fromisoformat(row[7]),
                )

                # Get chunks
                document.chunks = await self.get_document_chunks(document_id)

                return document

    async def list_documents(
        self,
        limit: int = 100,
        offset: int = 0,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Document]:
        """List documents with optional filtering."""
        async with aiosqlite.connect(self.db_path) as db:
            # Build query based on filters
            query = """
                SELECT id, title, content, metadata, epic_page_id, epic_path, created_at, updated_at
                FROM documents
            """
            params = []

            # Add filters if provided
            where_clauses = []
            if filters:
                if "epic_page_id" in filters:
                    where_clauses.append("epic_page_id = ?")
                    params.append(filters["epic_page_id"])

                if "title" in filters:
                    where_clauses.append("title LIKE ?")
                    params.append(f"%{filters['title']}%")

            if where_clauses:
                query += " WHERE " + " AND ".join(where_clauses)

            # Add limit and offset
            query += " ORDER BY updated_at DESC LIMIT ? OFFSET ?"
            params.extend([limit, offset])

            # Execute query
            documents = []
            async with db.execute(query, params) as cursor:
                async for row in cursor:
                    # Parse document
                    document = Document(
                        id=row[0],
                        title=row[1],
                        content=row[2],
                        metadata=json.loads(row[3]) if row[3] else {},
                        epic_page_id=row[4],
                        epic_path=row[5],
                        created_at=datetime.fromisoformat(row[6]),
                        updated_at=datetime.fromisoformat(row[7]),
                    )
                    documents.append(document)

            return documents

    async def delete_document(self, document_id: str) -> bool:
        """Delete a document by ID."""
        async with aiosqlite.connect(self.db_path) as db:
            # Delete document (will cascade to chunks)
            await db.execute("DELETE FROM documents WHERE id = ?", (document_id,))
            await db.commit()

            # Return success
            return True

    async def update_document(self, document: Document) -> Document:
        """Update an existing document."""
        # Just use save_document as it handles updates
        return await self.save_document(document)

    async def save_chunk(self, chunk: DocumentChunk) -> DocumentChunk:
        """Save a document chunk."""
        async with aiosqlite.connect(self.db_path) as db:
            # Convert metadata to JSON string
            metadata_json = json.dumps(chunk.metadata)

            # Check if chunk exists
            is_new = False
            async with db.execute(
                "SELECT id FROM chunks WHERE id = ?", (chunk.id,)
            ) as cursor:
                if await cursor.fetchone() is None:
                    is_new = True

            # Insert or replace chunk
            await db.execute(
                """
                INSERT OR REPLACE INTO chunks
                (id, document_id, content, metadata, chunk_index, previous_chunk_id, next_chunk_id, relevance_score, vector_id)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    chunk.id,
                    chunk.document_id,
                    chunk.content,
                    metadata_json,
                    chunk.chunk_index,
                    chunk.previous_chunk_id,
                    chunk.next_chunk_id,
                    chunk.relevance_score,
                    getattr(chunk, "vector_id", None),
                ),
            )

            # Update chunk count if it's a new chunk
            if is_new:
                await db.execute(
                    """
                    UPDATE statistics 
                    SET value = CAST(CAST(value AS INTEGER) + 1 AS TEXT),
                        updated_at = datetime('now')
                    WHERE key = 'chunk_count'
                """
                )

            await db.commit()

            return chunk

    async def get_document_chunks(self, document_id: str) -> List[DocumentChunk]:
        """Get all chunks for a document."""
        async with aiosqlite.connect(self.db_path) as db:
            # Get chunks
            chunks = []
            async with db.execute(
                """
                SELECT id, document_id, content, metadata, chunk_index, previous_chunk_id, next_chunk_id, relevance_score, vector_id
                FROM chunks
                WHERE document_id = ?
                ORDER BY chunk_index
            """,
                (document_id,),
            ) as cursor:
                async for row in cursor:
                    # Parse chunk
                    chunk = DocumentChunk(
                        id=row[0],
                        document_id=row[1],
                        content=row[2],
                        metadata=json.loads(row[3]) if row[3] else {},
                        chunk_index=row[4],
                        previous_chunk_id=row[5],
                        next_chunk_id=row[6],
                        relevance_score=row[7],
                    )
                    # Add vector_id if available
                    if row[8]:
                        chunk.vector_id = row[8]

                    chunks.append(chunk)

            return chunks

    async def get_chunk(self, chunk_id: str) -> Optional[DocumentChunk]:
        """Get a specific chunk by ID."""
        async with aiosqlite.connect(self.db_path) as db:
            # Get chunk
            async with db.execute(
                """
                SELECT id, document_id, content, metadata, chunk_index, previous_chunk_id, next_chunk_id, relevance_score, vector_id
                FROM chunks
                WHERE id = ?
            """,
                (chunk_id,),
            ) as cursor:
                row = await cursor.fetchone()

                if not row:
                    return None

                # Parse chunk
                chunk = DocumentChunk(
                    id=row[0],
                    document_id=row[1],
                    content=row[2],
                    metadata=json.loads(row[3]) if row[3] else {},
                    chunk_index=row[4],
                    previous_chunk_id=row[5],
                    next_chunk_id=row[6],
                    relevance_score=row[7],
                )
                # Add vector_id if available
                if row[8]:
                    chunk.vector_id = row[8]

                return chunk

    async def get_all_chunks(self, limit: int = 10000) -> List[DocumentChunk]:
        """Get all chunks from all documents.

        Args:
            limit: Maximum number of chunks to return

        Returns:
            List of all document chunks
        """
        async with aiosqlite.connect(self.db_path) as db:
            # Get all chunks
            chunks = []
            async with db.execute(
                """
                SELECT id, document_id, content, metadata, chunk_index, previous_chunk_id, next_chunk_id, relevance_score, vector_id
                FROM chunks
                ORDER BY document_id, chunk_index
                LIMIT ?
                """,
                (limit,),
            ) as cursor:
                async for row in cursor:
                    # Parse chunk
                    chunk = DocumentChunk(
                        id=row[0],
                        document_id=row[1],
                        content=row[2],
                        metadata=json.loads(row[3]) if row[3] else {},
                        chunk_index=row[4],
                        previous_chunk_id=row[5],
                        next_chunk_id=row[6],
                        relevance_score=row[7],
                    )
                    # Add vector_id if available
                    if row[8]:
                        chunk.vector_id = row[8]

                    chunks.append(chunk)

            return chunks

    async def find_document_by_epic_page_id(
        self, epic_page_id: str
    ) -> Optional[Document]:
        """Find a document by its Epic page ID.

        Args:
            epic_page_id: The Epic page ID to search for

        Returns:
            Document object if found, None otherwise
        """
        logger.debug(f"Finding document by epic_page_id: {epic_page_id}")
        async with aiosqlite.connect(self.db_path) as db:
            async with db.execute(
                "SELECT id FROM documents WHERE epic_page_id = ?", (epic_page_id,)
            ) as cursor:
                row = await cursor.fetchone()
                if not row:
                    logger.debug(f"No document found with epic_page_id: {epic_page_id}")
                    return None

                logger.debug(
                    f"Found document with id: {row[0]} for epic_page_id: {epic_page_id}"
                )
                return await self.get_document(row[0])

    async def replace_document(self, document: Document) -> Document:
        """Replace a document by its Epic page ID, deleting all existing chunks.

        Args:
            document: The document to replace an existing one

        Returns:
            The saved document
        """
        if not document.epic_page_id:
            logger.warning("Cannot replace document without epic_page_id")
            return await self.save_document(document)

        logger.info(f"Replacing document with epic_page_id: {document.epic_page_id}")

        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("BEGIN TRANSACTION")
            try:
                # Find existing document by epic_page_id
                existing_doc = None
                async with db.execute(
                    "SELECT id FROM documents WHERE epic_page_id = ?",
                    (document.epic_page_id,),
                ) as cursor:
                    row = await cursor.fetchone()
                    if row:
                        existing_doc_id = row[0]

                        # Count chunks for statistics update
                        async with db.execute(
                            "SELECT COUNT(*) FROM chunks WHERE document_id = ?",
                            (existing_doc_id,),
                        ) as count_cursor:
                            chunk_count = (await count_cursor.fetchone())[0]

                        logger.info(
                            f"Found existing document with id: {existing_doc_id} and {chunk_count} chunks"
                        )

                        # Delete the document (will cascade to chunks with ON DELETE CASCADE)
                        await db.execute(
                            "DELETE FROM documents WHERE id = ?", (existing_doc_id,)
                        )

                        # Update statistics
                        await db.execute(
                            """
                            UPDATE statistics
                            SET value = CAST(CAST(value AS INTEGER) - ? AS TEXT),
                                updated_at = datetime('now')
                            WHERE key = 'chunk_count'
                            """,
                            (chunk_count,),
                        )

                        # Update statistics for document count
                        await db.execute(
                            """
                            UPDATE statistics
                            SET value = CAST(CAST(value AS INTEGER) - 1 AS TEXT),
                                updated_at = datetime('now')
                            WHERE key = 'document_count'
                            """
                        )

                        logger.info(
                            f"Deleted existing document with id: {existing_doc_id}"
                        )

                # Save new document with fresh ID
                saved_doc = await self.save_document(document)
                await db.commit()
                logger.info(f"Saved replacement document with id: {saved_doc.id}")
                return saved_doc

            except Exception as e:
                logger.error(f"Error replacing document: {str(e)}")
                await db.execute("ROLLBACK")
                raise e

    async def find_orphaned_chunks(self) -> List[str]:
        """Find chunks that don't have a parent document.

        Returns:
            List of chunk IDs that are orphaned
        """
        logger.debug("Finding orphaned chunks")
        orphaned_chunks = []

        async with aiosqlite.connect(self.db_path) as db:
            # Find orphaned chunks - chunks where the document_id doesn't exist in documents table
            async with db.execute(
                """
                SELECT c.id FROM chunks c 
                LEFT JOIN documents d ON c.document_id = d.id 
                WHERE d.id IS NULL
            """
            ) as cursor:
                async for row in cursor:
                    orphaned_chunks.append(row[0])

        logger.info(f"Found {len(orphaned_chunks)} orphaned chunks")
        return orphaned_chunks

    async def delete_orphaned_chunks(self) -> int:
        """Delete chunks that don't have a parent document.

        Returns:
            Number of chunks deleted
        """
        logger.info("Deleting orphaned chunks")
        orphaned_chunks = await self.find_orphaned_chunks()

        if not orphaned_chunks:
            logger.info("No orphaned chunks found")
            return 0

        async with aiosqlite.connect(self.db_path) as db:
            # Delete orphaned chunks in a single transaction for performance
            await db.execute("BEGIN TRANSACTION")
            try:
                for chunk_id in orphaned_chunks:
                    await db.execute("DELETE FROM chunks WHERE id = ?", (chunk_id,))

                # Update statistics
                await db.execute(
                    """
                    UPDATE statistics
                    SET value = (SELECT COUNT(*) FROM chunks),
                        updated_at = datetime('now')
                    WHERE key = 'chunk_count'
                """
                )

                await db.commit()
                logger.info(
                    f"Successfully deleted {len(orphaned_chunks)} orphaned chunks"
                )
                return len(orphaned_chunks)

            except Exception as e:
                logger.error(f"Error deleting orphaned chunks: {str(e)}")
                await db.execute("ROLLBACK")
                raise e

    async def vacuum_database(self) -> Tuple[float, float]:
        """Run vacuum operation on the database to reclaim unused space.

        Returns:
            Tuple of (size_before_mb, size_after_mb)
        """
        logger.info("Running VACUUM operation on database")

        async with aiosqlite.connect(self.db_path) as db:
            # Get database size before vacuum
            async with db.execute("PRAGMA page_count") as cursor:
                page_count_before = (await cursor.fetchone())[0]

            async with db.execute("PRAGMA page_size") as cursor:
                page_size = (await cursor.fetchone())[0]

            size_before = page_count_before * page_size / (1024 * 1024)  # in MB
            logger.info(f"Database size before vacuum: {size_before:.2f} MB")

            # Run vacuum
            await db.execute("VACUUM")

            # Get database size after vacuum
            async with db.execute("PRAGMA page_count") as cursor:
                page_count_after = (await cursor.fetchone())[0]

            size_after = page_count_after * page_size / (1024 * 1024)  # in MB
            logger.info(f"Database size after vacuum: {size_after:.2f} MB")
            logger.info(
                f"Space saved: {size_before - size_after:.2f} MB ({((size_before - size_after) / size_before) * 100:.2f}%)"
            )

            return (size_before, size_after)

    async def get_statistics(self) -> Dict[str, Any]:
        """Get repository statistics."""
        logger.debug("Getting repository statistics")
        async with aiosqlite.connect(self.db_path) as db:
            stats = {}

            # Get statistics from the statistics table
            async with db.execute(
                "SELECT key, value, updated_at FROM statistics"
            ) as cursor:
                async for row in cursor:
                    key, value, updated_at = row
                    # Try to convert numeric values
                    try:
                        if "." in value:
                            value = float(value)
                        else:
                            value = int(value)
                    except (ValueError, TypeError):
                        pass

                    stats[key] = {"value": value, "updated_at": updated_at}

            # Get additional statistics
            # Count total content size
            async with db.execute(
                "SELECT SUM(LENGTH(content)) FROM documents"
            ) as cursor:
                row = await cursor.fetchone()
                if row and row[0]:
                    stats["total_content_size"] = {
                        "value": row[0],
                        "updated_at": datetime.now().isoformat(),
                    }

            # Get average chunk size
            async with db.execute("SELECT AVG(LENGTH(content)) FROM chunks") as cursor:
                row = await cursor.fetchone()
                if row and row[0]:
                    stats["avg_chunk_size"] = {
                        "value": int(row[0]),
                        "updated_at": datetime.now().isoformat(),
                    }

            # Get database file size
            import os

            if os.path.exists(self.db_path):
                file_size = os.path.getsize(self.db_path)
                stats["database_size"] = {
                    "value": file_size,
                    "updated_at": datetime.now().isoformat(),
                }

            # Get document replacement metrics
            async with db.execute(
                "SELECT COUNT(*) FROM documents WHERE epic_page_id IS NOT NULL"
            ) as cursor:
                row = await cursor.fetchone()
                if row:
                    stats["replaceable_documents"] = {
                        "value": row[0],
                        "updated_at": datetime.now().isoformat(),
                    }

            return stats

    async def save_query(
        self,
        query_text: str,
        transformed_query: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Save a query to the query history.

        Args:
            query_text: The original query text
            transformed_query: Optional transformed query text
            metadata: Optional metadata about the query (results, etc)

        Returns:
            The ID of the saved query
        """
        from ...domain.models.ident import new_id

        # Generate a new ID
        query_id = new_id("query")

        # Convert metadata to JSON string if present
        metadata_json = json.dumps(metadata) if metadata else None

        # Get current timestamp
        timestamp = datetime.now().isoformat()

        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                """
                INSERT INTO query_history
                (id, query_text, transformed_query, timestamp, metadata)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    query_id,
                    query_text,
                    transformed_query,
                    timestamp,
                    metadata_json,
                ),
            )
            await db.commit()

            return query_id

    async def get_query_history(
        self,
        limit: int = 100,
        offset: int = 0,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> List[Dict[str, Any]]:
        """Get query history with optional date filtering.

        Args:
            limit: Maximum number of queries to return
            offset: Offset for pagination
            start_date: Optional start date for filtering
            end_date: Optional end date for filtering

        Returns:
            List of query history records
        """
        async with aiosqlite.connect(self.db_path) as db:
            # Build query based on date filters
            query = """
                SELECT id, query_text, transformed_query, timestamp, metadata
                FROM query_history
            """
            params = []

            # Add date filters if provided
            where_clauses = []
            if start_date:
                where_clauses.append("timestamp >= ?")
                params.append(start_date.isoformat())

            if end_date:
                where_clauses.append("timestamp <= ?")
                params.append(end_date.isoformat())

            if where_clauses:
                query += " WHERE " + " AND ".join(where_clauses)

            # Add limit and offset
            query += " ORDER BY timestamp DESC LIMIT ? OFFSET ?"
            params.extend([limit, offset])

            # Execute query
            results = []
            async with db.execute(query, params) as cursor:
                async for row in cursor:
                    # Parse query history record
                    record = {
                        "id": row[0],
                        "query_text": row[1],
                        "transformed_query": row[2],
                        "timestamp": row[3],
                        "metadata": json.loads(row[4]) if row[4] else {},
                    }
                    results.append(record)

            return results
