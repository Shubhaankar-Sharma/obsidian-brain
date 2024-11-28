# Standard library imports first
import asyncio
import json
import logging
import os
import pickle
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional

# Third-party imports
import lancedb
import nltk
import numpy as np
from langchain_community.vectorstores import LanceDB
from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from rank_bm25 import BM25Okapi
from watchfiles import awatch

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

model_name = "BAAI/bge-large-en"
model_kwargs = {"device": "cpu"}
encode_kwargs = {"normalize_embeddings": True}

hf = HuggingFaceBgeEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)

@dataclass
class BM25State:
    documents: List[str]
    document_metadata: List[Dict[str, Any]]
    tokenized_documents: List[List[str]]
  
@dataclass
class SearchResult:
    content: str
    metadata: Dict[str, Any]
    score: float
    source: str  # 'vector' or 'keyword'

class EnhancedMind:
    def __init__(self, mind_path: str):
        self.mind_path = Path(mind_path)
        if not self.mind_path.exists():
            raise ValueError(f"Mind path does not exist: {mind_path}")

        self.index_path = Path("./index")
        self.bm25_path = self.index_path / "bm25_state.pkl"

        self.index_path.mkdir(parents=True, exist_ok=True)

        self.db = lancedb.connect(self.index_path)
        self.store = LanceDB(self.db, table_name="mind", embedding=hf)
        
        self.documents: List[str] = []
        self.document_metadata: List[Dict[str, Any]] = []
        self.bm25: Optional[BM25Okapi] = None
        self.tokenized_documents: List[List[str]] = []
        
        self._load_bm25_state()
        if self.tokenized_documents:
            self.bm25 = BM25Okapi(self.tokenized_documents)

        self.executor = ThreadPoolExecutor(max_workers=4)
        
        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)
        self.stop_words = set(stopwords.words('english'))

        self._processing_lock = asyncio.Lock()
        
        logger.info(f"Initialized EnhancedMind with path: {mind_path}")

    def preprocess_text(self, text: str) -> List[str]:
        tokens = word_tokenize(text.lower())
        return [token for token in tokens if token not in self.stop_words 
                and token.isalnum()]

    async def process_file(self, file_path: str):
        async with self._processing_lock:
            try:
                logger.info(f"Processing file: {file_path}")

                docs = UnstructuredMarkdownLoader(file_path).load()

                for doc in docs:
                    doc.metadata.update({
                        "title": Path(file_path).stem,
                        "path": file_path,
                        "last_modified": datetime.now().isoformat(),
                        "file_size": os.path.getsize(file_path)
                    })
                    
                    preprocessed_text = self.preprocess_text(doc.page_content)
                    self.documents.append(doc.page_content)
                    self.document_metadata.append(doc.metadata)
                    self.tokenized_documents.append(preprocessed_text)
                
                self.bm25 = BM25Okapi(self.tokenized_documents)
                await asyncio.to_thread(self.store.add_documents, docs)
                self._save_bm25_state()

                logger.info(f"Successfully processed file: {file_path}")
                
            except Exception as e:
                logger.error("WWTFFDSOIHFSDOIHFS?????")
                logger.error(f"Error processing file {file_path}: {e}")
                import traceback
                logger.error(f"Traceback:\n{''.join(traceback.format_tb(e.__traceback__))}")
                raise

    def _load_bm25_state(self):
        try:
            if self.bm25_path.exists():
                logger.info("Loading BM25 state from disk")
                with open(self.bm25_path, 'rb') as f:
                    state: BM25State = pickle.load(f)
                    self.documents = state.documents
                    self.document_metadata = state.document_metadata
                    self.tokenized_documents = state.tokenized_documents
                logger.info(f"Loaded {len(self.documents)} documents from BM25 state")
        except Exception as e:
            logger.error(f"Error loading BM25 state: {e}")
            self.documents = []
            self.document_metadata = []
            self.tokenized_documents = []

    def _save_bm25_state(self):
        try:
            logger.info("Saving BM25 state to disk")
            state = BM25State(
                documents=self.documents,
                document_metadata=self.document_metadata,
                tokenized_documents=self.tokenized_documents
            )
            with open(self.bm25_path, 'wb') as f:
                pickle.dump(state, f)
            logger.info("BM25 state saved successfully")
        except Exception as e:
            logger.error(f"Error saving BM25 state: {e}")

    async def hybrid_search(
        self, 
        query: str, 
        k: int = 10,
        vector_weight: float = 0.5
    ) -> List[SearchResult]:
        try:
            # vector search
            vector_results = await self.store.asimilarity_search(query, k=k)
            
            preprocessed_query = self.preprocess_text(query)
            bm25_scores = self.bm25.get_scores(preprocessed_query)
            
            max_bm25 = max(bm25_scores) if any(bm25_scores) else 1
            bm25_scores = [score/max_bm25 for score in bm25_scores]

            hybrid_results = []
            
            for doc in vector_results:
                score = doc.metadata.get('score', 0.0) * vector_weight
                hybrid_results.append(SearchResult(
                    content=doc.page_content,
                    metadata=doc.metadata,
                    score=score,
                    source='vector'
                ))
            
            keyword_weight = 1 - vector_weight
            top_bm25_indices = np.argsort(bm25_scores)[-k:][::-1]
            for idx in top_bm25_indices:
                if bm25_scores[idx] > 0:
                    score = bm25_scores[idx] * keyword_weight
                    hybrid_results.append(SearchResult(
                        content=self.documents[idx],
                        metadata=self.document_metadata[idx],
                        score=score,
                        source='keyword'
                    ))

            seen_contents = set()
            unique_results = []
            for result in sorted(hybrid_results, key=lambda x: x.score, reverse=True):
                content_hash = hash(result.content)
                if content_hash not in seen_contents:
                    seen_contents.add(content_hash)
                    unique_results.append(result)
            
            return unique_results[:k]
            
        except Exception as e:
            logger.error(f"Error in hybrid search: {e}")
            raise

    async def add_note(
        self, 
        title: str, 
        content: str, 
        tags: List[str] = None,
        folder: str = None
    ) -> str:
        """
        Add a new note to the Obsidian vault.
        """
        try:
            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
            filename = f"{timestamp}-{title.lower().replace(' ', '-')}.md"
            
            if folder:
                folder_path = self.mind_path / folder
                folder_path.mkdir(parents=True, exist_ok=True)
                filepath = folder_path / filename
            else:
                filepath = self.mind_path / filename
            
            frontmatter = "---\n"
            frontmatter += f"title: {title}\n"
            frontmatter += f"date: {datetime.now().isoformat()}\n"
            if tags:
                frontmatter += f"tags: {json.dumps(tags)}\n"
            frontmatter += "---\n\n"
            
            full_content = frontmatter + content
            filepath.write_text(full_content)
            
            # Process the new file
            # watcher should do it
            # await self.process_file(str(filepath))
            
            logger.info(f"Successfully added note: {filepath}")
            return str(filepath)
            
        except Exception as e:
            logger.error(f"Error adding note: {e}")
            raise

    async def init_watcher(self):
        """Initialize and run the file system watcher using watchfiles."""
        try:
            logger.info(f"Starting the watcher for {self.mind_path}")
            last_processed = {}
            
            async for changes in awatch(str(self.mind_path), recursive=True):
                for change_type, path in changes:
                    if not path.endswith('.md'):
                        continue
                        
                    current_time = time.time()
                    
                    # Simple debouncing
                    if path in last_processed:
                        if current_time - last_processed[path] < 1.0:  # 1 second debounce
                            continue
                    
                    last_processed[path] = current_time
                    
                    try:
                        logger.info(f"Processing file from watcher: {path}")
                        await self.process_file(path)
                    except Exception as e:
                        logger.error(f"Error processing {path}: {e}")
                        
        except asyncio.CancelledError:
            logger.info("Watcher stopped - received cancellation")
            raise
        except Exception as e:
            logger.error(f"Watcher error: {e}")
            raise


    async def process_files(self):
        logger.info("Starting initial file processing")
        for root, _, files in os.walk(self.mind_path):
            for file in files:
                if file.endswith(".md"):
                    file_path = os.path.join(root, file)
                    await self.process_file(file_path)
        logger.info("Completed initial file processing")
