"""
AIDocumentIndexer - RAPTOR Retriever Service
==============================================

Implements RAPTOR (Recursive Abstractive Processing for Tree-Organized Retrieval).

RAPTOR Architecture (ICLR 2024):
- Build hierarchical tree of document summaries
- Bottom-up: Cluster chunks, summarize clusters
- Recursive: Summarize summaries at higher levels
- Top-down query: Start from high-level, drill into relevant branches

Key Insight:
Standard RAG retrieves flat chunks, missing global context.
RAPTOR builds a tree where:
- Leaves = original chunks
- Internal nodes = summaries of child clusters
- Root = document-level summary

Benefits:
- 20% accuracy improvement on QuALITY benchmark
- Captures both local details and global themes
- Better for multi-hop reasoning queries

Research:
- "RAPTOR: Recursive Abstractive Processing for Tree-Organized Retrieval" (ICLR 2024)
- Stanford University
"""

import asyncio
import hashlib
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import structlog

from backend.core.config import settings
from backend.core.performance import gather_with_concurrency, LRUCache

logger = structlog.get_logger(__name__)


# =============================================================================
# Configuration
# =============================================================================

class ClusteringMethod(str, Enum):
    """Methods for clustering chunks."""
    KMEANS = "kmeans"         # K-means on embeddings
    HIERARCHICAL = "hierarchical"  # Agglomerative
    GMM = "gmm"               # Gaussian Mixture Model
    SEMANTIC = "semantic"     # Semantic similarity threshold


class TraversalStrategy(str, Enum):
    """Strategies for traversing the RAPTOR tree."""
    TOP_DOWN = "top_down"     # Start from root, drill down
    BOTTOM_UP = "bottom_up"   # Start from leaves, expand
    HYBRID = "hybrid"         # Both simultaneously


@dataclass
class RAPTORConfig:
    """Configuration for RAPTOR retriever."""
    # Tree structure
    max_levels: int = 3           # Maximum tree depth
    cluster_size: int = 10        # Target chunks per cluster
    min_cluster_size: int = 3     # Minimum cluster size
    max_cluster_size: int = 15    # Maximum cluster size

    # Clustering
    clustering_method: ClusteringMethod = ClusteringMethod.SEMANTIC
    similarity_threshold: float = 0.7  # For semantic clustering

    # Summarization
    summary_model: str = "gpt-4o-mini"
    summary_provider: str = "openai"
    summary_max_tokens: int = 500

    # Retrieval
    traversal_strategy: TraversalStrategy = TraversalStrategy.TOP_DOWN
    top_k_per_level: int = 5      # Branches to explore per level
    final_top_k: int = 10         # Final results

    # Caching
    enable_cache: bool = True
    cache_ttl_days: int = 7


@dataclass(slots=True)
class RAPTORNode:
    """Node in the RAPTOR tree."""
    id: str
    content: str                  # Summary or original chunk text
    embedding: Optional[List[float]] = None
    level: int = 0                # 0 = leaf (original chunk)
    parent_id: Optional[str] = None
    children_ids: List[str] = field(default_factory=list)
    chunk_ids: List[str] = field(default_factory=list)  # Original chunks covered
    document_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RAPTORTree:
    """Complete RAPTOR tree structure."""
    document_id: str
    root_id: str
    nodes: Dict[str, RAPTORNode]
    num_levels: int
    leaf_count: int
    created_at: float = field(default_factory=time.time)

    def get_level_nodes(self, level: int) -> List[RAPTORNode]:
        """Get all nodes at a specific level."""
        return [n for n in self.nodes.values() if n.level == level]

    def get_children(self, node_id: str) -> List[RAPTORNode]:
        """Get children of a node."""
        node = self.nodes.get(node_id)
        if not node:
            return []
        return [self.nodes[cid] for cid in node.children_ids if cid in self.nodes]

    def get_ancestors(self, node_id: str) -> List[RAPTORNode]:
        """Get all ancestors from node to root."""
        ancestors = []
        current = self.nodes.get(node_id)
        while current and current.parent_id:
            parent = self.nodes.get(current.parent_id)
            if parent:
                ancestors.append(parent)
                current = parent
            else:
                break
        return ancestors


@dataclass(slots=True)
class RAPTORResult:
    """Result from RAPTOR retrieval."""
    chunk_id: str
    document_id: str
    content: str
    score: float
    level: int
    path: List[str] = field(default_factory=list)  # Node IDs from root to leaf
    context_summary: Optional[str] = None  # Summary from parent nodes
    metadata: Dict[str, Any] = field(default_factory=dict)
    # Document metadata for proper citation
    document_filename: Optional[str] = None
    document_title: Optional[str] = None
    page_number: Optional[int] = None
    section_title: Optional[str] = None


# =============================================================================
# Prompts
# =============================================================================

CLUSTER_SUMMARY_PROMPT = """Summarize these related text passages into a coherent summary.
Capture the main themes, key facts, and important relationships.

Passages:
{passages}

Write a comprehensive summary (max {max_tokens} words):"""

RELEVANCE_PROMPT = """Given this query and summary, rate relevance 1-10.

Query: {query}

Summary: {summary}

Relevance score (1-10):"""


# =============================================================================
# RAPTOR Tree Builder
# =============================================================================

class RAPTORTreeBuilder:
    """
    Builds RAPTOR trees from document chunks.

    Process:
    1. Start with leaf nodes (original chunks)
    2. Cluster similar chunks using embeddings
    3. Generate summary for each cluster
    4. Recursively cluster and summarize until root
    """

    def __init__(
        self,
        config: Optional[RAPTORConfig] = None,
        embedding_service=None,
    ):
        self.config = config or RAPTORConfig()
        self.embedding_service = embedding_service
        self._llm = None
        self._node_counter = 0
        self._initialized = False

    async def initialize(self) -> bool:
        """Initialize services."""
        if self._initialized:
            return True

        try:
            from backend.services.llm import LLMFactory

            self._llm = LLMFactory.get_chat_model(
                provider=self.config.summary_provider,
                model=self.config.summary_model,
                temperature=0.3,
                max_tokens=self.config.summary_max_tokens,
            )

            if not self.embedding_service:
                from backend.services.embeddings import get_embedding_service
                self.embedding_service = get_embedding_service()

            self._initialized = True
            return True

        except Exception as e:
            logger.error("Failed to initialize RAPTOR builder", error=str(e))
            return False

    async def build_tree(
        self,
        chunks: List[Dict[str, Any]],
        document_id: str,
    ) -> RAPTORTree:
        """
        Build RAPTOR tree from document chunks.

        Args:
            chunks: List of dicts with 'id', 'content', 'embedding' (optional)
            document_id: Document identifier

        Returns:
            RAPTORTree structure
        """
        if not await self.initialize():
            raise RuntimeError("RAPTOR builder not initialized")

        start_time = time.time()
        self._node_counter = 0

        logger.info(
            "Building RAPTOR tree",
            document_id=document_id,
            num_chunks=len(chunks),
        )

        nodes: Dict[str, RAPTORNode] = {}

        # Level 0: Create leaf nodes from chunks
        leaf_nodes = []
        for chunk in chunks:
            node = self._create_node(
                content=chunk.get('content', ''),
                embedding=chunk.get('embedding'),
                level=0,
                document_id=document_id,
                chunk_ids=[chunk.get('id', '')],
            )
            nodes[node.id] = node
            leaf_nodes.append(node)

        # Ensure embeddings for all leaves
        await self._ensure_embeddings(leaf_nodes)

        # Build tree levels
        current_level_nodes = leaf_nodes
        level = 0

        while len(current_level_nodes) > 1 and level < self.config.max_levels - 1:
            level += 1

            # Cluster nodes
            clusters = await self._cluster_nodes(current_level_nodes)

            # Create parent nodes for each cluster
            next_level_nodes = []

            for cluster in clusters:
                if len(cluster) == 0:
                    continue

                # Generate summary for cluster
                summary = await self._summarize_cluster(cluster)

                # Create parent node
                child_ids = [n.id for n in cluster]
                all_chunk_ids = []
                for n in cluster:
                    all_chunk_ids.extend(n.chunk_ids)

                parent = self._create_node(
                    content=summary,
                    level=level,
                    document_id=document_id,
                    children_ids=child_ids,
                    chunk_ids=all_chunk_ids,
                )

                # Update children's parent
                for child in cluster:
                    child.parent_id = parent.id
                    nodes[child.id] = child

                nodes[parent.id] = parent
                next_level_nodes.append(parent)

            # Ensure embeddings for new level
            await self._ensure_embeddings(next_level_nodes)

            current_level_nodes = next_level_nodes
            logger.debug(f"Built level {level} with {len(current_level_nodes)} nodes")

        # Root is the last remaining node(s)
        if current_level_nodes:
            root_id = current_level_nodes[0].id
        else:
            root_id = leaf_nodes[0].id if leaf_nodes else ""

        elapsed = (time.time() - start_time) * 1000

        logger.info(
            "RAPTOR tree built",
            document_id=document_id,
            num_levels=level + 1,
            total_nodes=len(nodes),
            elapsed_ms=round(elapsed, 2),
        )

        return RAPTORTree(
            document_id=document_id,
            root_id=root_id,
            nodes=nodes,
            num_levels=level + 1,
            leaf_count=len(leaf_nodes),
        )

    def _create_node(
        self,
        content: str,
        level: int,
        document_id: str,
        embedding: Optional[List[float]] = None,
        children_ids: Optional[List[str]] = None,
        chunk_ids: Optional[List[str]] = None,
    ) -> RAPTORNode:
        """Create a new RAPTOR node."""
        self._node_counter += 1
        node_id = f"raptor_{document_id[:8]}_{level}_{self._node_counter}"

        return RAPTORNode(
            id=node_id,
            content=content,
            embedding=embedding,
            level=level,
            document_id=document_id,
            children_ids=children_ids or [],
            chunk_ids=chunk_ids or [],
        )

    async def _ensure_embeddings(self, nodes: List[RAPTORNode]) -> None:
        """Ensure all nodes have embeddings."""
        nodes_needing_embeddings = [n for n in nodes if not n.embedding]

        if not nodes_needing_embeddings:
            return

        texts = [n.content for n in nodes_needing_embeddings]
        embeddings = await self.embedding_service.embed_texts(texts)

        for node, emb in zip(nodes_needing_embeddings, embeddings):
            node.embedding = emb

    async def _cluster_nodes(
        self,
        nodes: List[RAPTORNode],
    ) -> List[List[RAPTORNode]]:
        """Cluster nodes based on embedding similarity."""
        if len(nodes) <= self.config.cluster_size:
            return [nodes]

        if self.config.clustering_method == ClusteringMethod.SEMANTIC:
            return await self._semantic_clustering(nodes)
        else:
            # Fall back to simple chunking
            return self._simple_chunking(nodes)

    async def _semantic_clustering(
        self,
        nodes: List[RAPTORNode],
    ) -> List[List[RAPTORNode]]:
        """Cluster by semantic similarity using embeddings."""
        import numpy as np

        # Get embeddings
        embeddings = np.array([n.embedding for n in nodes if n.embedding])

        if len(embeddings) == 0:
            return [nodes]

        # Compute similarity matrix
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        normalized = embeddings / (norms + 1e-10)
        similarity_matrix = np.dot(normalized, normalized.T)

        # Greedy clustering based on similarity threshold
        clusters = []
        assigned = set()

        for i, node in enumerate(nodes):
            if i in assigned:
                continue

            cluster = [node]
            assigned.add(i)

            # Find similar nodes
            for j, other_node in enumerate(nodes):
                if j in assigned:
                    continue
                if j < len(similarity_matrix) and similarity_matrix[i][j] >= self.config.similarity_threshold:
                    if len(cluster) < self.config.max_cluster_size:
                        cluster.append(other_node)
                        assigned.add(j)

            clusters.append(cluster)

        # Merge small clusters
        final_clusters = []
        small_nodes = []

        for cluster in clusters:
            if len(cluster) < self.config.min_cluster_size:
                small_nodes.extend(cluster)
            else:
                final_clusters.append(cluster)

        # Add remaining small nodes to existing clusters or create new one
        if small_nodes:
            if final_clusters:
                for node in small_nodes:
                    # Add to smallest cluster
                    smallest = min(final_clusters, key=len)
                    smallest.append(node)
            else:
                final_clusters.append(small_nodes)

        return final_clusters

    def _simple_chunking(
        self,
        nodes: List[RAPTORNode],
    ) -> List[List[RAPTORNode]]:
        """Simple sequential chunking."""
        clusters = []
        for i in range(0, len(nodes), self.config.cluster_size):
            clusters.append(nodes[i:i + self.config.cluster_size])
        return clusters

    async def _summarize_cluster(
        self,
        cluster: List[RAPTORNode],
    ) -> str:
        """Generate summary for a cluster of nodes."""
        from langchain_core.messages import HumanMessage

        passages = "\n\n---\n\n".join(
            f"Passage {i+1}:\n{node.content[:500]}"
            for i, node in enumerate(cluster[:10])  # Limit passages
        )

        prompt = CLUSTER_SUMMARY_PROMPT.format(
            passages=passages,
            max_tokens=self.config.summary_max_tokens,
        )

        try:
            response = await self._llm.ainvoke([HumanMessage(content=prompt)])
            return response.content.strip()
        except Exception as e:
            logger.warning("Cluster summarization failed", error=str(e))
            # Fall back to first few sentences of first node
            return cluster[0].content[:300] if cluster else ""


# =============================================================================
# RAPTOR Retriever
# =============================================================================

class RAPTORRetriever:
    """
    RAPTOR retriever for hierarchical document search.

    Traverses the RAPTOR tree to find relevant chunks,
    using summaries at higher levels to guide search.

    Usage:
        retriever = RAPTORRetriever()

        # Build tree (once per document)
        tree = await retriever.build_tree(chunks, document_id)

        # Query
        results = await retriever.retrieve(
            query="What are the main findings?",
            tree=tree,
        )
    """

    def __init__(
        self,
        config: Optional[RAPTORConfig] = None,
        embedding_service=None,
    ):
        self.config = config or RAPTORConfig()
        self.embedding_service = embedding_service
        self._builder = RAPTORTreeBuilder(config, embedding_service)
        self._trees: Dict[str, RAPTORTree] = {}
        self._cache = LRUCache[RAPTORTree](capacity=100)
        self._initialized = False

    async def initialize(self) -> bool:
        """Initialize retriever."""
        if self._initialized:
            return True

        try:
            await self._builder.initialize()

            if not self.embedding_service:
                from backend.services.embeddings import get_embedding_service
                self.embedding_service = get_embedding_service()

            self._initialized = True
            return True

        except Exception as e:
            logger.error("Failed to initialize RAPTOR retriever", error=str(e))
            return False

    async def build_tree(
        self,
        chunks: List[Dict[str, Any]],
        document_id: str,
    ) -> RAPTORTree:
        """Build and cache a RAPTOR tree."""
        if not await self.initialize():
            raise RuntimeError("RAPTOR retriever not initialized")

        tree = await self._builder.build_tree(chunks, document_id)

        # Cache tree
        self._trees[document_id] = tree
        await self._cache.set(document_id, tree)

        return tree

    async def retrieve(
        self,
        query: str,
        tree: Optional[RAPTORTree] = None,
        document_id: Optional[str] = None,
        top_k: Optional[int] = None,
        organization_id: Optional[str] = None,
        is_superadmin: bool = False,
    ) -> List[RAPTORResult]:
        """
        Retrieve relevant chunks using RAPTOR tree.

        Args:
            query: Search query
            tree: RAPTOR tree (or provide document_id)
            document_id: Document ID to look up tree
            top_k: Number of results

        Returns:
            List of RAPTORResult with hierarchical context
        """
        if not await self.initialize():
            return []

        # Get tree
        if tree is None:
            if document_id:
                tree = self._trees.get(document_id)
                if not tree:
                    tree = await self._cache.get(document_id)
            if not tree:
                logger.warning("No RAPTOR tree found")
                return []

        start_time = time.time()
        final_top_k = top_k or self.config.final_top_k

        logger.info(
            "RAPTOR retrieval",
            query_length=len(query),
            tree_levels=tree.num_levels,
            strategy=self.config.traversal_strategy.value,
        )

        # Get query embedding
        query_embedding = await self.embedding_service.embed_text(query)

        # Traverse based on strategy
        if self.config.traversal_strategy == TraversalStrategy.TOP_DOWN:
            results = await self._top_down_retrieval(
                query, query_embedding, tree, final_top_k
            )
        elif self.config.traversal_strategy == TraversalStrategy.BOTTOM_UP:
            results = await self._bottom_up_retrieval(
                query, query_embedding, tree, final_top_k
            )
        else:
            results = await self._hybrid_retrieval(
                query, query_embedding, tree, final_top_k
            )

        elapsed = (time.time() - start_time) * 1000

        logger.info(
            "RAPTOR retrieval complete",
            results=len(results),
            elapsed_ms=round(elapsed, 2),
        )

        return results

    async def _top_down_retrieval(
        self,
        query: str,
        query_embedding: List[float],
        tree: RAPTORTree,
        top_k: int,
    ) -> List[RAPTORResult]:
        """
        Top-down traversal: Start from root, drill into relevant branches.
        """
        import numpy as np

        def cosine_similarity(a, b):
            a, b = np.array(a), np.array(b)
            return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10)

        results = []
        visited_paths: Dict[str, List[str]] = {}  # node_id -> path

        # Start from root level
        current_level = tree.num_levels - 1
        current_nodes = tree.get_level_nodes(current_level)

        while current_level >= 0 and len(results) < top_k:
            # Score current level nodes
            scored_nodes = []
            for node in current_nodes:
                if node.embedding:
                    score = cosine_similarity(query_embedding, node.embedding)
                    scored_nodes.append((node, score))

            # Sort by score
            scored_nodes.sort(key=lambda x: x[1], reverse=True)

            # Take top-k per level
            top_nodes = scored_nodes[:self.config.top_k_per_level]

            for node, score in top_nodes:
                # Build path
                path = visited_paths.get(node.parent_id, []).copy()
                path.append(node.id)
                visited_paths[node.id] = path

                if node.level == 0:  # Leaf node
                    # Get context from ancestors
                    ancestors = tree.get_ancestors(node.id)
                    context_summary = " -> ".join(
                        a.content[:100] for a in reversed(ancestors)
                    ) if ancestors else None

                    results.append(RAPTORResult(
                        chunk_id=node.chunk_ids[0] if node.chunk_ids else node.id,
                        document_id=node.document_id or "",
                        content=node.content,
                        score=score,
                        level=node.level,
                        path=path,
                        context_summary=context_summary,
                    ))

            # Move to children level
            if current_level > 0:
                # Get children of selected nodes
                next_nodes = []
                for node, _ in top_nodes:
                    children = tree.get_children(node.id)
                    for child in children:
                        if child.id not in visited_paths:
                            visited_paths[child.id] = visited_paths[node.id] + [child.id]
                    next_nodes.extend(children)
                current_nodes = next_nodes

            current_level -= 1

        # Sort final results by score
        results.sort(key=lambda r: r.score, reverse=True)
        return results[:top_k]

    async def _bottom_up_retrieval(
        self,
        query: str,
        query_embedding: List[float],
        tree: RAPTORTree,
        top_k: int,
    ) -> List[RAPTORResult]:
        """
        Bottom-up: Search leaves first, expand with context from parents.
        """
        import numpy as np

        def cosine_similarity(a, b):
            a, b = np.array(a), np.array(b)
            return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10)

        # Score all leaf nodes
        leaves = tree.get_level_nodes(0)
        scored = []

        for leaf in leaves:
            if leaf.embedding:
                score = cosine_similarity(query_embedding, leaf.embedding)
                scored.append((leaf, score))

        # Sort and take top candidates
        scored.sort(key=lambda x: x[1], reverse=True)

        results = []
        for leaf, score in scored[:top_k * 2]:
            # Get ancestors for context
            ancestors = tree.get_ancestors(leaf.id)
            path = [a.id for a in reversed(ancestors)] + [leaf.id]
            context_summary = " -> ".join(
                a.content[:100] for a in reversed(ancestors)
            ) if ancestors else None

            results.append(RAPTORResult(
                chunk_id=leaf.chunk_ids[0] if leaf.chunk_ids else leaf.id,
                document_id=leaf.document_id or "",
                content=leaf.content,
                score=score,
                level=0,
                path=path,
                context_summary=context_summary,
            ))

        return results[:top_k]

    async def _hybrid_retrieval(
        self,
        query: str,
        query_embedding: List[float],
        tree: RAPTORTree,
        top_k: int,
    ) -> List[RAPTORResult]:
        """
        Hybrid: Combine top-down and bottom-up results.
        """
        # Run both strategies
        top_down = await self._top_down_retrieval(
            query, query_embedding, tree, top_k
        )
        bottom_up = await self._bottom_up_retrieval(
            query, query_embedding, tree, top_k
        )

        # Combine using RRF
        from backend.services.hybrid_retriever import reciprocal_rank_fusion

        fused = reciprocal_rank_fusion(
            [
                ("top_down", top_down),
                ("bottom_up", bottom_up),
            ],
            k=60,
            id_extractor=lambda x: x.chunk_id,
            score_extractor=lambda x: x.score,
        )

        # Build results
        result_map = {r.chunk_id: r for r in top_down + bottom_up}
        final_results = []

        for chunk_id, rrf_score, _ in fused[:top_k]:
            if chunk_id in result_map:
                result = result_map[chunk_id]
                result.score = rrf_score
                result.metadata["fusion"] = "raptor_hybrid"
                final_results.append(result)

        return final_results


# =============================================================================
# Singleton and Factory
# =============================================================================

_raptor_retriever: Optional[RAPTORRetriever] = None
_retriever_lock = asyncio.Lock()


async def get_raptor_retriever(
    config: Optional[RAPTORConfig] = None,
) -> RAPTORRetriever:
    """Get or create RAPTOR retriever singleton."""
    global _raptor_retriever

    if _raptor_retriever is not None:
        return _raptor_retriever

    async with _retriever_lock:
        if _raptor_retriever is not None:
            return _raptor_retriever

        _raptor_retriever = RAPTORRetriever(config)
        return _raptor_retriever


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    # Config
    "RAPTORConfig",
    "ClusteringMethod",
    "TraversalStrategy",
    # Data
    "RAPTORNode",
    "RAPTORTree",
    "RAPTORResult",
    # Classes
    "RAPTORTreeBuilder",
    "RAPTORRetriever",
    # Factory
    "get_raptor_retriever",
]
