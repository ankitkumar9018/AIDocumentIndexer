"""
AIDocumentIndexer - LangChain RAG Chains
=========================================

RAG (Retrieval-Augmented Generation) chains for document Q&A
with source citations and conversation memory.
"""

from typing import Any, Dict, List, Optional

import structlog
from langchain_core.documents import Document
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough, RunnableLambda

from backend.services.llm import get_chat_model, get_embeddings

logger = structlog.get_logger(__name__)


# =============================================================================
# Prompt Templates
# =============================================================================

RAG_SYSTEM_PROMPT = """You are an intelligent document assistant for a company with 25+ years of presentations, reports, and strategic documents.

Your role is to:
1. Answer questions based on the provided context from the company's document archive
2. Always cite your sources by referencing the document names
3. Be helpful and professional
4. If you don't have enough information in the context, say so honestly
5. Provide actionable insights when relevant

Context from documents:
{context}

Guidelines:
- Always mention which documents you're referencing
- If multiple documents contain relevant information, synthesize them
- Be concise but comprehensive
- Use bullet points for clarity when listing multiple items
- If asked to create content, draw inspiration from the existing documents"""

RAG_PROMPT = ChatPromptTemplate.from_messages([
    ("system", RAG_SYSTEM_PROMPT),
    MessagesPlaceholder(variable_name="chat_history", optional=True),
    ("human", "{question}"),
])

CONDENSE_QUESTION_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """Given the following conversation and a follow-up question, rephrase the follow-up question to be a standalone question that captures all relevant context from the conversation.

Chat History:
{chat_history}

Follow-up Question: {question}

Standalone Question:"""),
])


# =============================================================================
# RAG Chain Builder
# =============================================================================

class RAGChain:
    """
    RAG chain for document Q&A with source citations.

    Features:
    - Hybrid search (vector + keyword)
    - Conversation memory
    - Source document tracking
    - Multi-language support
    """

    def __init__(
        self,
        retriever: Any,  # Vector store retriever
        llm_provider: Optional[str] = None,
        llm_model: Optional[str] = None,
        temperature: float = 0.7,
    ):
        """
        Initialize RAG chain.

        Args:
            retriever: Document retriever (from vector store)
            llm_provider: LLM provider
            llm_model: LLM model name
            temperature: Response temperature
        """
        self.retriever = retriever
        self.llm = get_chat_model(
            provider=llm_provider,
            model=llm_model,
            temperature=temperature,
        )
        self.chat_history: List[BaseMessage] = []

    def _format_docs(self, docs: List[Document]) -> str:
        """Format retrieved documents for the prompt."""
        formatted = []
        for i, doc in enumerate(docs, 1):
            source = doc.metadata.get("source", "Unknown")
            page = doc.metadata.get("page", "")
            page_info = f" (Page {page})" if page else ""

            formatted.append(f"[Document {i}: {source}{page_info}]\n{doc.page_content}\n")

        return "\n---\n".join(formatted)

    def _extract_sources(self, docs: List[Document]) -> List[Dict[str, Any]]:
        """Extract source information from documents."""
        sources = []
        seen = set()

        for doc in docs:
            source_id = doc.metadata.get("document_id")
            if source_id and source_id not in seen:
                seen.add(source_id)
                sources.append({
                    "document_id": source_id,
                    "filename": doc.metadata.get("filename", "Unknown"),
                    "page": doc.metadata.get("page"),
                    "chunk_id": doc.metadata.get("chunk_id"),
                    "relevance_score": doc.metadata.get("score"),
                })

        return sources

    async def invoke(
        self,
        question: str,
        chat_history: Optional[List[BaseMessage]] = None,
    ) -> Dict[str, Any]:
        """
        Invoke the RAG chain.

        Args:
            question: User question
            chat_history: Optional conversation history

        Returns:
            dict: Response with answer and sources
        """
        history = chat_history or self.chat_history

        # Retrieve relevant documents
        docs = await self.retriever.ainvoke(question)

        if not docs:
            return {
                "answer": "I couldn't find any relevant information in the document archive to answer your question. Could you try rephrasing or asking about a different topic?",
                "sources": [],
                "context_used": False,
            }

        # Format context
        context = self._format_docs(docs)
        sources = self._extract_sources(docs)

        # Build prompt
        prompt = RAG_PROMPT.format_messages(
            context=context,
            chat_history=history,
            question=question,
        )

        # Generate response
        response = await self.llm.ainvoke(prompt)

        # Update chat history
        self.chat_history.append(HumanMessage(content=question))
        self.chat_history.append(AIMessage(content=response.content))

        return {
            "answer": response.content,
            "sources": sources,
            "context_used": True,
            "documents_retrieved": len(docs),
        }

    async def stream(
        self,
        question: str,
        chat_history: Optional[List[BaseMessage]] = None,
    ):
        """
        Stream RAG response.

        Args:
            question: User question
            chat_history: Optional conversation history

        Yields:
            str: Response chunks
        """
        history = chat_history or self.chat_history

        # Retrieve relevant documents
        docs = await self.retriever.ainvoke(question)

        if not docs:
            yield {
                "type": "answer",
                "content": "I couldn't find any relevant information in the document archive.",
            }
            return

        # Format context
        context = self._format_docs(docs)
        sources = self._extract_sources(docs)

        # Yield sources first
        yield {
            "type": "sources",
            "content": sources,
        }

        # Build prompt
        prompt = RAG_PROMPT.format_messages(
            context=context,
            chat_history=history,
            question=question,
        )

        # Stream response
        full_response = ""
        async for chunk in self.llm.astream(prompt):
            content = chunk.content
            full_response += content
            yield {
                "type": "token",
                "content": content,
            }

        # Update chat history
        self.chat_history.append(HumanMessage(content=question))
        self.chat_history.append(AIMessage(content=full_response))

        yield {
            "type": "done",
            "content": full_response,
        }

    def clear_history(self):
        """Clear conversation history."""
        self.chat_history = []


# =============================================================================
# Query-Only Chain (No RAG storage)
# =============================================================================

class QueryOnlyChain:
    """
    Chain for querying uploaded documents without storing them.

    Used when users want to ask questions about a document
    without adding it to the permanent RAG index.
    """

    def __init__(
        self,
        llm_provider: Optional[str] = None,
        llm_model: Optional[str] = None,
    ):
        self.llm = get_chat_model(
            provider=llm_provider,
            model=llm_model,
        )

    async def query_document(
        self,
        document_content: str,
        question: str,
        document_name: str = "Uploaded Document",
    ) -> Dict[str, Any]:
        """
        Query a document without storing it.

        Args:
            document_content: Full document text
            question: User question
            document_name: Name for citation

        Returns:
            dict: Response with answer
        """
        prompt = ChatPromptTemplate.from_messages([
            ("system", f"""You are analyzing a document titled "{document_name}".

Document content:
{document_content}

Answer the user's question based on this document. Be specific and cite relevant sections."""),
            ("human", "{question}"),
        ])

        response = await self.llm.ainvoke(
            prompt.format_messages(question=question)
        )

        return {
            "answer": response.content,
            "source": document_name,
        }


# =============================================================================
# Multi-Document Synthesis Chain
# =============================================================================

class SynthesisChain:
    """
    Chain for synthesizing information from multiple documents.

    Used for creating summaries, comparisons, or new content
    based on multiple source documents.
    """

    def __init__(
        self,
        llm_provider: Optional[str] = None,
        llm_model: Optional[str] = None,
    ):
        self.llm = get_chat_model(
            provider=llm_provider,
            model=llm_model,
        )

    async def synthesize(
        self,
        documents: List[Document],
        task: str,
    ) -> Dict[str, Any]:
        """
        Synthesize information from multiple documents.

        Args:
            documents: List of source documents
            task: What to create/synthesize

        Returns:
            dict: Synthesized content with sources
        """
        # Format documents
        doc_contents = []
        for i, doc in enumerate(documents, 1):
            source = doc.metadata.get("source", f"Document {i}")
            doc_contents.append(f"[{source}]\n{doc.page_content}")

        all_docs = "\n\n---\n\n".join(doc_contents)

        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a creative assistant that synthesizes information from multiple documents.

Source Documents:
{documents}

Guidelines:
- Draw inspiration from all provided documents
- Create original content based on patterns and ideas in the sources
- Always cite which documents influenced each part of your output
- Be creative while staying true to the source material's style and quality"""),
            ("human", "{task}"),
        ])

        response = await self.llm.ainvoke(
            prompt.format_messages(documents=all_docs, task=task)
        )

        sources = [
            {
                "document_id": doc.metadata.get("document_id"),
                "filename": doc.metadata.get("filename"),
            }
            for doc in documents
        ]

        return {
            "content": response.content,
            "sources": sources,
            "task": task,
        }


# =============================================================================
# Factory Functions
# =============================================================================

def create_rag_chain(
    retriever: Any,
    llm_provider: Optional[str] = None,
    llm_model: Optional[str] = None,
) -> RAGChain:
    """Create a RAG chain instance."""
    return RAGChain(
        retriever=retriever,
        llm_provider=llm_provider,
        llm_model=llm_model,
    )


def create_query_chain(
    llm_provider: Optional[str] = None,
    llm_model: Optional[str] = None,
) -> QueryOnlyChain:
    """Create a query-only chain instance."""
    return QueryOnlyChain(
        llm_provider=llm_provider,
        llm_model=llm_model,
    )


def create_synthesis_chain(
    llm_provider: Optional[str] = None,
    llm_model: Optional[str] = None,
) -> SynthesisChain:
    """Create a synthesis chain instance."""
    return SynthesisChain(
        llm_provider=llm_provider,
        llm_model=llm_model,
    )
