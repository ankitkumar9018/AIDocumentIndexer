"""
AIDocumentIndexer - Teams Adaptive Cards
=========================================

Adaptive Card templates for Teams bot responses.
"""

from typing import List, Dict, Any, Optional


def create_help_card() -> Dict[str, Any]:
    """Create help card with available commands."""
    return {
        "$schema": "http://adaptivecards.io/schemas/adaptive-card.json",
        "type": "AdaptiveCard",
        "version": "1.4",
        "body": [
            {
                "type": "TextBlock",
                "text": "📚 AIDocumentIndexer Bot",
                "weight": "Bolder",
                "size": "Large",
            },
            {
                "type": "TextBlock",
                "text": "I can help you search and explore your organization's documents.",
                "wrap": True,
            },
            {
                "type": "TextBlock",
                "text": "Available Commands",
                "weight": "Bolder",
                "size": "Medium",
                "spacing": "Large",
            },
            {
                "type": "FactSet",
                "facts": [
                    {"title": "/help", "value": "Show this help message"},
                    {"title": "/ask <question>", "value": "Ask a question about your documents"},
                    {"title": "/search <query>", "value": "Search for documents"},
                    {"title": "/summarize <doc>", "value": "Get a summary of a document"},
                    {"title": "/documents", "value": "List recent documents"},
                ],
            },
            {
                "type": "TextBlock",
                "text": "💡 Tip: You can also just type a question directly!",
                "wrap": True,
                "spacing": "Large",
                "isSubtle": True,
            },
        ],
        "actions": [
            {
                "type": "Action.Submit",
                "title": "📄 Show Recent Documents",
                "data": {"action": "list_documents"},
            },
        ],
    }


def create_answer_card(
    question: str,
    answer: str,
    sources: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Create answer card with sources."""
    body = [
        {
            "type": "TextBlock",
            "text": "Question",
            "weight": "Bolder",
            "size": "Small",
            "color": "Accent",
        },
        {
            "type": "TextBlock",
            "text": question,
            "wrap": True,
            "weight": "Bolder",
        },
        {
            "type": "TextBlock",
            "text": "Answer",
            "weight": "Bolder",
            "size": "Small",
            "color": "Accent",
            "spacing": "Large",
        },
        {
            "type": "TextBlock",
            "text": answer,
            "wrap": True,
        },
    ]

    # Add sources if available
    if sources and len(sources) > 0:
        body.append({
            "type": "TextBlock",
            "text": "Sources",
            "weight": "Bolder",
            "size": "Small",
            "color": "Accent",
            "spacing": "Large",
        })

        source_items = []
        for i, source in enumerate(sources[:5]):  # Limit to 5 sources
            source_name = source.get("name") or source.get("filename") or f"Source {i + 1}"
            source_items.append({
                "type": "TextBlock",
                "text": f"📄 {source_name}",
                "wrap": True,
                "size": "Small",
            })

        body.extend(source_items)

    actions = [
        {
            "type": "Action.Submit",
            "title": "👍 Helpful",
            "data": {"action": "feedback", "rating": "positive", "question": question},
        },
        {
            "type": "Action.Submit",
            "title": "👎 Not Helpful",
            "data": {"action": "feedback", "rating": "negative", "question": question},
        },
    ]

    return {
        "$schema": "http://adaptivecards.io/schemas/adaptive-card.json",
        "type": "AdaptiveCard",
        "version": "1.4",
        "body": body,
        "actions": actions,
    }


def create_search_results_card(
    query: str,
    results: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Create search results card."""
    body = [
        {
            "type": "TextBlock",
            "text": f"🔍 Search Results for \"{query}\"",
            "weight": "Bolder",
            "size": "Medium",
        },
    ]

    if not results:
        body.append({
            "type": "TextBlock",
            "text": "No documents found matching your query.",
            "wrap": True,
            "isSubtle": True,
        })
    else:
        body.append({
            "type": "TextBlock",
            "text": f"Found {len(results)} document(s)",
            "size": "Small",
            "isSubtle": True,
        })

        # Add result items
        for i, result in enumerate(results[:10]):  # Limit to 10 results
            doc_name = result.get("name") or result.get("filename") or f"Document {i + 1}"
            doc_type = result.get("type") or result.get("file_type") or "Unknown"
            score = result.get("score")

            container = {
                "type": "Container",
                "separator": True,
                "spacing": "Small",
                "items": [
                    {
                        "type": "ColumnSet",
                        "columns": [
                            {
                                "type": "Column",
                                "width": "stretch",
                                "items": [
                                    {
                                        "type": "TextBlock",
                                        "text": f"📄 {doc_name}",
                                        "weight": "Bolder",
                                        "wrap": True,
                                    },
                                    {
                                        "type": "TextBlock",
                                        "text": f"Type: {doc_type}",
                                        "size": "Small",
                                        "isSubtle": True,
                                    },
                                ],
                            },
                        ],
                    },
                ],
                "selectAction": {
                    "type": "Action.Submit",
                    "data": {
                        "action": "view_document",
                        "document_id": result.get("id"),
                    },
                },
            }

            if score is not None:
                container["items"][0]["columns"].append({
                    "type": "Column",
                    "width": "auto",
                    "items": [
                        {
                            "type": "TextBlock",
                            "text": f"{score:.0%}",
                            "color": "Good" if score > 0.7 else "Warning" if score > 0.4 else "Default",
                        },
                    ],
                })

            body.append(container)

    return {
        "$schema": "http://adaptivecards.io/schemas/adaptive-card.json",
        "type": "AdaptiveCard",
        "version": "1.4",
        "body": body,
    }


def create_documents_list_card(
    documents: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Create documents list card."""
    body = [
        {
            "type": "TextBlock",
            "text": "📚 Recent Documents",
            "weight": "Bolder",
            "size": "Medium",
        },
    ]

    if not documents:
        body.append({
            "type": "TextBlock",
            "text": "No documents found.",
            "wrap": True,
            "isSubtle": True,
        })
    else:
        body.append({
            "type": "TextBlock",
            "text": f"Showing {len(documents)} most recent document(s)",
            "size": "Small",
            "isSubtle": True,
        })

        for doc in documents:
            doc_name = doc.get("name") or doc.get("filename") or "Untitled"
            doc_type = doc.get("type") or doc.get("file_type") or "Unknown"
            created_at = doc.get("created_at", "")

            # Format date if available
            if created_at and "T" in created_at:
                created_at = created_at.split("T")[0]

            container = {
                "type": "Container",
                "separator": True,
                "spacing": "Small",
                "items": [
                    {
                        "type": "ColumnSet",
                        "columns": [
                            {
                                "type": "Column",
                                "width": "stretch",
                                "items": [
                                    {
                                        "type": "TextBlock",
                                        "text": f"📄 {doc_name}",
                                        "weight": "Bolder",
                                        "wrap": True,
                                    },
                                    {
                                        "type": "TextBlock",
                                        "text": f"{doc_type} • {created_at}" if created_at else doc_type,
                                        "size": "Small",
                                        "isSubtle": True,
                                    },
                                ],
                            },
                            {
                                "type": "Column",
                                "width": "auto",
                                "items": [
                                    {
                                        "type": "ActionSet",
                                        "actions": [
                                            {
                                                "type": "Action.Submit",
                                                "title": "View",
                                                "data": {
                                                    "action": "view_document",
                                                    "document_id": doc.get("id"),
                                                },
                                            },
                                        ],
                                    },
                                ],
                            },
                        ],
                    },
                ],
            }

            body.append(container)

    return {
        "$schema": "http://adaptivecards.io/schemas/adaptive-card.json",
        "type": "AdaptiveCard",
        "version": "1.4",
        "body": body,
    }


def create_summary_card(
    document_name: str,
    summary: str,
) -> Dict[str, Any]:
    """Create document summary card."""
    return {
        "$schema": "http://adaptivecards.io/schemas/adaptive-card.json",
        "type": "AdaptiveCard",
        "version": "1.4",
        "body": [
            {
                "type": "TextBlock",
                "text": f"📄 Summary: {document_name}",
                "weight": "Bolder",
                "size": "Medium",
                "wrap": True,
            },
            {
                "type": "TextBlock",
                "text": summary,
                "wrap": True,
                "spacing": "Medium",
            },
        ],
        "actions": [
            {
                "type": "Action.Submit",
                "title": "Ask a Question",
                "data": {
                    "action": "ask_about_document",
                    "document_name": document_name,
                },
            },
        ],
    }


def create_error_card(
    error_message: str,
    suggestion: Optional[str] = None,
) -> Dict[str, Any]:
    """Create error card."""
    body = [
        {
            "type": "TextBlock",
            "text": "⚠️ Something went wrong",
            "weight": "Bolder",
            "size": "Medium",
            "color": "Attention",
        },
        {
            "type": "TextBlock",
            "text": error_message,
            "wrap": True,
        },
    ]

    if suggestion:
        body.append({
            "type": "TextBlock",
            "text": f"💡 {suggestion}",
            "wrap": True,
            "spacing": "Medium",
            "isSubtle": True,
        })

    return {
        "$schema": "http://adaptivecards.io/schemas/adaptive-card.json",
        "type": "AdaptiveCard",
        "version": "1.4",
        "body": body,
        "actions": [
            {
                "type": "Action.Submit",
                "title": "Show Help",
                "data": {"action": "show_help"},
            },
        ],
    }


def create_processing_card(
    message: str = "Processing your request...",
) -> Dict[str, Any]:
    """Create processing/loading card."""
    return {
        "$schema": "http://adaptivecards.io/schemas/adaptive-card.json",
        "type": "AdaptiveCard",
        "version": "1.4",
        "body": [
            {
                "type": "TextBlock",
                "text": f"⏳ {message}",
                "wrap": True,
                "isSubtle": True,
            },
        ],
    }
