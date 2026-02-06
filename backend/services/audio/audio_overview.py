"""
AIDocumentIndexer - Audio Overview Service
==========================================

Orchestrates the generation of audio overviews from documents.

This is the main service that coordinates:
1. Document content retrieval
2. Script generation
3. TTS synthesis
4. Audio file management
"""

import asyncio
import uuid
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional, AsyncGenerator

import structlog
from sqlalchemy import select
from sqlalchemy.orm import selectinload

from backend.services.base import BaseService, CRUDService, ServiceException, NotFoundException
from backend.services.audio.script_generator import ScriptGenerator, DialogueScript, DialogueTurn
from backend.services.audio.tts_service import TTSService, TTSProvider, VoiceConfig
from backend.db.models import AudioOverview, AudioOverviewFormat, AudioOverviewStatus, Document
from backend.core.config import settings

logger = structlog.get_logger(__name__)


class AudioOverviewService(CRUDService[AudioOverview]):
    """
    Service for creating and managing audio overviews.

    Coordinates the pipeline:
    Document(s) -> Script Generation -> TTS -> Audio File
    """

    model_class = AudioOverview
    model_name = "AudioOverview"

    def __init__(
        self,
        session=None,
        organization_id=None,
        user_id=None,
        storage_path: Optional[str] = None,
    ):
        super().__init__(session, organization_id, user_id)
        # Get backend directory for storage (backend/services/audio -> backend)
        backend_root = Path(__file__).resolve().parent.parent.parent
        default_storage = backend_root / "storage" / "audio"
        self.storage_path = Path(storage_path or getattr(settings, "AUDIO_STORAGE_PATH", None) or default_storage)
        self.storage_path.mkdir(parents=True, exist_ok=True)

    async def create_overview(
        self,
        document_ids: List[uuid.UUID],
        format: AudioOverviewFormat = AudioOverviewFormat.DEEP_DIVE,
        title: Optional[str] = None,
        custom_instructions: Optional[str] = None,
        tts_provider: TTSProvider = TTSProvider.OPENAI,
        host_config: Optional[Dict[str, Any]] = None,
    ) -> AudioOverview:
        """
        Create a new audio overview from documents.

        Args:
            document_ids: List of document UUIDs to include
            format: Audio format (deep_dive, brief, etc.)
            title: Optional custom title
            custom_instructions: Additional instructions for script generation
            tts_provider: TTS provider to use
            host_config: Voice configuration for hosts (e.g., {"host1_voice": "en-US-AndrewMultilingualNeural"})

        Returns:
            AudioOverview model instance
        """
        self.log_info(
            "Creating audio overview",
            document_count=len(document_ids),
            format=format.value,
        )

        session = await self.get_session()

        # Validate documents exist and belong to organization
        documents = await self._get_documents(document_ids)
        if len(documents) != len(document_ids):
            found_ids = {str(d.id) for d in documents}
            missing_ids = [str(did) for did in document_ids if str(did) not in found_ids]
            raise NotFoundException("Document", ", ".join(missing_ids))

        # Create the audio overview record
        overview = AudioOverview(
            id=uuid.uuid4(),
            organization_id=self._organization_id,
            document_ids=[str(did) for did in document_ids],
            format=format.value,
            title=title or "Audio Overview",
            status=AudioOverviewStatus.PENDING.value,
            tts_provider=tts_provider.value,
            description=custom_instructions,
            host_config=host_config,
            created_by_id=self._user_id if isinstance(self._user_id, uuid.UUID) else (uuid.UUID(self._user_id) if self._user_id else None),
        )

        session.add(overview)
        await session.commit()
        await session.refresh(overview)

        self.log_info("Audio overview created", overview_id=str(overview.id))

        return overview

    async def generate_overview(
        self,
        overview_id: uuid.UUID,
        background: bool = False,
    ) -> AudioOverview:
        """
        Generate the audio for an overview.

        Args:
            overview_id: The overview to generate
            background: If True, return immediately and generate in background

        Returns:
            Updated AudioOverview
        """
        overview = await self.get_by_id_or_raise(overview_id)

        # Only skip if status is ready AND audio file actually exists
        if overview.status == AudioOverviewStatus.READY.value:
            if overview.storage_path:
                import os
                if os.path.exists(overview.storage_path):
                    self.log_info("Overview already generated", overview_id=str(overview_id))
                    return overview
            # Status is ready but no file - reset and continue
            self.log_info("Overview marked ready but no audio file, regenerating", overview_id=str(overview_id))
            overview.status = AudioOverviewStatus.PENDING.value
            session = await self.get_session()
            await session.commit()

        if background:
            # Start background task
            logger.info("Creating background task for audio generation", overview_id=str(overview_id))
            task = asyncio.create_task(self._generate_in_background(overview_id))
            logger.info("Background task created", overview_id=str(overview_id), task_name=task.get_name())
            return overview

        return await self._generate_overview_impl(overview)

    async def _generate_in_background(self, overview_id: uuid.UUID):
        """Background task for generation."""
        logger.info("Background generation task started", overview_id=str(overview_id))
        try:
            # Get fresh session for background task
            from backend.db.database import async_session_context

            async with async_session_context() as session:
                # Create new service instance with fresh session
                service = AudioOverviewService(
                    session=session,
                    organization_id=self._organization_id,
                    user_id=self._user_id,
                )
                overview = await service.get_by_id_or_raise(overview_id)
                await service._generate_overview_impl(overview)

        except Exception as e:
            logger.error("Background generation failed", overview_id=str(overview_id), error=str(e), exc_info=True)
            # Update status to failed in a new session
            try:
                from backend.db.database import async_session_context
                async with async_session_context() as session:
                    from sqlalchemy import select
                    result = await session.execute(
                        select(AudioOverview).where(AudioOverview.id == overview_id)
                    )
                    overview = result.scalar_one_or_none()
                    if overview:
                        overview.status = AudioOverviewStatus.FAILED.value
                        overview.error_message = str(e)
                        await session.commit()
            except Exception as update_error:
                logger.error("Failed to update overview status", error=str(update_error))

    async def _generate_overview_impl(self, overview: AudioOverview) -> AudioOverview:
        """Internal implementation of overview generation."""
        session = await self.get_session()

        try:
            # Update status
            overview.status = AudioOverviewStatus.GENERATING_SCRIPT.value
            await session.commit()

            # Get document contents (handle both UUID objects and strings)
            doc_ids = [did if isinstance(did, uuid.UUID) else uuid.UUID(did) for did in overview.document_ids]
            self.log_info("Fetching documents", doc_count=len(doc_ids))
            documents = await self._get_documents(doc_ids)
            self.log_info("Documents fetched", found_count=len(documents))

            if not documents:
                raise ServiceException(
                    "No documents found for audio overview",
                    code="AUDIO_GENERATION_ERROR",
                    details={"document_ids": [str(d) for d in doc_ids]},
                )

            # Use RAG-enhanced content preparation with format-specific queries
            document_contents = await self._prepare_document_contents(
                documents,
                format_type=overview.format,
                use_rag=True,
                use_knowledge_graph=True,
            )
            total_content_length = sum(len(d.get("content", "")) for d in document_contents)
            entity_count = sum(len(d.get("entities", [])) for d in document_contents)
            self.log_info(
                "Document contents prepared with RAG/KG",
                doc_count=len(document_contents),
                total_chars=total_content_length,
                entities_found=entity_count,
            )

            if total_content_length == 0:
                raise ServiceException(
                    "Documents have no text content for audio overview",
                    code="AUDIO_GENERATION_ERROR",
                )

            # Generate script
            script_generator = ScriptGenerator(
                session=session,
                organization_id=self._organization_id,
                user_id=self._user_id,
            )

            # Get duration preference and custom names from host_config
            host_config = overview.host_config or {}
            duration_preference = host_config.get("duration_preference", "standard")
            host1_name = host_config.get("host1_name", "Alex")
            host2_name = host_config.get("host2_name", "Jordan")

            script = await script_generator.generate_script(
                document_contents=document_contents,
                format=overview.format,
                title=overview.title,
                custom_instructions=overview.description,  # Custom instructions stored in description
                duration_preference=duration_preference,
                host1_name=host1_name,
                host2_name=host2_name,
            )

            # Store script
            overview.script = script.model_dump()
            overview.status = AudioOverviewStatus.GENERATING_AUDIO.value
            await session.commit()

            # Generate audio
            tts_provider = TTSProvider(overview.tts_provider) if overview.tts_provider else TTSProvider.OPENAI
            tts_service = TTSService(
                session=session,
                organization_id=self._organization_id,
                user_id=self._user_id,
                default_provider=tts_provider,
            )

            # Build voice configs from script speakers, using user-selected voices from host_config
            # Use 'or' to handle both missing keys AND explicit None values
            host_config = overview.host_config or {}
            host1_voice = host_config.get("host1_voice") or "alloy"
            host2_voice = host_config.get("host2_voice") or "echo"

            # PHASE 12: Debug logging for voice selection troubleshooting
            logger.info(
                "Voice configuration for audio generation",
                overview_id=str(overview.id),
                format=str(overview.format),
                host_config=host_config,
                host1_voice=host1_voice,
                host2_voice=host2_voice,
                script_speakers=[s["id"] for s in script.speakers],
            )

            speaker_voices = {}
            for speaker in script.speakers:
                # Map speaker ID to user-selected voice
                # Handle all format types: standard (host1/host2), interview (interviewer/expert),
                # and lecture (lecturer) - all primary speakers should use host1_voice
                if speaker["id"] in ["host1", "interviewer", "lecturer"]:
                    voice_id = host1_voice
                elif speaker["id"] in ["host2", "expert"]:
                    voice_id = host2_voice
                else:
                    # Fallback for any other speaker IDs - use speaker's configured voice or default
                    voice_id = speaker.get("voice") or "alloy"

                logger.debug(f"Mapping speaker {speaker['id']} to voice {voice_id}")

                speaker_voices[speaker["id"]] = VoiceConfig(
                    provider=tts_provider,
                    voice_id=voice_id,
                    name=speaker["name"],
                    speed=1.0,
                    style=speaker.get("style"),
                )

            # Generate audio file
            output_filename = f"{overview.id}.mp3"
            output_path = self.storage_path / output_filename

            await tts_service.synthesize_dialogue(
                turns=[turn.model_dump() for turn in script.turns],
                speaker_voices=speaker_voices,
                output_path=str(output_path),
                add_pauses=True,
                pause_between_speakers_ms=600,  # Natural pause between speakers
            )

            # Update overview
            overview.audio_url = f"/api/v1/audio/files/{output_filename}"
            overview.storage_path = str(output_path)  # Store the actual file path
            overview.duration_seconds = script.estimated_duration_seconds
            overview.status = AudioOverviewStatus.READY.value
            overview.error_message = None  # Clear any previous error
            overview.completed_at = datetime.utcnow()
            logger.info("Audio overview completed", overview_id=str(overview.id), storage_path=str(output_path), audio_url=overview.audio_url)

            await session.commit()
            await session.refresh(overview)

            self.log_info(
                "Audio overview generated",
                overview_id=str(overview.id),
                duration_seconds=overview.duration_seconds,
            )

            return overview

        except Exception as e:
            self.log_error("Generation failed", error=e, overview_id=str(overview.id))

            overview.status = AudioOverviewStatus.FAILED.value
            overview.error_message = str(e)
            await session.commit()

            raise ServiceException(
                f"Audio generation failed: {str(e)}",
                code="AUDIO_GENERATION_ERROR",
            )

    async def generate_overview_streaming(
        self,
        overview_id: uuid.UUID,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Generate overview with streaming updates.

        Yields status updates as the generation progresses.
        """
        overview = await self.get_by_id_or_raise(overview_id)
        session = await self.get_session()

        try:
            yield {"type": "status", "status": "starting", "message": "Starting generation..."}

            # Update status
            overview.status = AudioOverviewStatus.GENERATING_SCRIPT.value
            await session.commit()

            yield {"type": "status", "status": "script", "message": "Generating script..."}

            # Get documents (handle both UUID objects and strings)
            doc_ids = [did if isinstance(did, uuid.UUID) else uuid.UUID(did) for did in overview.document_ids]
            documents = await self._get_documents(doc_ids)

            # Use RAG-enhanced content preparation with format-specific queries
            document_contents = await self._prepare_document_contents(
                documents,
                format_type=overview.format,
                use_rag=True,
                use_knowledge_graph=True,
            )

            # Generate script with streaming
            script_generator = ScriptGenerator(
                session=session,
                organization_id=self._organization_id,
                user_id=self._user_id,
            )

            # Get duration preference and custom names from host_config
            host_config = overview.host_config or {}
            duration_preference = host_config.get("duration_preference", "standard")
            host1_name = host_config.get("host1_name", "Alex")
            host2_name = host_config.get("host2_name", "Jordan")

            turns = []
            turn_count = 0

            async for turn in script_generator.generate_script_streaming(
                document_contents=document_contents,
                format=overview.format,
                title=overview.title,
                duration_preference=duration_preference,
                host1_name=host1_name,
                host2_name=host2_name,
            ):
                turns.append(turn)
                turn_count += 1
                yield {
                    "type": "turn",
                    "turn_number": turn_count,
                    "speaker": turn.speaker,
                    "text": turn.text[:100] + "..." if len(turn.text) > 100 else turn.text,
                }

            # Get voice configuration from host_config or use defaults
            # Use 'or' to handle both missing keys AND explicit None values
            host1_voice = host_config.get("host1_voice") or "alloy"
            host2_voice = host_config.get("host2_voice") or "echo"

            # PHASE 12 FIX: Build speakers list based on format type
            # Interview format uses "interviewer" and "expert" as speaker IDs
            from backend.db.models import AudioOverviewFormat
            if overview.format == AudioOverviewFormat.INTERVIEW:
                speakers = [
                    {"id": "interviewer", "name": host1_name, "voice": host1_voice},
                    {"id": "expert", "name": host2_name, "voice": host2_voice},
                ]
            elif overview.format == AudioOverviewFormat.LECTURE:
                speakers = [
                    {"id": "lecturer", "name": host1_name, "voice": host1_voice},
                ]
            else:
                # Standard two-host formats (deep_dive, brief, critique, debate)
                speakers = [
                    {"id": "host1", "name": host1_name, "voice": host1_voice},
                    {"id": "host2", "name": host2_name, "voice": host2_voice},
                ]

            logger.info(
                "Building script with speakers",
                format=str(overview.format),
                speakers=[s["id"] for s in speakers],
                host1_name=host1_name,
                host2_name=host2_name,
            )

            # Build final script with custom names
            script = DialogueScript(
                title=overview.title or "Audio Overview",
                format=overview.format.value,
                estimated_duration_seconds=int(sum(len(t.text.split()) for t in turns) / 150 * 60),
                speakers=speakers,
                turns=turns,
            )

            overview.script = script.model_dump()
            overview.status = AudioOverviewStatus.GENERATING_AUDIO.value
            await session.commit()

            yield {"type": "status", "status": "audio", "message": "Generating audio..."}

            # Generate audio
            tts_provider = TTSProvider(overview.tts_provider) if overview.tts_provider else TTSProvider.OPENAI
            tts_service = TTSService(
                session=session,
                organization_id=self._organization_id,
                user_id=self._user_id,
                default_provider=tts_provider,
            )

            # Build speaker_voices dynamically from script.speakers
            # Handle all format types: standard (host1/host2), interview (interviewer/expert),
            # and lecture (lecturer) - all primary speakers should use host1_voice
            speaker_voices = {}
            for speaker in script.speakers:
                speaker_id = speaker["id"]
                # Map to user-selected voice based on speaker position
                if speaker_id in ["host1", "interviewer", "lecturer"]:
                    voice_id = host1_voice
                elif speaker_id in ["host2", "expert"]:
                    voice_id = host2_voice
                else:
                    # Fallback for any other speaker IDs - use speaker's configured voice or default
                    voice_id = speaker.get("voice") or "alloy"

                speaker_voices[speaker_id] = VoiceConfig(
                    provider=tts_provider,
                    voice_id=voice_id,
                    name=speaker["name"],
                )

            output_filename = f"{overview.id}.mp3"
            output_path = self.storage_path / output_filename

            # Generate in batches and report progress
            total_turns = len(turns)
            for i, turn in enumerate(turns):
                yield {
                    "type": "audio_progress",
                    "current": i + 1,
                    "total": total_turns,
                    "percentage": int((i + 1) / total_turns * 100),
                }

            await tts_service.synthesize_dialogue(
                turns=[turn.model_dump() for turn in turns],
                speaker_voices=speaker_voices,
                output_path=str(output_path),
                add_pauses=True,
                pause_between_speakers_ms=600,  # Natural pause between speakers
            )

            # Update overview
            overview.audio_url = f"/api/v1/audio/files/{output_filename}"
            overview.storage_path = str(output_path)  # Store the actual file path
            overview.duration_seconds = script.estimated_duration_seconds
            overview.status = AudioOverviewStatus.READY.value
            overview.completed_at = datetime.utcnow()

            await session.commit()

            yield {
                "type": "complete",
                "overview_id": str(overview.id),
                "audio_url": overview.audio_url,
                "storage_path": str(output_path),
                "duration_seconds": overview.duration_seconds,
            }

        except Exception as e:
            self.log_error("Streaming generation failed", error=e)

            overview.status = AudioOverviewStatus.FAILED.value
            overview.error_message = str(e)
            await session.commit()

            yield {"type": "error", "message": str(e)}

    async def _get_documents(self, document_ids: List[uuid.UUID]) -> List[Document]:
        """Get documents by IDs."""
        session = await self.get_session()

        # Documents don't have organization_id - they're organized through folders
        # Security is handled by folder permissions at the API layer
        query = select(Document).where(Document.id.in_(document_ids))

        result = await session.execute(query)
        return list(result.scalars().all())

    async def _prepare_document_contents(
        self,
        documents: List[Document],
        format_type: str = "deep_dive",
        use_rag: bool = True,
        use_knowledge_graph: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Prepare document contents for script generation using RAG and optionally Knowledge Graph.

        Args:
            documents: List of documents to process
            format_type: Audio format type (deep_dive, brief, debate, etc.)
            use_rag: Use semantic search for content selection
            use_knowledge_graph: Use knowledge graph for entity context
        """
        contents = []

        # Format-specific queries for semantic search
        format_queries = {
            "deep_dive": ["main findings and key insights", "important details and examples", "implications and conclusions"],
            "brief": ["executive summary", "key takeaways and highlights", "main conclusions"],
            "debate": ["arguments supporting this view", "arguments against this view", "contrasting perspectives"],
            "critique": ["strengths and benefits", "weaknesses and limitations", "recommendations"],
            "lecture": ["fundamental concepts", "key principles", "practical applications"],
            "interview": ["expert insights", "key achievements", "future directions"],
        }

        queries = format_queries.get(format_type, ["main content", "key information"])

        for doc in documents:
            if use_rag:
                # Use RAG for semantic content selection
                content, entities = await self._get_document_content_with_rag(doc, queries, use_knowledge_graph)
            else:
                # Fallback to sequential chunk reading
                content = await self._get_document_text(doc)
                entities = []

            doc_data = {
                "name": doc.filename or doc.title or "Untitled",
                "content": content,
                "summary": doc.enhanced_metadata.get("summary_short", "") if doc.enhanced_metadata else "",
                "metadata": {
                    "file_type": doc.file_type,
                    "word_count": doc.word_count,
                    "page_count": doc.page_count,
                },
            }

            # Add entity context if available
            if entities:
                doc_data["entities"] = entities
                # Add entity summary for script generation
                entity_names = [e.get("name", "") for e in entities[:10]]
                doc_data["key_entities"] = ", ".join(entity_names)

            contents.append(doc_data)

        return contents

    async def _get_document_content_with_rag(
        self,
        document: Document,
        queries: List[str],
        use_knowledge_graph: bool = True,
    ) -> tuple[str, List[Dict[str, Any]]]:
        """
        Get document content using RAG semantic search and optional KG.

        Returns:
            Tuple of (content_string, entities_list)
        """
        try:
            from backend.services.rag import RAGService

            rag = RAGService()
            all_chunks = []
            chunk_ids_seen = set()

            # Search for each format-specific query
            for query in queries:
                try:
                    # Use RAG search with document filter
                    results = await rag.search(
                        query=query,
                        top_k=5,
                        document_ids=[str(document.id)],
                        organization_id=str(self._organization_id) if self._organization_id else None,
                    )

                    # Deduplicate and collect chunks
                    for result in results:
                        chunk_id = result.get("chunk_id") or result.get("id")
                        if chunk_id and chunk_id not in chunk_ids_seen:
                            chunk_ids_seen.add(chunk_id)
                            all_chunks.append({
                                "content": result.get("content", ""),
                                "score": result.get("score", 0.0),
                                "query": query,
                            })
                except Exception as e:
                    logger.warning(f"RAG search failed for query '{query}': {e}")
                    continue

            # Sort by relevance score and concatenate
            all_chunks.sort(key=lambda x: x.get("score", 0), reverse=True)
            content = "\n\n".join([c["content"] for c in all_chunks[:15]])

            # Get entities from Knowledge Graph if enabled
            entities = []
            if use_knowledge_graph:
                try:
                    entities = await self._get_entities_for_document(document)
                except Exception as e:
                    logger.warning(f"Knowledge Graph lookup failed: {e}")

            # If RAG returned nothing, fallback to sequential reading
            if not content.strip():
                logger.info("RAG returned no content, falling back to sequential chunks")
                content = await self._get_document_text(document)

            return content, entities

        except ImportError:
            logger.warning("RAG service not available, falling back to sequential chunks")
            return await self._get_document_text(document), []
        except Exception as e:
            logger.error(f"Error in RAG content retrieval: {e}")
            return await self._get_document_text(document), []

    async def _get_entities_for_document(self, document: Document) -> List[Dict[str, Any]]:
        """Get Knowledge Graph entities associated with a document.

        Phase 60: Uses settings-controlled KG integration for audio overview.
        """
        # Phase 60: Check if KG is enabled for audio
        from backend.core.config import settings
        if not settings.KG_ENABLED or not settings.KG_ENABLED_IN_AUDIO:
            logger.debug("KG disabled for audio overview")
            return []

        try:
            from backend.services.knowledge_graph import get_kg_service

            # Use singleton KG service
            kg = await get_kg_service()

            # Get entities mentioned in this document's chunks
            entities = await kg.find_entities_for_document(str(document.id))

            # Phase 60: Score entities by importance for audio overview
            # Prioritize entities with more mentions and higher confidence
            scored_entities = []
            for e in entities:
                score = (
                    (e.mention_count or 1) * 0.5 +
                    (e.confidence or 0.5) * 0.5
                )
                scored_entities.append((e, score))

            # Sort by score and take top entities
            scored_entities.sort(key=lambda x: x[1], reverse=True)

            return [
                {
                    "name": e.canonical_name,
                    "type": e.entity_type.value if hasattr(e.entity_type, 'value') else str(e.entity_type),
                    "aliases": e.aliases[:3] if e.aliases else [],
                    "importance": score,  # Include score for script generation
                }
                for e, score in scored_entities[:15]  # Limit to top 15 entities
            ]
        except ImportError:
            logger.debug("Knowledge Graph service not available")
            return []
        except Exception as e:
            logger.warning(f"Error getting entities for document: {e}")
            return []

    async def _get_document_text(self, document: Document) -> str:
        """Extract text content from a document."""
        session = await self.get_session()

        # Try to get from chunks
        from backend.db.models import Chunk

        query = select(Chunk).where(
            Chunk.document_id == document.id
        ).order_by(Chunk.chunk_index)

        result = await session.execute(query)
        chunks = result.scalars().all()

        if chunks:
            return "\n\n".join(chunk.content for chunk in chunks if chunk.content)

        # Fallback to stored content if available
        if document.enhanced_metadata and document.enhanced_metadata.get("full_text"):
            return document.enhanced_metadata["full_text"]

        return ""

    async def get_audio_file_path(self, overview_id: uuid.UUID) -> Optional[Path]:
        """Get the file path for an overview's audio file."""
        overview = await self.get_by_id(overview_id)
        if not overview or not overview.audio_url:
            return None

        filename = overview.audio_url.split("/")[-1]
        path = self.storage_path / filename

        if path.exists():
            return path

        return None

    async def list_by_documents(
        self,
        document_ids: List[uuid.UUID],
        page: int = 1,
        page_size: int = 20,
    ) -> tuple[List[AudioOverview], int]:
        """List audio overviews that include specific documents."""
        session = await self.get_session()
        from sqlalchemy import func, or_

        # Build query for JSON array contains
        doc_id_strs = [str(did) for did in document_ids]

        query = select(AudioOverview).where(
            AudioOverview.organization_id == self._organization_id
        )

        # Filter by document IDs (check if any of the IDs are in the array)
        # This is a simplified approach - production would use proper JSON array operations
        conditions = []
        for doc_id in doc_id_strs:
            conditions.append(AudioOverview.document_ids.contains([doc_id]))

        if conditions:
            query = query.where(or_(*conditions))

        # Count
        count_query = select(func.count()).select_from(query.subquery())
        total_result = await session.execute(count_query)
        total = total_result.scalar() or 0

        # Paginate
        query = query.order_by(AudioOverview.created_at.desc())
        query = query.offset((page - 1) * page_size).limit(page_size)

        result = await session.execute(query)
        overviews = list(result.scalars().all())

        return overviews, total

    async def estimate_cost(
        self,
        document_ids: List[uuid.UUID],
        format: AudioOverviewFormat,
        tts_provider: TTSProvider = TTSProvider.OPENAI,
    ) -> Dict[str, Any]:
        """
        Estimate the cost of generating an audio overview.

        Returns cost breakdown by component.
        """
        # Get documents
        documents = await self._get_documents(document_ids)
        document_contents = await self._prepare_document_contents(documents)

        # Estimate script generation cost
        total_chars = sum(len(doc.get("content", "")) for doc in document_contents)

        # Estimate script length based on format
        script_generator = ScriptGenerator(
            session=self._session,
            organization_id=self._organization_id,
            user_id=self._user_id,
        )
        duration_estimate = await script_generator.estimate_duration(document_contents, format)

        # Estimate TTS cost
        estimated_script_chars = duration_estimate["target_seconds"] * 15  # ~15 chars/second speech
        tts_service = TTSService(
            session=self._session,
            organization_id=self._organization_id,
            user_id=self._user_id,
        )
        tts_cost = await tts_service.estimate_cost("x" * estimated_script_chars, tts_provider)

        # LLM cost for script generation (rough estimate)
        llm_input_tokens = total_chars // 4  # ~4 chars per token
        llm_output_tokens = estimated_script_chars // 4
        llm_cost = (llm_input_tokens * 0.00001) + (llm_output_tokens * 0.00003)  # GPT-4 pricing

        return {
            "document_count": len(documents),
            "total_document_chars": total_chars,
            "estimated_duration": duration_estimate,
            "estimated_costs": {
                "llm_script_generation_usd": round(llm_cost, 4),
                "tts_synthesis_usd": round(tts_cost["estimated_cost_usd"], 4),
                "total_usd": round(llm_cost + tts_cost["estimated_cost_usd"], 4),
            },
            "tts_provider": tts_provider.value,
        }
