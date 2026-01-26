"""
OCR Manager Service
===================

Centralized service for managing OCR engines (PaddleOCR, Tesseract) with:
- Model download and installation
- Provider switching
- Model information and status
- Settings-based initialization
"""

import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import structlog

from backend.services.settings import SettingsService

logger = structlog.get_logger(__name__)


class OCRManager:
    """Manages OCR engines and model lifecycle."""

    def __init__(self, settings_service: SettingsService):
        """Initialize OCR Manager.

        Args:
            settings_service: Settings service for configuration
        """
        self.settings_service = settings_service
        self._paddle_engine: Optional[Any] = None
        self._tesseract_engine: Optional[Any] = None
        self._easyocr_engine: Optional[Any] = None
        self._mistral_engine: Optional[Any] = None  # Phase 68: Mistral OCR 3
        self._initialized_provider: Optional[str] = None

    async def get_model_info(self) -> Dict[str, Any]:
        """Get information about downloaded PaddleOCR models.

        Returns:
            dict: Model information including:
                - downloaded: List of downloaded models
                - total_size: Total size in MB
                - model_dir: Path to model directory
                - status: Installation status
        """
        try:
            settings = await self.settings_service.get_all_settings()
            model_dir = Path(settings.get("ocr.paddle.model_dir", "./data/paddle_models"))

            if not model_dir.exists():
                return {
                    "downloaded": [],
                    "total_size": 0,
                    "model_dir": str(model_dir),
                    "status": "not_installed",
                }

            # Calculate total size
            total_size = 0
            models = []

            # Check for downloaded models
            for item in model_dir.rglob("*"):
                if item.is_file():
                    size = item.stat().st_size
                    total_size += size

                    # Identify model files
                    if item.suffix in [".pdparams", ".pdiparams", ".pdmodel", ".pdimodel"]:
                        models.append({
                            "name": item.stem,
                            "type": item.suffix,
                            "size": f"{size / (1024 * 1024):.1f} MB",
                            "path": str(item.relative_to(model_dir)),
                        })

            return {
                "downloaded": models,
                "total_size": f"{total_size / (1024 * 1024):.1f} MB",
                "model_dir": str(model_dir),
                "status": "installed" if models else "empty",
            }

        except Exception as e:
            logger.error("Failed to get model info", error=str(e))
            return {
                "downloaded": [],
                "total_size": "0 MB",
                "model_dir": "unknown",
                "status": "error",
                "error": str(e),
            }

    async def download_models(
        self,
        languages: Optional[List[str]] = None,
        variant: Optional[str] = None,
        progress_callback: Optional[callable] = None
    ) -> Dict[str, Any]:
        """Download PaddleOCR models for specified languages with progress tracking.

        Args:
            languages: List of language codes (e.g., ["en", "de"]).
                      If None, uses settings.
            variant: Model variant ("server" or "mobile").
                    If None, uses settings.
            progress_callback: Optional callback function(current, total, language, status)
                             for progress updates.

        Returns:
            dict: Download result with status and details
        """
        try:
            # Get settings
            settings = await self.settings_service.get_all_settings()

            if languages is None:
                languages = settings.get("ocr.paddle.languages", ["en"])
            if variant is None:
                variant = settings.get("ocr.paddle.variant", "server")

            model_dir = settings.get("ocr.paddle.model_dir", "./data/paddle_models")

            # Set environment variables for PaddleOCR
            os.environ["PADDLEX_HOME"] = model_dir
            os.environ["PADDLE_HUB_HOME"] = os.path.join(model_dir, "official_models")
            os.environ["PADDLE_PDX_MODEL_SOURCE"] = "HF"  # Use HuggingFace mirror

            # Create directories
            Path(model_dir).mkdir(parents=True, exist_ok=True)

            logger.info(
                "Starting PaddleOCR model download",
                languages=languages,
                variant=variant,
                model_dir=model_dir,
            )

            # Import PaddleOCR (triggers download if models missing)
            try:
                from paddleocr import PaddleOCR
            except ImportError:
                return {
                    "status": "error",
                    "message": "PaddleOCR not installed. Install with: pip install paddleocr paddlepaddle",
                }

            downloaded = []
            failed = []
            total = len(languages)

            for idx, lang in enumerate(languages, 1):
                try:
                    logger.info(f"Downloading {lang.upper()} models ({idx}/{total})")

                    if progress_callback:
                        await progress_callback(idx - 1, total, lang, "downloading")

                    # Initialize PaddleOCR to trigger model download
                    ocr = PaddleOCR(
                        use_textline_orientation=True,
                        use_doc_orientation_classify=False,
                        use_doc_unwarping=False,
                        lang=lang,
                    )

                    downloaded.append(lang)
                    logger.info(f"✓ {lang.upper()} models downloaded ({idx}/{total})")

                    if progress_callback:
                        await progress_callback(idx, total, lang, "completed")

                except Exception as e:
                    failed.append({"language": lang, "error": str(e)})
                    logger.error(f"Failed to download {lang.upper()} models", error=str(e))

                    if progress_callback:
                        await progress_callback(idx, total, lang, "failed")

            # Get updated model info
            model_info = await self.get_model_info()

            return {
                "status": "success" if not failed else "partial",
                "downloaded": downloaded,
                "failed": failed,
                "model_info": model_info,
                "progress": {
                    "current": total,
                    "total": total,
                    "percentage": 100,
                }
            }

        except Exception as e:
            logger.error("Model download failed", error=str(e))
            return {
                "status": "error",
                "message": str(e),
            }

    async def download_models_batch(
        self,
        language_batches: List[List[str]],
        variant: Optional[str] = None,
        progress_callback: Optional[callable] = None
    ) -> Dict[str, Any]:
        """Download PaddleOCR models in batches with progress tracking.

        Useful for downloading many languages without blocking for too long.

        Args:
            language_batches: List of language code batches (e.g., [["en", "de"], ["fr", "es"]]).
            variant: Model variant ("server" or "mobile").
            progress_callback: Optional callback for progress updates.

        Returns:
            dict: Combined download results from all batches
        """
        try:
            all_downloaded = []
            all_failed = []
            total_batches = len(language_batches)

            for batch_idx, languages in enumerate(language_batches, 1):
                logger.info(f"Processing batch {batch_idx}/{total_batches}: {languages}")

                result = await self.download_models(
                    languages=languages,
                    variant=variant,
                    progress_callback=progress_callback
                )

                all_downloaded.extend(result.get("downloaded", []))
                all_failed.extend(result.get("failed", []))

            # Get updated model info
            model_info = await self.get_model_info()

            return {
                "status": "success" if not all_failed else "partial",
                "downloaded": all_downloaded,
                "failed": all_failed,
                "model_info": model_info,
                "batches_processed": total_batches,
            }

        except Exception as e:
            logger.error("Batch model download failed", error=str(e))
            return {
                "status": "error",
                "message": str(e),
            }

    async def initialize_ocr(self) -> None:
        """Initialize OCR engine based on current settings.

        Raises:
            ImportError: If required OCR library not installed
            Exception: If initialization fails
        """
        try:
            settings = await self.settings_service.get_all_settings()
            provider = settings.get("ocr.provider", "paddleocr")

            logger.info("Initializing OCR engine", provider=provider)

            if provider in ["paddleocr", "auto"]:
                await self._initialize_paddleocr()

            if provider in ["easyocr", "auto"]:
                await self._initialize_easyocr()

            if provider in ["tesseract", "auto"]:
                self._initialize_tesseract()

            # Phase 68: Mistral OCR 3 - highest accuracy for complex documents
            if provider in ["mistral", "mistral-ocr", "auto"]:
                await self._initialize_mistral_ocr()

            self._initialized_provider = provider
            logger.info("OCR engine initialized", provider=provider)

        except Exception as e:
            logger.error("OCR initialization failed", error=str(e))
            raise

    async def _initialize_paddleocr(self) -> None:
        """Initialize PaddleOCR engine."""
        try:
            from paddleocr import PaddleOCR
        except ImportError:
            raise ImportError(
                "PaddleOCR not installed. Install with: pip install paddleocr paddlepaddle"
            )

        settings = await self.settings_service.get_all()
        languages = settings.get("ocr.paddle.languages", ["en"])
        model_dir = settings.get("ocr.paddle.model_dir", "./data/paddle_models")

        # Set environment variables
        os.environ["PADDLEX_HOME"] = model_dir
        os.environ["PADDLE_HUB_HOME"] = os.path.join(model_dir, "official_models")
        os.environ["PADDLE_PDX_MODEL_SOURCE"] = "HF"

        # Initialize with first language (can switch later)
        lang = languages[0] if languages else "en"

        self._paddle_engine = PaddleOCR(
            use_textline_orientation=True,
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            lang=lang,
        )

        logger.info("PaddleOCR initialized", language=lang)

    def _initialize_tesseract(self) -> None:
        """Initialize Tesseract engine."""
        try:
            import pytesseract
        except ImportError:
            raise ImportError(
                "pytesseract not installed. Install with: pip install pytesseract"
            )

        # Test if tesseract is available
        try:
            pytesseract.get_tesseract_version()
            self._tesseract_engine = pytesseract
            logger.info("Tesseract initialized")
        except Exception as e:
            raise RuntimeError(f"Tesseract not found or not working: {e}")

    async def _initialize_easyocr(self) -> None:
        """Initialize EasyOCR engine."""
        try:
            import easyocr
        except ImportError:
            raise ImportError(
                "EasyOCR not installed. Install with: pip install easyocr"
            )

        settings = await self.settings_service.get_all_settings()
        languages = settings.get("ocr.easyocr.languages", ["en"])

        # EasyOCR uses GPU by default, allow CPU fallback
        gpu = settings.get("ocr.easyocr.use_gpu", True)

        # Map common language codes to EasyOCR format
        easyocr_langs = []
        lang_map = {
            "en": "en",
            "de": "de",
            "fr": "fr",
            "es": "es",
            "zh": "ch_sim",  # Chinese Simplified
            "ja": "ja",
            "ko": "ko",
            "ar": "ar",
        }

        for lang in languages:
            easyocr_langs.append(lang_map.get(lang, lang))

        try:
            self._easyocr_engine = easyocr.Reader(easyocr_langs, gpu=gpu)
            logger.info("EasyOCR initialized", languages=easyocr_langs, gpu=gpu)
        except Exception as e:
            logger.error("EasyOCR initialization failed", error=str(e))
            raise RuntimeError(f"EasyOCR initialization failed: {e}")

    async def _initialize_mistral_ocr(self) -> None:
        """Initialize Mistral OCR 3 engine.

        Phase 68: Mistral OCR 3 provides 74% win rate over previous versions
        with superior handling of forms, tables, and handwriting.
        """
        try:
            from backend.services.mistral_ocr import MistralOCRService, MistralOCRModel
        except ImportError:
            raise ImportError(
                "Mistral OCR service not available. Check mistral_ocr.py module."
            )

        settings = await self.settings_service.get_all_settings()

        # Get model from settings (default to OCR 3)
        model_name = settings.get("ocr.mistral.model", "mistral-ocr-3")
        model = MistralOCRModel.OCR_3 if "3" in model_name else MistralOCRModel.OCR_2

        # Initialize service
        api_key = settings.get("ocr.mistral.api_key") or os.getenv("MISTRAL_API_KEY")

        if not api_key:
            logger.warning(
                "Mistral API key not configured. "
                "Set MISTRAL_API_KEY env var or ocr.mistral.api_key setting."
            )
            return

        self._mistral_engine = MistralOCRService(
            api_key=api_key,
            model=model,
        )

        logger.info("Mistral OCR initialized", model=model.value)

    def get_ocr_engine(self) -> Any:
        """Get the initialized OCR engine.

        Returns:
            OCR engine instance (PaddleOCR, EasyOCR, pytesseract, or MistralOCR)

        Raises:
            RuntimeError: If OCR not initialized
        """
        if self._initialized_provider is None:
            raise RuntimeError(
                "OCR not initialized. Call initialize_ocr() first."
            )

        if self._initialized_provider in ["paddleocr", "auto"]:
            if self._paddle_engine is None:
                raise RuntimeError("PaddleOCR engine not available")
            return self._paddle_engine

        if self._initialized_provider == "easyocr":
            if self._easyocr_engine is None:
                raise RuntimeError("EasyOCR engine not available")
            return self._easyocr_engine

        if self._initialized_provider == "tesseract":
            if self._tesseract_engine is None:
                raise RuntimeError("Tesseract engine not available")
            return self._tesseract_engine

        # Phase 68: Mistral OCR 3
        if self._initialized_provider in ["mistral", "mistral-ocr"]:
            if self._mistral_engine is None:
                raise RuntimeError("Mistral OCR engine not available")
            return self._mistral_engine

        raise RuntimeError(f"Unknown OCR provider: {self._initialized_provider}")

    def get_paddle_engine(self) -> Optional[Any]:
        """Get PaddleOCR engine if available."""
        return self._paddle_engine

    def get_easyocr_engine(self) -> Optional[Any]:
        """Get EasyOCR engine if available."""
        return self._easyocr_engine

    def get_tesseract_engine(self) -> Optional[Any]:
        """Get Tesseract engine if available."""
        return self._tesseract_engine

    def get_mistral_engine(self) -> Optional[Any]:
        """Get Mistral OCR engine if available.

        Phase 68: Returns MistralOCRService instance.
        """
        return self._mistral_engine

    async def check_model_updates(self) -> Dict[str, Any]:
        """Check if newer PaddleOCR models are available.

        Returns:
            dict: Update information including available versions
        """
        try:
            import httpx
            from packaging import version as pkg_version
            from backend.services.http_client import get_http_client

            # Check PaddleOCR version
            try:
                import paddleocr
                current_version = paddleocr.__version__
            except (ImportError, AttributeError):
                return {
                    "status": "error",
                    "message": "PaddleOCR not installed",
                }

            # Check PyPI for latest version using async client
            try:
                client = await get_http_client()
                response = await client.get(
                    "https://pypi.org/pypi/paddleocr/json",
                    timeout=5.0
                )
                if response.status_code == 200:
                    data = response.json()
                    latest_version = data["info"]["version"]

                    update_available = pkg_version.parse(latest_version) > pkg_version.parse(current_version)

                    return {
                        "status": "success",
                        "current_version": current_version,
                        "latest_version": latest_version,
                        "update_available": update_available,
                        "release_date": data["urls"][0].get("upload_time", "unknown") if data.get("urls") else "unknown",
                    }
                else:
                    return {
                        "status": "error",
                        "message": f"Failed to fetch version info: HTTP {response.status_code}",
                    }
            except httpx.HTTPError as e:
                return {
                    "status": "error",
                    "message": f"Network error: {str(e)}",
                }

        except Exception as e:
            logger.error("Failed to check model updates", error=str(e))
            return {
                "status": "error",
                "message": str(e),
            }

    async def get_installed_models_info(self) -> Dict[str, Any]:
        """Get detailed information about installed models including versions.

        Returns:
            dict: Detailed model information
        """
        try:
            settings = await self.settings_service.get_all_settings()
            model_dir = Path(settings.get("ocr.paddle.model_dir", "./data/paddle_models"))

            if not model_dir.exists():
                return {
                    "status": "not_installed",
                    "models": [],
                }

            models = []

            # Check each model directory
            official_models_dir = model_dir / "official_models"
            if official_models_dir.exists():
                for model_path in official_models_dir.iterdir():
                    if model_path.is_dir():
                        # Get model size
                        total_size = sum(f.stat().st_size for f in model_path.rglob('*') if f.is_file())

                        # Get modification time as version indicator
                        mod_time = max(f.stat().st_mtime for f in model_path.rglob('*') if f.is_file()) if list(model_path.rglob('*')) else 0

                        models.append({
                            "name": model_path.name,
                            "path": str(model_path.relative_to(model_dir)),
                            "size_mb": round(total_size / (1024 * 1024), 2),
                            "last_modified": datetime.fromtimestamp(mod_time).isoformat() if mod_time else None,
                        })

            return {
                "status": "success",
                "models": models,
                "total_models": len(models),
                "model_dir": str(model_dir),
            }

        except Exception as e:
            logger.error("Failed to get installed models info", error=str(e))
            return {
                "status": "error",
                "message": str(e),
            }

    async def cleanup(self) -> None:
        """Cleanup OCR resources."""
        self._paddle_engine = None
        self._easyocr_engine = None
        self._tesseract_engine = None
        self._mistral_engine = None  # Phase 68
        self._initialized_provider = None
        logger.info("OCR resources cleaned up")
