"""
Format Generator Factory

Factory for creating format-specific document generators.
"""

from typing import Dict, Type, Optional

from ..models import OutputFormat
from .base import BaseFormatGenerator


class FormatGeneratorFactory:
    """Factory for creating format-specific generators.

    Usage:
        factory = FormatGeneratorFactory()
        generator = factory.get(OutputFormat.PPTX)
        output_path = await generator.generate(job, filename)
    """

    _generators: Dict[OutputFormat, Type[BaseFormatGenerator]] = {}
    _instances: Dict[OutputFormat, BaseFormatGenerator] = {}

    @classmethod
    def register(cls, format_type: OutputFormat, generator_class: Type[BaseFormatGenerator]) -> None:
        """Register a generator class for a format type."""
        cls._generators[format_type] = generator_class

    @classmethod
    def get(cls, format_type: OutputFormat) -> Optional[BaseFormatGenerator]:
        """Get a generator instance for the specified format.

        Returns a cached instance if available, otherwise creates a new one.
        """
        if format_type not in cls._instances:
            generator_class = cls._generators.get(format_type)
            if generator_class:
                cls._instances[format_type] = generator_class()
        return cls._instances.get(format_type)

    @classmethod
    def get_supported_formats(cls) -> list:
        """Return list of supported output formats."""
        return list(cls._generators.keys())

    @classmethod
    def is_supported(cls, format_type: OutputFormat) -> bool:
        """Check if a format is supported."""
        return format_type in cls._generators


# Auto-register generators when they're imported
def register_generator(format_type: OutputFormat):
    """Decorator to register a generator class."""
    def decorator(cls: Type[BaseFormatGenerator]) -> Type[BaseFormatGenerator]:
        FormatGeneratorFactory.register(format_type, cls)
        return cls
    return decorator
