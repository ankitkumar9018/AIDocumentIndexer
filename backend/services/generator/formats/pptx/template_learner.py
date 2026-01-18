"""
PPTX Template Layout Learner

Deep analysis of PPTX templates to learn layout patterns, branding zones,
and content constraints for intelligent per-slide content generation.

This module provides:
- TemplateLayoutLearner: XML-based analysis of slide layouts
- VisionTemplateAnalyzer: Optional vision-based template analysis
- SlideContentPlanner: Per-slide content constraint planning
"""

import zipfile
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List, TYPE_CHECKING

import structlog
from lxml import etree

if TYPE_CHECKING:
    from pptx import Presentation
    from pptx.slide import SlideLayout

from ...models import LearnedLayout

logger = structlog.get_logger(__name__)

# Constants
EMU_PER_INCH = 914400


@dataclass
class SlideContentPlan:
    """Content plan for a specific slide based on its learned layout."""
    layout: LearnedLayout
    title_constraints: Dict[str, Any] = field(default_factory=dict)
    content_constraints: Dict[str, Any] = field(default_factory=dict)
    image_placement: Optional[Dict[str, Any]] = None
    avoid_zones: List[Dict[str, Any]] = field(default_factory=list)
    visual_style: Dict[str, Any] = field(default_factory=dict)


class TemplateLayoutLearner:
    """Learns layout patterns from template slides for intelligent generation.

    Analyzes each slide layout in a template to understand:
    - Safe zones for content (avoiding branding elements like logos)
    - Content constraints (max characters based on actual placeholder sizes)
    - Layout types and their appropriate use cases
    - Branding element positions to avoid

    Usage:
        learner = TemplateLayoutLearner(presentation)
        learned_layouts = await learner.learn_from_template(template_path)
        # Each layout now has safe zones and constraints calculated
    """

    def __init__(self, presentation: "Presentation", config: Optional[Dict[str, Any]] = None):
        """Initialize the template learner.

        Args:
            presentation: The python-pptx Presentation object
            config: Optional configuration dict with keys:
                - enable_template_vision_analysis: bool
                - template_vision_model: str
        """
        self.prs = presentation
        self.config = config or {}
        self.slide_width = presentation.slide_width
        self.slide_height = presentation.slide_height
        self.learned_layouts: Dict[str, LearnedLayout] = {}

        # Vision analysis settings
        self.enable_vision = self.config.get('enable_template_vision_analysis', False)
        self.vision_model = self.config.get('template_vision_model', 'auto')

        # Detected logo position (minimum x coordinate of all logos found)
        # Used to calculate safe title width to avoid overlap
        self.detected_logo_x: Optional[int] = None

        # PHASE 10 FIX: Footer and content bottom positioning
        # detected_content_bottom_y: Y coordinate (in EMU) where content should stop
        # safe_footer_y: Y coordinate where our footer elements should be placed
        # footer_placeholder_y: Actual Y position from template's footer placeholder (if found)
        self.detected_content_bottom_y: Optional[int] = None
        self.footer_placeholder_y: Optional[int] = None  # Read from template master
        # Default footer position: 0.35" from bottom of slide (standard PowerPoint position)
        self.safe_footer_y: int = int(self.slide_height - (0.35 * EMU_PER_INCH))

        # Analyze slide master for footer placeholder position
        self._extract_footer_position_from_master()

    async def learn_from_template(self, template_path: Optional[str] = None) -> Dict[str, LearnedLayout]:
        """Analyze template with XML parsing + optional vision enhancement.

        Args:
            template_path: Path to the template file (needed for vision analysis)

        Returns:
            Dictionary mapping layout names to LearnedLayout objects
        """
        # Step 1: XML-based analysis (always runs)
        xml_layouts = self._analyze_all_layouts_xml()

        # Step 2: Analyze actual slides in template for real-world usage
        self._learn_from_template_slides()

        # Step 3: Vision-based analysis (if enabled)
        if self.enable_vision and template_path:
            try:
                vision_analyzer = VisionTemplateAnalyzer(self.vision_model)
                vision_insights = await vision_analyzer.analyze_template_visually(template_path)

                # Merge vision insights into XML-learned layouts
                xml_layouts = self._merge_vision_insights(xml_layouts, vision_insights)
                logger.info("Vision analysis completed and merged", num_slides=len(vision_insights))
            except Exception as e:
                logger.warning(f"Vision analysis failed, using XML-only: {e}")

        self.learned_layouts = xml_layouts
        return xml_layouts

    def _analyze_all_layouts_xml(self) -> Dict[str, LearnedLayout]:
        """Analyze all slide layouts using XML parsing."""
        layouts = {}

        for slide_layout in self.prs.slide_layouts:
            try:
                layout_analysis = self._analyze_layout(slide_layout)
                layouts[slide_layout.name] = layout_analysis
                logger.debug(
                    f"Analyzed layout: {slide_layout.name}",
                    layout_type=layout_analysis.layout_type,
                    max_title_chars=layout_analysis.max_title_chars,
                    typical_bullets=layout_analysis.typical_bullet_count,
                )
            except Exception as e:
                logger.warning(f"Failed to analyze layout {slide_layout.name}: {e}")
                # Create a default layout
                layouts[slide_layout.name] = LearnedLayout(
                    layout_name=slide_layout.name,
                    layout_type='content',
                )

        return layouts

    def _analyze_layout(self, slide_layout: "SlideLayout") -> LearnedLayout:
        """Deep analysis of a single slide layout."""
        from pptx.enum.shapes import PP_PLACEHOLDER, MSO_SHAPE_TYPE

        layout_type = self._detect_layout_type(slide_layout)

        # Find all placeholders and their purposes
        title_zone = None
        content_zones = []
        image_zone = None
        branding_zones = []

        for shape in slide_layout.shapes:
            zone = self._get_shape_zone(shape)

            if self._is_branding_element(shape):
                branding_zones.append(zone)
            elif hasattr(shape, 'placeholder_format') and shape.placeholder_format:
                ph_type = shape.placeholder_format.type
                if ph_type in [PP_PLACEHOLDER.TITLE, PP_PLACEHOLDER.CENTER_TITLE]:
                    title_zone = zone
                elif ph_type == PP_PLACEHOLDER.PICTURE:
                    image_zone = zone
                elif ph_type in [PP_PLACEHOLDER.BODY, PP_PLACEHOLDER.OBJECT]:
                    content_zones.append(zone)
            elif hasattr(shape, 'shape_type'):
                # Check for pictures/logos that aren't placeholders
                if shape.shape_type == MSO_SHAPE_TYPE.PICTURE:
                    if self._is_likely_logo(shape):
                        branding_zones.append(zone)

        # Calculate safe title width considering branding
        safe_title_zone = self._calculate_safe_title_zone(title_zone, branding_zones)

        # Estimate content constraints from zone sizes
        typical_bullet_count, typical_bullet_length = self._estimate_content_constraints(content_zones)

        return LearnedLayout(
            layout_name=slide_layout.name,
            layout_type=layout_type,
            title_zone=safe_title_zone,
            content_zones=content_zones,
            image_zone=image_zone,
            branding_zones=branding_zones,
            typical_bullet_count=typical_bullet_count,
            typical_bullet_length=typical_bullet_length,
            max_title_chars=safe_title_zone.get('max_chars', 50) if safe_title_zone else 50,
            has_picture_placeholder=image_zone is not None,
            has_footer=any(z.get('top', 0) > self.slide_height * 0.9 for z in branding_zones),
            has_page_number=self._detect_page_number(slide_layout),
            font_size_title=self._detect_title_font_size(slide_layout),
            font_size_bullets=self._detect_bullet_font_size(slide_layout),
            bullet_indent=self._detect_bullet_indent(slide_layout),
        )

    def _detect_layout_type(self, slide_layout: "SlideLayout") -> str:
        """Detect the type of layout based on placeholder arrangement."""
        from pptx.enum.shapes import PP_PLACEHOLDER

        placeholders = list(slide_layout.placeholders)
        ph_types = []
        for ph in placeholders:
            if hasattr(ph, 'placeholder_format') and ph.placeholder_format:
                ph_types.append(ph.placeholder_format.type)

        # Title slide: has center title, no body
        if PP_PLACEHOLDER.CENTER_TITLE in ph_types:
            return 'title'

        # Section header: title only, no body
        if PP_PLACEHOLDER.TITLE in ph_types and PP_PLACEHOLDER.BODY not in ph_types:
            body_count = sum(1 for pt in ph_types if pt in [PP_PLACEHOLDER.BODY, PP_PLACEHOLDER.OBJECT])
            if body_count == 0:
                return 'section_header'

        # Check for two-column layout
        body_placeholders = []
        for ph in placeholders:
            if hasattr(ph, 'placeholder_format') and ph.placeholder_format:
                if ph.placeholder_format.type == PP_PLACEHOLDER.BODY:
                    body_placeholders.append(ph)
        if len(body_placeholders) >= 2:
            return 'two_column'

        # Image + text layout
        has_picture = PP_PLACEHOLDER.PICTURE in ph_types
        has_body = PP_PLACEHOLDER.BODY in ph_types
        if has_picture and has_body:
            return 'image_text'

        # Standard content
        if has_body:
            return 'content'

        return 'blank'

    def _get_shape_zone(self, shape) -> Dict[str, Any]:
        """Get the zone (position and size) of a shape."""
        return {
            'left': shape.left,
            'top': shape.top,
            'width': shape.width,
            'height': shape.height,
            'left_inches': shape.left / EMU_PER_INCH,
            'top_inches': shape.top / EMU_PER_INCH,
            'width_inches': shape.width / EMU_PER_INCH,
            'height_inches': shape.height / EMU_PER_INCH,
        }

    def _is_branding_element(self, shape) -> bool:
        """Detect if shape is a branding element (logo, footer, branding bar)."""
        from pptx.enum.shapes import MSO_SHAPE_TYPE

        # Images in corners are likely logos
        if hasattr(shape, 'shape_type') and shape.shape_type == MSO_SHAPE_TYPE.PICTURE:
            return self._is_likely_logo(shape)

        # Small text boxes at bottom are likely footers
        if hasattr(shape, 'has_text_frame') and shape.has_text_frame:
            if shape.top > self.slide_height * 0.9:
                return True

        # PHASE 10 FIX: Detect branding bars at bottom of slide
        # These are typically rectangles with solid fills in the bottom 30% of the slide
        # that span most of the slide width (template branding zone)
        # NOTE: This sets content_bottom, but NO LONGER overrides safe_footer_y
        # Footer position should come from template master or stay at slide bottom
        if hasattr(shape, 'shape_type'):
            shape_bottom = shape.top + shape.height

            # Check if shape is in bottom 30% of slide
            if shape.top > self.slide_height * 0.7:
                # Wide shape (spans >60% of slide) - likely a branding bar
                if shape.width > self.slide_width * 0.6:
                    # Update safe content bottom to be above this branding element
                    # BUT don't change footer position - footer goes AT THE BOTTOM
                    safe_y = shape.top - int(0.1 * EMU_PER_INCH)  # 0.1" margin above branding
                    if self.detected_content_bottom_y is None or safe_y < self.detected_content_bottom_y:
                        self.detected_content_bottom_y = safe_y
                        # PHASE 10: Do NOT override safe_footer_y here
                        # Footer should be at bottom of slide, not above branding bar
                        logger.debug(
                            f"Detected branding bar at y={shape.top / EMU_PER_INCH:.2f}in, "
                            f"setting content_bottom_y={safe_y / EMU_PER_INCH:.2f}in (footer unchanged)"
                        )
                    return True

        return False

    def _extract_footer_position_from_master(self) -> None:
        """PHASE 10: Extract footer placeholder position from slide master.

        Reads the actual footer/slide number placeholder positions from the
        template's slide master to ensure our generated footers match the
        template's intended positioning.
        """
        from pptx.enum.shapes import PP_PLACEHOLDER

        try:
            # Get the slide master (first one, which is the main master)
            if not self.prs.slide_masters:
                logger.debug("No slide master found, using default footer position")
                return

            slide_master = self.prs.slide_masters[0]

            # Look for footer and slide number placeholders in the master
            footer_types = [
                PP_PLACEHOLDER.FOOTER,
                PP_PLACEHOLDER.SLIDE_NUMBER,
                PP_PLACEHOLDER.DATE_TIME,
            ]

            for shape in slide_master.shapes:
                # Check if it's a placeholder
                if not hasattr(shape, 'placeholder_format') or shape.placeholder_format is None:
                    continue

                ph_type = shape.placeholder_format.type
                if ph_type in footer_types:
                    # Found a footer-related placeholder
                    if self.footer_placeholder_y is None or shape.top > self.footer_placeholder_y:
                        self.footer_placeholder_y = shape.top
                        # Use this as the safe footer Y position
                        self.safe_footer_y = shape.top
                        logger.info(
                            f"Found footer placeholder in master at y={shape.top / EMU_PER_INCH:.2f}in, "
                            f"type={ph_type}"
                        )

            # If we found footer placeholders, use them; otherwise check for text at bottom
            if self.footer_placeholder_y is None:
                # Look for any small text boxes at bottom of master (custom footer positioning)
                for shape in slide_master.shapes:
                    if hasattr(shape, 'has_text_frame') and shape.has_text_frame:
                        # Text in bottom 10% of slide that's small (likely footer)
                        if shape.top > self.slide_height * 0.9:
                            if shape.height < self.slide_height * 0.1:  # Less than 10% of slide height
                                self.footer_placeholder_y = shape.top
                                self.safe_footer_y = shape.top
                                logger.info(
                                    f"Found footer-like text in master at y={shape.top / EMU_PER_INCH:.2f}in"
                                )
                                break

        except Exception as e:
            logger.warning(f"Failed to extract footer position from master: {e}")
            # Keep using default position

    def _is_likely_logo(self, shape) -> bool:
        """Check if a picture shape is likely a logo.

        Also updates detected_logo_x with the minimum x position of logos found,
        which is used to calculate safe title width.
        """
        is_logo = False

        # Top-right corner (common logo zone)
        if shape.left > self.slide_width * 0.6 and shape.top < self.slide_height * 0.2:
            is_logo = True
            # Update detected_logo_x to track where logos start
            if self.detected_logo_x is None or shape.left < self.detected_logo_x:
                self.detected_logo_x = shape.left
                logger.debug(
                    f"Logo detected at x={shape.left / EMU_PER_INCH:.2f}in, "
                    f"updating detected_logo_x"
                )

        # Top-left corner (secondary logo)
        if shape.left < self.slide_width * 0.2 and shape.top < self.slide_height * 0.15:
            is_logo = True

        # Bottom corners (footer logos)
        if shape.top > self.slide_height * 0.85:
            is_logo = True

        return is_logo

    def _calculate_safe_title_zone(
        self,
        title_zone: Optional[Dict[str, Any]],
        branding_zones: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Calculate title zone that doesn't overlap with branding."""
        from pptx.util import Inches

        if not title_zone:
            # Default title zone
            default_width = self.slide_width - Inches(1)
            return {
                'left': Inches(0.5),
                'top': Inches(0.3),
                'width': default_width,
                'height': Inches(1),
                'max_chars': 50,
                'left_inches': 0.5,
                'top_inches': 0.3,
                'width_inches': default_width / EMU_PER_INCH,
                'height_inches': 1.0,
            }

        # Safely get width and left with defaults (title_zone may be missing keys)
        safe_width = title_zone.get('width', self.slide_width * 0.7)
        title_left = title_zone.get('left', int(self.slide_width * 0.05))

        # Check for branding in title area (top 20% of slide)
        for brand in branding_zones:
            brand_top = brand.get('top', 0)
            brand_left = brand.get('left', 0)

            # Only consider branding in top area
            if brand_top < self.slide_height * 0.25:
                # Logo on right side - reduce title width
                if brand_left > self.slide_width * 0.5:
                    from pptx.util import Inches
                    margin = Inches(0.3)
                    available_width = brand_left - title_left - margin
                    safe_width = min(safe_width, available_width)
                    logger.debug(
                        f"Reducing title width due to right-side branding",
                        original_width_inches=title_zone['width'] / EMU_PER_INCH,
                        safe_width_inches=safe_width / EMU_PER_INCH,
                    )

        # Update the zone with safe width
        safe_zone = title_zone.copy()
        safe_zone['width'] = safe_width
        safe_zone['width_inches'] = safe_width / EMU_PER_INCH

        # Calculate max characters (~10 chars per inch for typical fonts)
        chars_per_inch = 9  # Conservative estimate
        safe_zone['max_chars'] = int((safe_width / EMU_PER_INCH) * chars_per_inch)

        return safe_zone

    def _estimate_content_constraints(self, content_zones: List[Dict[str, Any]]) -> tuple:
        """Estimate bullet count and length from content zone sizes."""
        if not content_zones:
            return 6, 70  # Defaults

        # Use the largest content zone
        largest_zone = max(content_zones, key=lambda z: z.get('height', 0) * z.get('width', 0))

        height_inches = largest_zone.get('height_inches', 5.0)
        width_inches = largest_zone.get('width_inches', 10.0)

        # Estimate bullets based on height (~0.6 inch per bullet with spacing)
        bullet_spacing = 0.5
        estimated_bullets = int(height_inches / bullet_spacing)
        typical_bullet_count = max(3, min(8, estimated_bullets))

        # Estimate bullet length based on width (~10 chars per inch)
        chars_per_inch = 9
        typical_bullet_length = int(width_inches * chars_per_inch)
        typical_bullet_length = max(50, min(100, typical_bullet_length))

        return typical_bullet_count, typical_bullet_length

    def _learn_from_template_slides(self):
        """Learn from actual slides in the template (not just layouts)."""
        for slide in self.prs.slides:
            try:
                layout_name = slide.slide_layout.name
                if layout_name not in self.learned_layouts:
                    continue

                layout = self.learned_layouts[layout_name]

                # Analyze actual content usage in template slides
                for shape in slide.shapes:
                    # Look for branding elements in actual slides
                    if self._is_branding_element(shape):
                        zone = self._get_shape_zone(shape)
                        if zone not in layout.branding_zones:
                            layout.branding_zones.append(zone)
                            # Recalculate safe title zone
                            layout.title_zone = self._calculate_safe_title_zone(
                                layout.title_zone, layout.branding_zones
                            )
                            layout.max_title_chars = layout.title_zone.get('max_chars', 50)

                    # Count actual bullets used
                    if hasattr(shape, 'has_text_frame') and shape.has_text_frame:
                        bullet_count = len([
                            p for p in shape.text_frame.paragraphs
                            if p.text.strip()
                        ])
                        if bullet_count > 0 and bullet_count > layout.typical_bullet_count:
                            # Update based on actual usage
                            layout.typical_bullet_count = bullet_count

            except Exception as e:
                logger.debug(f"Failed to learn from slide: {e}")

    def _detect_page_number(self, slide_layout: "SlideLayout") -> bool:
        """Detect if layout has page number placeholder."""
        from pptx.enum.shapes import PP_PLACEHOLDER

        for ph in slide_layout.placeholders:
            if hasattr(ph, 'placeholder_format') and ph.placeholder_format:
                if ph.placeholder_format.type == PP_PLACEHOLDER.SLIDE_NUMBER:
                    return True
        return False

    def _detect_title_font_size(self, slide_layout: "SlideLayout") -> int:
        """Detect title font size from layout."""
        from pptx.enum.shapes import PP_PLACEHOLDER

        for ph in slide_layout.placeholders:
            if hasattr(ph, 'placeholder_format') and ph.placeholder_format:
                if ph.placeholder_format.type in [PP_PLACEHOLDER.TITLE, PP_PLACEHOLDER.CENTER_TITLE]:
                    try:
                        if hasattr(ph, 'text_frame') and ph.text_frame.paragraphs:
                            for para in ph.text_frame.paragraphs:
                                if para.font.size:
                                    return para.font.size.pt
                    except Exception:
                        pass
        return 32  # Default

    def _detect_bullet_font_size(self, slide_layout: "SlideLayout") -> int:
        """Detect bullet font size from layout."""
        from pptx.enum.shapes import PP_PLACEHOLDER

        for ph in slide_layout.placeholders:
            if hasattr(ph, 'placeholder_format') and ph.placeholder_format:
                if ph.placeholder_format.type == PP_PLACEHOLDER.BODY:
                    try:
                        if hasattr(ph, 'text_frame') and ph.text_frame.paragraphs:
                            for para in ph.text_frame.paragraphs:
                                if para.font.size:
                                    return para.font.size.pt
                    except Exception:
                        pass
        return 18  # Default

    def _detect_bullet_indent(self, slide_layout: "SlideLayout") -> int:
        """Detect bullet indent from layout."""
        return 0  # Default, would need deeper XML analysis

    def _merge_vision_insights(
        self,
        xml_layouts: Dict[str, LearnedLayout],
        vision_insights: Dict[int, Dict[str, Any]]
    ) -> Dict[str, LearnedLayout]:
        """Enhance XML-learned layouts with vision model insights."""
        for slide_num, insights in vision_insights.items():
            # Find matching layout by type
            insight_type = insights.get('layout_type', 'content')

            for layout_name, layout in xml_layouts.items():
                if layout.layout_type == insight_type:
                    # Update with vision-detected values
                    if 'avoid_zones' in insights:
                        for zone in insights['avoid_zones']:
                            # Convert percentage-based zones to EMUs
                            converted_zone = self._convert_vision_zone(zone)
                            if converted_zone and converted_zone not in layout.branding_zones:
                                layout.branding_zones.append(converted_zone)

                    if 'style' in insights:
                        layout.visual_style = insights['style']

                    # Recalculate safe zones with new branding info
                    layout.title_zone = self._calculate_safe_title_zone(
                        layout.title_zone, layout.branding_zones
                    )
                    layout.max_title_chars = layout.title_zone.get('max_chars', 50)
                    break

        return xml_layouts

    def _convert_vision_zone(self, vision_zone: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Convert vision-detected zone (percentage-based) to EMUs."""
        try:
            # Vision zones might be like: {"position": "top-right", "size": "small"}
            position = vision_zone.get('position', '')

            if 'top-right' in position:
                return {
                    'left': int(self.slide_width * 0.7),
                    'top': int(self.slide_height * 0.02),
                    'width': int(self.slide_width * 0.25),
                    'height': int(self.slide_height * 0.1),
                }
            elif 'top-left' in position:
                return {
                    'left': int(self.slide_width * 0.02),
                    'top': int(self.slide_height * 0.02),
                    'width': int(self.slide_width * 0.25),
                    'height': int(self.slide_height * 0.1),
                }
            elif 'bottom' in position:
                return {
                    'left': 0,
                    'top': int(self.slide_height * 0.9),
                    'width': self.slide_width,
                    'height': int(self.slide_height * 0.1),
                }
        except Exception:
            pass
        return None

    def get_layout_for_section(
        self,
        section_order: int,
        total_sections: int,
        has_image: bool = False,
        section_title: str = ""
    ) -> LearnedLayout:
        """Select best layout for a section based on its position and content.

        Args:
            section_order: Position of section (0-indexed)
            total_sections: Total number of sections
            has_image: Whether section has an image
            section_title: Title of the section (for keyword matching)

        Returns:
            Most appropriate LearnedLayout for this section
        """
        layouts = list(self.learned_layouts.values())

        if not layouts:
            # Return a default layout if none learned
            return LearnedLayout(layout_name="default", layout_type="content")

        # Title slide for first section
        if section_order == 0:
            title_layouts = [l for l in layouts if l.layout_type == 'title']
            if title_layouts:
                return title_layouts[0]

        # Section header for major divisions (every 4-5 slides)
        if section_order > 0 and total_sections > 5 and section_order % 4 == 0:
            section_layouts = [l for l in layouts if l.layout_type == 'section_header']
            if section_layouts:
                return section_layouts[0]

        # Image + text if section has image
        if has_image:
            image_layouts = [
                l for l in layouts
                if l.layout_type == 'image_text' or l.has_picture_placeholder
            ]
            if image_layouts:
                return image_layouts[0]

        # Default to content layout
        content_layouts = [l for l in layouts if l.layout_type == 'content']
        if content_layouts:
            return content_layouts[0]

        return layouts[0]


class VisionTemplateAnalyzer:
    """Uses vision model to understand template styling from rendered slides.

    This is an OPTIONAL component that provides deeper template understanding
    by actually rendering slides and analyzing them visually.

    Enable via config: enable_template_vision_analysis = True
    """

    def __init__(self, vision_model: str = "auto"):
        """Initialize the vision analyzer.

        Args:
            vision_model: Vision model to use (auto, ollama-llava, claude-3-sonnet, gpt-4-vision)
        """
        self.vision_model = vision_model

    async def analyze_template_visually(self, template_path: str) -> Dict[int, Dict[str, Any]]:
        """Render each template slide and analyze with vision model.

        Args:
            template_path: Path to the template PPTX file

        Returns:
            Dictionary mapping slide numbers to visual analysis results
        """
        import shutil

        visual_analyses = {}
        self._temp_render_dir = None  # Initialize for cleanup

        try:
            # Render all slides to images
            slide_images = await self._render_slides_to_images(template_path)

            for slide_num, image_path in slide_images.items():
                try:
                    analysis = await self._analyze_slide_image(image_path, slide_num)
                    visual_analyses[slide_num] = analysis
                except Exception as e:
                    logger.warning(f"Vision analysis failed for slide {slide_num}: {e}")

        except Exception as e:
            logger.warning(f"Could not render template for vision analysis: {e}")

        finally:
            # Clean up temp directory after all analyses are done
            if self._temp_render_dir:
                try:
                    shutil.rmtree(self._temp_render_dir, ignore_errors=True)
                    logger.debug(f"Cleaned up vision temp directory: {self._temp_render_dir}")
                except Exception:
                    pass

        return visual_analyses

    async def _render_slides_to_images(self, template_path: str) -> Dict[int, str]:
        """Render template slides to images using LibreOffice + pdf2image.

        LibreOffice's direct PNG export only produces one image for the first slide.
        We need to convert to PDF first, then use pdf2image to extract per-page images.

        Returns:
            Dictionary mapping slide numbers to image paths

        Note: The returned paths are in a temp directory that persists until
        the process ends. Caller should clean up if needed.
        """
        import subprocess
        import tempfile
        import os
        import shutil

        slide_images = {}

        try:
            # Create temp directory that persists (not auto-deleted)
            # We need the files to exist after this function returns
            temp_dir = tempfile.mkdtemp(prefix="pptx_vision_")

            # Step 1: Convert PPTX to PDF using LibreOffice
            # This preserves all slides in a single PDF
            result = subprocess.run([
                'soffice',
                '--headless',
                '--convert-to', 'pdf',
                '--outdir', temp_dir,
                template_path
            ], capture_output=True, timeout=120)

            if result.returncode != 0:
                logger.warning(f"LibreOffice PDF conversion failed: {result.stderr}")
                shutil.rmtree(temp_dir, ignore_errors=True)
                return slide_images

            # Find the generated PDF
            pdf_path = None
            base_name = os.path.splitext(os.path.basename(template_path))[0]
            expected_pdf = os.path.join(temp_dir, f"{base_name}.pdf")
            if os.path.exists(expected_pdf):
                pdf_path = expected_pdf
            else:
                # Search for any PDF
                for f in os.listdir(temp_dir):
                    if f.endswith('.pdf'):
                        pdf_path = os.path.join(temp_dir, f)
                        break

            if not pdf_path:
                logger.warning("No PDF generated from template")
                shutil.rmtree(temp_dir, ignore_errors=True)
                return slide_images

            # Step 2: Convert PDF pages to images
            # Try pdf2image first (requires poppler), fall back to ImageMagick
            try:
                from pdf2image import convert_from_path
                images = convert_from_path(pdf_path, dpi=150, fmt='png')
                for i, image in enumerate(images):
                    image_path = os.path.join(temp_dir, f"slide_{i + 1}.png")
                    image.save(image_path, 'PNG')
                    slide_images[i + 1] = image_path
                    logger.debug(f"Rendered slide {i + 1} to {image_path}")
            except ImportError:
                logger.debug("pdf2image not available, trying ImageMagick")
                # Fall back to ImageMagick convert
                result = subprocess.run([
                    'convert',
                    '-density', '150',
                    pdf_path,
                    os.path.join(temp_dir, 'slide_%d.png')
                ], capture_output=True, timeout=120)

                if result.returncode == 0:
                    for filename in sorted(os.listdir(temp_dir)):
                        if filename.startswith('slide_') and filename.endswith('.png'):
                            try:
                                # ImageMagick uses 0-indexed slide numbers
                                slide_num = int(filename.replace('slide_', '').replace('.png', '')) + 1
                            except ValueError:
                                slide_num = len(slide_images) + 1
                            slide_images[slide_num] = os.path.join(temp_dir, filename)
                else:
                    logger.warning(f"ImageMagick conversion failed: {result.stderr}")

            # Clean up PDF file (keep only PNGs)
            if pdf_path and os.path.exists(pdf_path):
                os.remove(pdf_path)

            # Store temp_dir for cleanup later
            if slide_images:
                self._temp_render_dir = temp_dir
                logger.info(f"Vision analysis: rendered {len(slide_images)} slides to images")
            else:
                shutil.rmtree(temp_dir, ignore_errors=True)

        except subprocess.TimeoutExpired:
            logger.warning("LibreOffice/ImageMagick conversion timed out")
        except FileNotFoundError as e:
            logger.warning(f"Required tool not found for vision analysis: {e}")
        except Exception as e:
            logger.warning(f"Failed to render slides to images: {e}")

        return slide_images

    async def _analyze_slide_image(self, image_path: str, slide_num: int) -> Dict[str, Any]:
        """Use vision model to understand slide layout and styling."""
        try:
            from backend.services.llm import EnhancedLLMFactory
            import base64

            # Read and encode image
            with open(image_path, "rb") as f:
                image_data = base64.standard_b64encode(f.read()).decode("utf-8")

            prompt = """Analyze this presentation slide and identify:

1. LAYOUT TYPE: Is this a title slide, content slide, section header, two-column, or image+text?

2. BRANDING ELEMENTS: Identify any logos, footers, page numbers and their positions:
   - Logo position (top-left, top-right, bottom-left, bottom-right)
   - Footer text and position
   - Any other fixed branding elements

3. CONTENT ZONES: Where is text content placed?
   - Title area: position and approximate size
   - Main content area: position and approximate size
   - Any secondary content areas

4. VISUAL STYLE:
   - Primary colors used
   - Font style (modern, classic, bold, minimal)
   - Overall design density (sparse, moderate, dense)

5. SAFE ZONES: Based on the layout, what areas should be avoided for new content?

Return as JSON:
{
    "layout_type": "content",
    "branding": {
        "logo": {"position": "top-right", "size": "small"},
        "footer": {"present": true, "position": "bottom"}
    },
    "title_zone": {"top": "5%", "left": "5%", "width": "70%", "height": "15%"},
    "content_zone": {"top": "25%", "left": "5%", "width": "90%", "height": "65%"},
    "style": {"colors": ["#FF6B00", "#333333"], "density": "moderate"},
    "avoid_zones": [{"reason": "logo", "position": "top-right"}]
}
"""

            # Get vision-capable model
            if self.vision_model == "auto":
                llm, _ = await EnhancedLLMFactory.get_chat_model_for_operation(
                    operation="vision_review",
                    prefer_fast=True,
                )
            else:
                llm, _ = await EnhancedLLMFactory.get_chat_model_for_operation(
                    operation="vision_review",
                    model_override=self.vision_model,
                )

            from langchain_core.messages import HumanMessage

            message = HumanMessage(
                content=[
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{image_data}"}
                    },
                    {"type": "text", "text": prompt}
                ]
            )

            response = await llm.ainvoke([message])
            # Safely extract content from response
            if response and hasattr(response, 'content') and response.content:
                return self._parse_vision_response(str(response.content))
            return {"layout_type": "content", "error": "Empty response from vision model"}

        except Exception as e:
            logger.warning(f"Vision analysis for slide {slide_num} failed: {e}")
            return {"layout_type": "content", "error": str(e)}

    def _parse_vision_response(self, response: str) -> Dict[str, Any]:
        """Parse JSON from vision model response."""
        import json
        import re

        # Extract JSON from response - use non-greedy pattern to get first JSON object
        json_match = re.search(r'\{[\s\S]*?\}', response)
        if json_match:
            try:
                return json.loads(json_match.group())
            except json.JSONDecodeError:
                # Try greedy match as fallback for nested JSON
                json_match_greedy = re.search(r'\{[\s\S]*\}', response)
                if json_match_greedy:
                    try:
                        return json.loads(json_match_greedy.group())
                    except json.JSONDecodeError:
                        pass

        # Return default if parsing fails
        return {"layout_type": "content", "error": "Could not parse vision response"}


class SlideContentPlanner:
    """Plans content for each slide based on its learned layout.

    This class takes learned layouts and creates specific content plans
    for each slide, including:
    - Per-slide character limits
    - Bullet count constraints
    - Layout-specific formatting instructions
    """

    def __init__(self, layout_learner: TemplateLayoutLearner):
        """Initialize the content planner.

        Args:
            layout_learner: The template layout learner with learned layouts
        """
        self.learner = layout_learner

    def plan_slide_content(
        self,
        section_order: int,
        total_sections: int,
        has_image: bool = False,
        section_title: str = ""
    ) -> SlideContentPlan:
        """Create content plan specific to this slide's layout.

        Args:
            section_order: Position of section (0-indexed)
            total_sections: Total number of sections
            has_image: Whether section will have an image
            section_title: Title of the section

        Returns:
            SlideContentPlan with specific constraints for this slide
        """
        layout = self.learner.get_layout_for_section(
            section_order, total_sections, has_image, section_title
        )

        return SlideContentPlan(
            layout=layout,
            title_constraints={
                'max_chars': layout.max_title_chars,
                'max_width': layout.title_zone.get('width', 0),
                'font_size': layout.font_size_title,
            },
            content_constraints={
                'max_bullets': layout.typical_bullet_count,
                'max_bullet_chars': layout.typical_bullet_length,
                'content_zones': layout.content_zones,
                'font_size': layout.font_size_bullets,
            },
            image_placement=layout.image_zone,
            avoid_zones=layout.branding_zones,
            visual_style=layout.visual_style,
        )

    def get_format_instructions(self, plan: SlideContentPlan, section_order: int) -> str:
        """Generate LLM format instructions for this slide's constraints.

        Args:
            plan: The slide content plan
            section_order: Position of section (0-indexed)

        Returns:
            String with formatting instructions for the LLM
        """
        layout_type = plan.layout.layout_type.upper()
        title_max = plan.title_constraints.get('max_chars', 50)
        bullet_max = plan.content_constraints.get('max_bullet_chars', 120)  # PHASE 11: Increased
        max_bullets = plan.content_constraints.get('max_bullets', 6)

        base_instructions = f"""
SLIDE LAYOUT: {layout_type}
STRICT CONSTRAINTS (DO NOT EXCEED):
- Title: MAXIMUM {title_max} characters
- Each bullet: MAXIMUM {bullet_max} characters
- Total bullets: MAXIMUM {max_bullets}

CRITICAL: If any text exceeds these limits, the slide will look broken.
Every bullet MUST be a complete sentence that fits within {bullet_max} characters.
"""

        # Add layout-specific guidance
        if plan.layout.layout_type == 'title':
            return base_instructions + """
This is a TITLE SLIDE - keep it impactful and minimal:
- Use a powerful, concise headline
- Optional subtitle (keep short)
- NO bullet points on title slides
"""

        elif plan.layout.layout_type == 'section_header':
            return base_instructions + """
This is a SECTION HEADER - marks a new section:
- Clear section title
- Brief description (1-2 lines max)
- Transitions from previous content
"""

        elif plan.layout.layout_type == 'two_column':
            return base_instructions + """
This is a TWO-COLUMN LAYOUT:
- Split content into two balanced columns
- Use LEFT: and RIGHT: markers to separate
- Each column should have similar content volume
"""

        elif plan.layout.layout_type == 'image_text':
            return base_instructions + """
This is an IMAGE + TEXT LAYOUT:
- Image will be placed on one side
- Keep text concise on the other side
- Bullets should complement the visual
- Fewer bullets (3-5) work best here
"""

        else:  # Standard content
            return base_instructions + """
STANDARD CONTENT SLIDE:
- Clear, hierarchical bullet points
- Main points with optional sub-bullets
- Each bullet is a complete thought
- Use "• " for main points, "  ◦ " for sub-points
"""

    def validate_content_fits(self, content: str, plan: SlideContentPlan) -> List[str]:
        """Validate that generated content fits the slide constraints.

        Args:
            content: The generated content
            plan: The slide content plan

        Returns:
            List of validation issues (empty if content fits)
        """
        issues = []

        # Parse content into title and bullets
        lines = content.strip().split('\n')
        title = lines[0] if lines else ""
        bullets = [l.strip() for l in lines[1:] if l.strip()]

        # Check title length
        title_max = plan.title_constraints.get('max_chars', 50)
        if len(title) > title_max:
            issues.append(f"title_too_long:{len(title)}/{title_max}")

        # Check bullet lengths
        bullet_max = plan.content_constraints.get('max_bullet_chars', 120)  # PHASE 11: Increased
        for i, bullet in enumerate(bullets):
            # Strip bullet markers for length check
            clean_bullet = bullet.lstrip('•◦*- ')
            if len(clean_bullet) > bullet_max:
                issues.append(f"bullet_{i}_too_long:{len(clean_bullet)}/{bullet_max}")

        # Check bullet count
        max_bullets = plan.content_constraints.get('max_bullets', 6)
        if len(bullets) > max_bullets:
            issues.append(f"too_many_bullets:{len(bullets)}/{max_bullets}")

        return issues
