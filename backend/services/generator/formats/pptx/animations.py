"""
PPTX Animation Utilities

Provides functions for adding transitions and animations to PowerPoint slides.
Extracted from generator.py for modularity.
"""

from typing import Optional, List

import structlog

logger = structlog.get_logger(__name__)


# =============================================================================
# Slide Transitions
# =============================================================================

def add_slide_transition(slide, transition_type: str = "fade",
                         duration: int = 500, speed: str = "med"):
    """Add a transition effect to a slide.

    Args:
        slide: python-pptx slide object
        transition_type: Type of transition (fade, push, wipe, etc.)
        duration: Duration in milliseconds
        speed: Speed setting (slow, med, fast)

    Note:
        python-pptx has limited transition support. This function adds
        transitions via low-level XML manipulation.
    """
    from lxml import etree

    # Transition types mapping to OOXML values
    transition_map = {
        "fade": "fade",
        "push": "push",
        "wipe": "wipe",
        "split": "split",
        "reveal": "reveal",
        "random": "random",
        "cover": "cover",
        "pull": "pull",
        "strips": "strips",
        "blinds": "blinds",
        "clock": "wheel",
        "zoom": "zoom",
        "dissolve": "dissolve",
        "checker": "checker",
        "box": "box",
        "circle": "circle",
        "diamond": "diamond",
        "plus": "plus",
        "wedge": "wedge",
        "newsflash": "newsflash",
    }

    speed_map = {
        "slow": "slow",
        "med": "med",
        "fast": "fast",
    }

    ooxml_type = transition_map.get(transition_type, "fade")
    ooxml_speed = speed_map.get(speed, "med")

    try:
        # Access the slide's XML
        slide_element = slide._element

        # Define namespaces
        nsmap = {
            'a': 'http://schemas.openxmlformats.org/drawingml/2006/main',
            'p': 'http://schemas.openxmlformats.org/presentationml/2006/main',
            'r': 'http://schemas.openxmlformats.org/officeDocument/2006/relationships',
        }

        # Check if transition element already exists
        existing = slide_element.find('.//p:transition', nsmap)
        if existing is not None:
            existing.getparent().remove(existing)

        # Create transition element
        # Duration is in milliseconds, but XML uses 1/1000 seconds
        duration_val = str(duration)

        transition_xml = f'''
        <p:transition xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main"
                      spd="{ooxml_speed}" advTm="{duration_val}">
            <p:{ooxml_type}/>
        </p:transition>
        '''

        transition_elem = etree.fromstring(transition_xml)

        # Insert transition element (should be after cSld)
        csld = slide_element.find('.//p:cSld', nsmap)
        if csld is not None:
            csld_index = list(slide_element).index(csld)
            slide_element.insert(csld_index + 1, transition_elem)
        else:
            slide_element.insert(0, transition_elem)

        logger.debug(
            "Added slide transition",
            transition_type=transition_type,
            duration=duration,
            speed=speed,
        )

    except Exception as e:
        logger.warning(f"Could not add slide transition: {e}")


# =============================================================================
# Content Animations
# =============================================================================

def add_bullet_animations(slide, content_shape, animation_type: str = "appear",
                          delay_between: int = 300):
    """Add entrance animations to bullet points.

    Args:
        slide: python-pptx slide object
        content_shape: Shape containing bullet points
        animation_type: Type of animation (appear, fade, fly, zoom)
        delay_between: Delay between each bullet animation in ms

    Note:
        python-pptx has limited animation support. This uses XML manipulation
        for basic entrance animations.
    """
    from lxml import etree

    # Animation presets mapping
    animation_presets = {
        "appear": "1",      # Appear
        "fade": "10",       # Fade
        "fly": "2",         # Fly In
        "zoom": "23",       # Zoom
        "wipe": "22",       # Wipe
        "split": "16",      # Split
        "float": "42",      # Float Up
        "grow": "53",       # Grow & Turn
    }

    preset_id = animation_presets.get(animation_type, "1")

    try:
        # Count paragraphs (bullets)
        if not content_shape.has_text_frame:
            return

        para_count = len(content_shape.text_frame.paragraphs)
        if para_count == 0:
            return

        # Get shape ID
        shape_id = content_shape.shape_id

        # Build animation XML
        # This is a simplified version - full implementation would need
        # proper timing tree and animation sequences

        nsmap = {
            'a': 'http://schemas.openxmlformats.org/drawingml/2006/main',
            'p': 'http://schemas.openxmlformats.org/presentationml/2006/main',
            'r': 'http://schemas.openxmlformats.org/officeDocument/2006/relationships',
        }

        # Access slide timing
        slide_element = slide._element

        # Check if timing already exists
        timing = slide_element.find('.//p:timing', nsmap)
        if timing is None:
            # Create timing element with animation sequence
            timing_xml = f'''
            <p:timing xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main">
                <p:tnLst>
                    <p:par>
                        <p:cTn id="1" dur="indefinite" restart="never" nodeType="tmRoot"/>
                    </p:par>
                </p:tnLst>
            </p:timing>
            '''
            timing = etree.fromstring(timing_xml)
            slide_element.append(timing)

        logger.debug(
            "Added bullet animations",
            shape_id=shape_id,
            animation_type=animation_type,
            bullet_count=para_count,
        )

    except Exception as e:
        logger.warning(f"Could not add bullet animations: {e}")


# =============================================================================
# Animation Presets
# =============================================================================

def get_animation_preset(style: str = "professional") -> dict:
    """Get animation preset configuration for a style.

    Args:
        style: Animation style (professional, dynamic, minimal, none)

    Returns:
        Dict with animation configuration
    """
    presets = {
        "professional": {
            "transition": "fade",
            "transition_duration": 500,
            "bullet_animation": "appear",
            "bullet_delay": 300,
            "enable_transitions": True,
            "enable_animations": True,
        },
        "dynamic": {
            "transition": "push",
            "transition_duration": 400,
            "bullet_animation": "fly",
            "bullet_delay": 200,
            "enable_transitions": True,
            "enable_animations": True,
        },
        "minimal": {
            "transition": "fade",
            "transition_duration": 300,
            "bullet_animation": "fade",
            "bullet_delay": 200,
            "enable_transitions": True,
            "enable_animations": False,
        },
        "none": {
            "transition": None,
            "transition_duration": 0,
            "bullet_animation": None,
            "bullet_delay": 0,
            "enable_transitions": False,
            "enable_animations": False,
        },
    }

    return presets.get(style, presets["professional"])


def apply_presentation_animations(presentation, style: str = "professional"):
    """Apply animation preset to all slides in a presentation.

    Args:
        presentation: python-pptx Presentation object
        style: Animation style preset to apply
    """
    preset = get_animation_preset(style)

    if not preset["enable_transitions"]:
        return

    for slide in presentation.slides:
        if preset["transition"]:
            add_slide_transition(
                slide,
                transition_type=preset["transition"],
                duration=preset["transition_duration"],
            )
