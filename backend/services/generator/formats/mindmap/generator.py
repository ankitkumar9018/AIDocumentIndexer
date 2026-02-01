"""
Mind Map Generator
==================

Generates interactive mind maps from document outlines.
Uses D3.js for rendering and provides export to SVG/PNG.
"""

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import structlog

from ...models import GenerationJob, OutputFormat, Section
from ..base import BaseFormatGenerator
from ..factory import register_generator

logger = structlog.get_logger(__name__)


@dataclass
class MindMapNode:
    """A node in the mind map."""
    id: str
    label: str
    children: List["MindMapNode"]
    color: Optional[str] = None
    icon: Optional[str] = None
    notes: Optional[str] = None
    collapsed: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        result = {
            "id": self.id,
            "label": self.label,
            "children": [child.to_dict() for child in self.children],
        }
        if self.color:
            result["color"] = self.color
        if self.icon:
            result["icon"] = self.icon
        if self.notes:
            result["notes"] = self.notes
        if self.collapsed:
            result["collapsed"] = self.collapsed
        return result


# Color palette for mind map branches
BRANCH_COLORS = [
    "#4A90D9",  # Blue
    "#50C878",  # Green
    "#FF6B6B",  # Red
    "#FFB347",  # Orange
    "#9B59B6",  # Purple
    "#1ABC9C",  # Teal
    "#F39C12",  # Yellow
    "#E91E63",  # Pink
]


@register_generator(OutputFormat.MINDMAP)
class MindMapGenerator(BaseFormatGenerator):
    """
    Generates interactive mind maps from document content.

    Features:
    - Interactive pan/zoom
    - Collapsible branches
    - Export to SVG/PNG
    - Customizable themes
    - Search functionality
    """

    format_type = OutputFormat.MINDMAP
    file_extension = ".html"

    async def generate(self, job: GenerationJob, output_filename: str) -> Path:
        """Generate an interactive mind map from the job content."""
        logger.info("Generating mind map", job_id=job.id, title=job.topic)

        # Build mind map structure from sections
        root_node = self._build_mind_map(job)

        # Generate HTML with embedded visualization
        html_content = self._generate_html(root_node, job)

        # Save to file
        output_dir = Path(os.environ.get("OUTPUT_DIR", "./output"))
        output_dir.mkdir(parents=True, exist_ok=True)

        output_path = output_dir / f"{output_filename}.html"
        output_path.write_text(html_content, encoding="utf-8")

        logger.info("Mind map generated", path=str(output_path))
        return output_path

    def _build_mind_map(self, job: GenerationJob) -> MindMapNode:
        """Build mind map structure from job sections."""
        # Root node is the document title
        root = MindMapNode(
            id="root",
            label=job.topic,
            children=[],
            color="#2C3E50",
        )

        # Group sections into branches
        for i, section in enumerate(job.sections):
            color = BRANCH_COLORS[i % len(BRANCH_COLORS)]
            branch = self._section_to_node(section, color, i)
            root.children.append(branch)

        return root

    def _section_to_node(
        self,
        section: Section,
        color: str,
        index: int,
    ) -> MindMapNode:
        """Convert a section to a mind map node."""
        node = MindMapNode(
            id=f"section_{index}",
            label=section.title,
            children=[],
            color=color,
            notes=section.content[:500] if section.content else None,
        )

        # Extract key points from content as child nodes
        if section.content:
            key_points = self._extract_key_points(section.content)
            for j, point in enumerate(key_points[:8]):  # Limit to 8 points
                child = MindMapNode(
                    id=f"section_{index}_point_{j}",
                    label=point,
                    children=[],
                    color=self._lighten_color(color),
                )
                node.children.append(child)

        return node

    def _extract_key_points(self, content: str) -> List[str]:
        """Extract key points from section content."""
        points = []
        lines = content.split("\n")

        for line in lines:
            line = line.strip()
            # Look for bullet points, numbered lists, or short sentences
            if line.startswith(("-", "•", "*", "·")):
                point = line.lstrip("-•*· ").strip()
                if 10 < len(point) < 100:
                    points.append(point)
            elif line and line[0].isdigit() and "." in line[:3]:
                point = line.split(".", 1)[-1].strip()
                if 10 < len(point) < 100:
                    points.append(point)

        return points

    def _lighten_color(self, hex_color: str) -> str:
        """Lighten a hex color by 20%."""
        hex_color = hex_color.lstrip("#")
        r = min(255, int(hex_color[0:2], 16) + 40)
        g = min(255, int(hex_color[2:4], 16) + 40)
        b = min(255, int(hex_color[4:6], 16) + 40)
        return f"#{r:02x}{g:02x}{b:02x}"

    def _generate_html(self, root: MindMapNode, job: GenerationJob) -> str:
        """Generate the complete HTML mind map."""
        data_json = json.dumps(root.to_dict(), indent=2)
        # Pre-compute escaped topic for filename (backslashes not allowed in f-string expressions)
        escaped_topic = job.topic.replace('"', '\\"')

        return f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{job.topic} - Mind Map</title>
    <script src="https://d3js.org/d3.v7.min.js"></script>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}

        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif;
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            min-height: 100vh;
            overflow: hidden;
        }}

        #mindmap {{
            width: 100vw;
            height: 100vh;
        }}

        .node circle {{
            stroke-width: 2px;
            cursor: pointer;
            transition: all 0.3s ease;
        }}

        .node circle:hover {{
            stroke-width: 3px;
            filter: brightness(1.2);
        }}

        .node text {{
            font-size: 12px;
            fill: #ffffff;
            pointer-events: none;
            text-shadow: 0 1px 2px rgba(0,0,0,0.5);
        }}

        .link {{
            fill: none;
            stroke-opacity: 0.6;
            stroke-width: 2px;
        }}

        .controls {{
            position: fixed;
            top: 20px;
            right: 20px;
            display: flex;
            flex-direction: column;
            gap: 10px;
            z-index: 1000;
        }}

        .control-btn {{
            background: rgba(255,255,255,0.1);
            border: 1px solid rgba(255,255,255,0.2);
            color: white;
            padding: 10px 15px;
            border-radius: 8px;
            cursor: pointer;
            font-size: 14px;
            transition: all 0.3s ease;
        }}

        .control-btn:hover {{
            background: rgba(255,255,255,0.2);
        }}

        .title {{
            position: fixed;
            top: 20px;
            left: 20px;
            color: white;
            font-size: 24px;
            font-weight: 600;
            text-shadow: 0 2px 4px rgba(0,0,0,0.3);
        }}

        .tooltip {{
            position: absolute;
            background: rgba(0,0,0,0.9);
            color: white;
            padding: 12px 16px;
            border-radius: 8px;
            font-size: 13px;
            max-width: 300px;
            pointer-events: none;
            opacity: 0;
            transition: opacity 0.3s ease;
            z-index: 1001;
        }}

        .tooltip.visible {{
            opacity: 1;
        }}

        .search-box {{
            position: fixed;
            top: 20px;
            left: 50%;
            transform: translateX(-50%);
            z-index: 1000;
        }}

        .search-input {{
            background: rgba(255,255,255,0.1);
            border: 1px solid rgba(255,255,255,0.2);
            color: white;
            padding: 10px 20px;
            border-radius: 20px;
            font-size: 14px;
            width: 300px;
            outline: none;
        }}

        .search-input::placeholder {{
            color: rgba(255,255,255,0.5);
        }}
    </style>
</head>
<body>
    <div class="title">{job.topic}</div>

    <div class="search-box">
        <input type="text" class="search-input" placeholder="Search nodes..." id="searchInput">
    </div>

    <div class="controls">
        <button class="control-btn" onclick="zoomIn()">Zoom In (+)</button>
        <button class="control-btn" onclick="zoomOut()">Zoom Out (-)</button>
        <button class="control-btn" onclick="resetView()">Reset View</button>
        <button class="control-btn" onclick="expandAll()">Expand All</button>
        <button class="control-btn" onclick="collapseAll()">Collapse All</button>
        <button class="control-btn" onclick="exportSVG()">Export SVG</button>
    </div>

    <div id="mindmap"></div>
    <div class="tooltip" id="tooltip"></div>

    <script>
        const data = {data_json};

        const width = window.innerWidth;
        const height = window.innerHeight;

        const svg = d3.select("#mindmap")
            .append("svg")
            .attr("width", width)
            .attr("height", height);

        const g = svg.append("g");

        const zoom = d3.zoom()
            .scaleExtent([0.1, 4])
            .on("zoom", (event) => {{
                g.attr("transform", event.transform);
            }});

        svg.call(zoom);

        // Initial transform to center
        svg.call(zoom.transform, d3.zoomIdentity.translate(width/2, height/2).scale(0.8));

        const root = d3.hierarchy(data);

        const treeLayout = d3.tree()
            .size([2 * Math.PI, Math.min(width, height) / 2 - 100])
            .separation((a, b) => (a.parent == b.parent ? 1 : 2) / a.depth);

        treeLayout(root);

        // Draw links
        const links = g.selectAll(".link")
            .data(root.links())
            .join("path")
            .attr("class", "link")
            .attr("d", d3.linkRadial()
                .angle(d => d.x)
                .radius(d => d.y))
            .attr("stroke", d => d.target.data.color || "#888");

        // Draw nodes
        const nodes = g.selectAll(".node")
            .data(root.descendants())
            .join("g")
            .attr("class", "node")
            .attr("transform", d => `rotate(${{d.x * 180 / Math.PI - 90}}) translate(${{d.y}},0)`);

        nodes.append("circle")
            .attr("r", d => d.depth === 0 ? 20 : (d.children ? 10 : 6))
            .attr("fill", d => d.data.color || "#4A90D9")
            .attr("stroke", "#fff")
            .on("click", (event, d) => {{
                if (d.children) {{
                    d._children = d.children;
                    d.children = null;
                }} else if (d._children) {{
                    d.children = d._children;
                    d._children = null;
                }}
                update();
            }})
            .on("mouseover", showTooltip)
            .on("mouseout", hideTooltip);

        nodes.append("text")
            .attr("dy", "0.31em")
            .attr("x", d => d.x < Math.PI === !d.children ? 6 : -6)
            .attr("text-anchor", d => d.x < Math.PI === !d.children ? "start" : "end")
            .attr("transform", d => d.x >= Math.PI ? "rotate(180)" : null)
            .text(d => d.data.label.length > 30 ? d.data.label.slice(0, 30) + "..." : d.data.label);

        function update() {{
            treeLayout(root);

            links.data(root.links())
                .transition()
                .duration(500)
                .attr("d", d3.linkRadial()
                    .angle(d => d.x)
                    .radius(d => d.y));

            nodes.data(root.descendants())
                .transition()
                .duration(500)
                .attr("transform", d => `rotate(${{d.x * 180 / Math.PI - 90}}) translate(${{d.y}},0)`);
        }}

        const tooltip = document.getElementById("tooltip");

        function showTooltip(event, d) {{
            if (d.data.notes) {{
                tooltip.innerHTML = `<strong>${{d.data.label}}</strong><br><br>${{d.data.notes}}`;
                tooltip.style.left = (event.pageX + 10) + "px";
                tooltip.style.top = (event.pageY + 10) + "px";
                tooltip.classList.add("visible");
            }}
        }}

        function hideTooltip() {{
            tooltip.classList.remove("visible");
        }}

        function zoomIn() {{
            svg.transition().call(zoom.scaleBy, 1.3);
        }}

        function zoomOut() {{
            svg.transition().call(zoom.scaleBy, 0.7);
        }}

        function resetView() {{
            svg.transition().call(zoom.transform, d3.zoomIdentity.translate(width/2, height/2).scale(0.8));
        }}

        function expandAll() {{
            root.descendants().forEach(d => {{
                if (d._children) {{
                    d.children = d._children;
                    d._children = null;
                }}
            }});
            update();
        }}

        function collapseAll() {{
            root.descendants().forEach(d => {{
                if (d.depth > 1 && d.children) {{
                    d._children = d.children;
                    d.children = null;
                }}
            }});
            update();
        }}

        function exportSVG() {{
            const svgElement = document.querySelector("svg");
            const serializer = new XMLSerializer();
            const svgString = serializer.serializeToString(svgElement);
            const blob = new Blob([svgString], {{type: "image/svg+xml"}});
            const url = URL.createObjectURL(blob);
            const a = document.createElement("a");
            a.href = url;
            a.download = "{escaped_topic}_mindmap.svg";
            a.click();
            URL.revokeObjectURL(url);
        }}

        // Search functionality
        document.getElementById("searchInput").addEventListener("input", function(e) {{
            const query = e.target.value.toLowerCase();
            nodes.selectAll("circle")
                .attr("stroke-width", d => {{
                    if (query && d.data.label.toLowerCase().includes(query)) {{
                        return 4;
                    }}
                    return 2;
                }})
                .attr("stroke", d => {{
                    if (query && d.data.label.toLowerCase().includes(query)) {{
                        return "#FFD700";
                    }}
                    return "#fff";
                }});
        }});

        // Keyboard shortcuts
        document.addEventListener("keydown", function(e) {{
            if (e.key === "+" || e.key === "=") zoomIn();
            if (e.key === "-") zoomOut();
            if (e.key === "0") resetView();
        }});
    </script>
</body>
</html>'''

    async def estimate_pages(self, job: GenerationJob) -> int:
        """Mind maps are single-page HTML files."""
        return 1
