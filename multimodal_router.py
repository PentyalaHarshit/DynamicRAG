"""
Multimodal Router — Input Modal & Domain Classification Engine
Classifies incoming user inputs (Text, Image Base64/Bytes/Path, PDF Document)
and dispatches them to specialized Multimodal Agents:
- MATH_VISION: Image of math equation / calculus integral
- PHYSICS_VISION: Image of physics diagram / forces / acceleration
- CHART_DATA: Image of chart / graph / data table
- DOCUMENT_VISION: Multi-modal PDF document (text + images + tables)
- GENERAL_VISION: Image of real-world object / scene
- TEXT_ONLY: Standard text prompt
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import os
import re


@dataclass
class MultimodalInput:
    prompt: str = ""
    image_path: Optional[str] = None
    image_base64: Optional[str] = None
    pdf_path: Optional[str] = None


@dataclass
class RouteDecision:
    input_modal: str        # "TEXT_ONLY", "IMAGE", "PDF_DOCUMENT"
    route_target: str       # "MATH_VISION", "PHYSICS_VISION", "CHART_DATA", "DOCUMENT_VISION", "GENERAL_VISION", "TEXT_PIPELINE"
    confidence: float
    metadata: Dict[str, Any]


def route_multimodal_input(input_data: MultimodalInput) -> RouteDecision:
    """Evaluates input modal type and prompt keywords to determine agent workflow."""
    prompt_lower = (input_data.prompt or "").lower()

    # 1. PDF Document Stream
    if input_data.pdf_path or prompt_lower.endswith(".pdf"):
        return RouteDecision(
            input_modal="PDF_DOCUMENT",
            route_target="DOCUMENT_VISION",
            confidence=0.98,
            metadata={"pdf_path": input_data.pdf_path}
        )

    # 2. Image Stream (Path or Base64)
    if input_data.image_path or input_data.image_base64:
        # Check domain keywords in prompt or image filename
        path_str = (input_data.image_path or "").lower()

        if any(w in prompt_lower or w in path_str for w in ("integral", "derivative", "equation", "calculus", "math", "solve", "latex", "d/dx")):
            return RouteDecision(
                input_modal="IMAGE",
                route_target="MATH_VISION",
                confidence=0.95,
                metadata={"domain": "mathematics"}
            )

        if any(w in prompt_lower or w in path_str for w in ("physics", "force", "diagram", "mass", "velocity", "acceleration", "block", "angle", "relativity", "gravity")):
            return RouteDecision(
                input_modal="IMAGE",
                route_target="PHYSICS_VISION",
                confidence=0.95,
                metadata={"domain": "physics"}
            )

        if any(w in prompt_lower or w in path_str for w in ("chart", "graph", "plot", "table", "increase", "bar", "pie", "histogram")):
            return RouteDecision(
                input_modal="IMAGE",
                route_target="CHART_DATA",
                confidence=0.90,
                metadata={"domain": "data_chart"}
            )

        return RouteDecision(
            input_modal="IMAGE",
            route_target="GENERAL_VISION",
            confidence=0.85,
            metadata={"domain": "general_vision"}
        )

    # 3. Default Text Stream
    return RouteDecision(
        input_modal="TEXT_ONLY",
        route_target="TEXT_PIPELINE",
        confidence=1.0,
        metadata={"domain": "text"}
    )


if __name__ == "__main__":
    r1 = route_multimodal_input(MultimodalInput(prompt="Solve this integral", image_path="integral.png"))
    r2 = route_multimodal_input(MultimodalInput(prompt="Calculate the acceleration of the block", image_path="diagram.jpg"))
    r3 = route_multimodal_input(MultimodalInput(prompt="Extract data from report", pdf_path="annual_report.pdf"))
    print("Test 1:", r1)
    print("Test 2:", r2)
    print("Test 3:", r3)
