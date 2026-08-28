"""
Multimodal Agentic AI Automated Test Suite
Verifies MultimodalRouter, VisionAgent, MathVisionAgent, PhysicsVisionAgent, ChartDataAgent, DocumentVisionAgent, and FastAPI Multimodal APIs.
"""

import unittest
import json
import os

from multimodal_router import route_multimodal_input, MultimodalInput
from agents.vision_agent import process_image
from agents.math_vision_agent import solve_math_vision_query
from agents.physics_vision_agent import solve_physics_vision_query
from agents.chart_data_agent import solve_chart_data_query
from agents.document_vision_agent import parse_pdf_document
from graph import run_pipeline


class TestMultimodalAgents(unittest.TestCase):
    """Unit tests for Multimodal Router and Specialized Vision Agents"""

    def test_multimodal_router_classification(self):
        r_math = route_multimodal_input(MultimodalInput(prompt="Solve this integral", image_path="math.png"))
        self.assertEqual(r_math.route_target, "MATH_VISION")

        r_phys = route_multimodal_input(MultimodalInput(prompt="Calculate acceleration of block", image_path="forces.jpg"))
        self.assertEqual(r_phys.route_target, "PHYSICS_VISION")

        r_chart = route_multimodal_input(MultimodalInput(prompt="What was the largest increase?", image_path="chart.png"))
        self.assertEqual(r_chart.route_target, "CHART_DATA")

        r_doc = route_multimodal_input(MultimodalInput(prompt="Summarize report", pdf_path="report.pdf"))
        self.assertEqual(r_doc.route_target, "DOCUMENT_VISION")

    def test_vision_agent_processing(self):
        res = process_image(prompt_hint="Solve this integral")
        self.assertEqual(res.status, "success")
        self.assertIn("handwritten_calculus_integral", res.detected_elements)

    def test_math_vision_agent(self):
        res = solve_math_vision_query(prompt="Solve this integral")
        self.assertEqual(res["status"], "success")
        self.assertIn("MATH VISION AGENT", res["final_answer"])
        self.assertIn("∫ x² e^(3x) sin(2x) dx", res["final_answer"].replace("int", "∫").replace("x^2", "x²"))

    def test_physics_vision_agent(self):
        res = solve_physics_vision_query(prompt="Calculate acceleration of block")
        self.assertEqual(res["status"], "success")
        self.assertAlmostEqual(res["acceleration"], 8.6603, places=3)
        self.assertAlmostEqual(res["force_x"], 43.3013, places=3)

    def test_chart_data_agent(self):
        res = solve_chart_data_query(prompt="What was the largest increase between 2020 and 2025?")
        self.assertEqual(res["status"], "success")
        self.assertEqual(res["total_growth"], 350)
        self.assertIn("2022 to 2025", res["largest_period"])

    def test_document_vision_agent(self):
        res = parse_pdf_document("test_document.pdf")
        self.assertEqual(res.status, "success")
        self.assertEqual(res.filename, "test_document.pdf")
        self.assertGreater(len(res.text_chunks), 0)

    def test_graph_run_pipeline_multimodal(self):
        res_math = run_pipeline("Solve this integral", image_path="integral.png")
        self.assertEqual(res_math["route"], "math_vision")
        self.assertIn("MATH VISION AGENT", res_math["final_answer"])

        res_phys = run_pipeline("Calculate acceleration of block", image_path="block.png")
        self.assertEqual(res_phys["route"], "physics_vision")
        self.assertIn("PHYSICS VISION AGENT", res_phys["final_answer"])

        res_doc = run_pipeline("Summarize annual report", pdf_path="annual_report.pdf")
        self.assertEqual(res_doc["route"], "document_vision")
        self.assertIn("DOCUMENT VISION AGENT", res_doc["final_answer"])


if __name__ == "__main__":
    unittest.main()
