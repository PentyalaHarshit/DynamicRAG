"""
Comprehensive Automated Unit Testing & Component Troubleshooting Suite for Hybrid RAG
"""

import unittest

# Import system modules
from intent_detector import detect_intent
from verifier import verify_answer
from answerability_agent import (
    check_answerability_full,
    _extract_persons,
    _extract_dates,
    _extract_durations,
    _extract_numbers,
    _META_WORDS,
)
from query_expander import _heuristic_expansions
from travel_api import parse_travel_query, get_travel_info
from weather_api import get_weather
from finance_api import get_stock_quote
from llm_client import _fallback_synthesis, call_llm


class TestIntentDetector(unittest.TestCase):
    """Unit tests for Intent Detection Module"""

    def test_travel_intent(self):
        result = detect_intent("flights from NYC to London under $500")
        self.assertEqual(result.intent_type, "TRAVEL")

    def test_weather_intent(self):
        result = detect_intent("what is the weather in Tokyo tomorrow?")
        self.assertEqual(result.intent_type, "WEATHER")

    def test_finance_intent(self):
        result = detect_intent("what is the stock price of AAPL?")
        self.assertEqual(result.intent_type, "FINANCE")

    def test_currency_intent(self):
        result = detect_intent("convert 100 USD to EUR")
        self.assertEqual(result.intent_type, "CURRENCY")

    def test_biography_intent(self):
        result = detect_intent("who is Dennis Ritchie?")
        self.assertEqual(result.intent_type, "BIOGRAPHY")

    def test_coding_intent(self):
        result = detect_intent("Write a python function to implement quicksort")
        self.assertEqual(result.intent_type, "CODING")

    def test_math_intent(self):
        result = detect_intent("calculate integral of x^2 dx")
        self.assertEqual(result.intent_type, "MATH")


class TestVerifierAgent(unittest.TestCase):
    """Unit tests for 4D Verification Test Agent"""

    def test_verifier_score_calculation(self):
        res = verify_answer(
            question="Who is the creator of C?",
            answer="Dennis Ritchie created C programming language.",
            context="C was created by Dennis Ritchie in 1972.",
        )
        self.assertGreaterEqual(res.score, 0.4)

    def test_verifier_hallucination_detection(self):
        res = verify_answer(
            question="Who created Python?",
            answer="Albert Einstein created Python language.",
            context="Python was created by Guido van Rossum.",
        )
        self.assertIsNotNone(res)


class TestFallbackSynthesisLengthControl(unittest.TestCase):
    """Unit tests for LLM Fallback Synthesis and Length Modifier Control"""

    def test_rest_vs_graphql_big_explanation(self):
        ans = _fallback_synthesis("Explain the difference between REST API and GraphQL big explanation", "")
        self.assertIn("Comprehensive Architectural Deep-Dive", ans)
        self.assertIn("Network Data Fetching Efficiency", ans)
        self.assertGreater(len(ans), 800)

    def test_rest_vs_graphql_small_explanation(self):
        ans = _fallback_synthesis("REST vs GraphQL short summary small explanation", "")
        self.assertIn("REST API vs GraphQL (Short Summary)", ans)
        self.assertLess(len(ans), 500)

    def test_coding_algorithm_fallbacks(self):
        algorithms = [
            "quicksort algorithm in python",
            "merge sort python implementation",
            "binary search algorithm in python",
            "reverse linked list python",
            "bubble sort python code",
            "sieve of eratosthenes prime",
            "dijkstra shortest path python",
        ]
        for query in algorithms:
            with self.subTest(query=query):
                ans = _fallback_synthesis(query, "")
                self.assertIn("Detailed Overview", ans)


class TestAnswerabilityAgent(unittest.TestCase):
    """Unit tests for Answerability Agent Entity Extraction and Blocklist"""

    def test_organization_blocklist(self):
        text = "American National Standards Institute standardized C."
        persons = _extract_persons(text)
        self.assertNotIn("American National Standards Institute", persons)

    def test_person_extraction(self):
        text = "Dennis Ritchie and Brian Kernighan co-wrote the C book."
        persons = _extract_persons(text)
        self.assertIn("Dennis Ritchie", persons)
        self.assertIn("Brian Kernighan", persons)

    def test_date_extraction(self):
        text = "The dynasty ended in 1279 CE after ruling since 300 BCE."
        dates = _extract_dates(text)
        self.assertIn("1279", dates)


class TestStructuredAPIs(unittest.TestCase):
    """Unit tests for Dedicated API Nodes"""

    def test_travel_api_parsing(self):
        params = parse_travel_query("flights from NYC to Paris on 2nd September")
        self.assertIn("nyc", params["origin"].lower())
        self.assertIn("paris", params["destination"].lower())

    def test_weather_api_call(self):
        w = get_weather("London")
        self.assertIn("temperature_c", w)
        self.assertIn("condition", w)

    def test_finance_api_call(self):
        s = get_stock_quote("AAPL")
        self.assertEqual(s["ticker"], "AAPL")
        self.assertIn("price", s)

    def test_currency_conversion_api(self):
        from graph import run_pipeline
        res = run_pipeline("convert 100 USD to EUR")
        self.assertEqual(res["intent"].intent_type, "CURRENCY")
        self.assertIn("100", res["final_answer"])
        self.assertIn("EUR", res["final_answer"])


class TestMCPCodingRAG(unittest.TestCase):
    """Unit tests for MCP Web RAG Coding Tools Module"""

    def test_mcp_manifest(self):
        from mcp_coding_rag import handle_mcp_request
        import json
        req = json.dumps({"jsonrpc": "2.0", "id": 1, "method": "tools/list"})
        resp = json.loads(handle_mcp_request(req))
        self.assertIn("result", resp)
        self.assertEqual(len(resp["result"]["tools"]), 4)

    def test_mcp_web_rag_search(self):
        from mcp_coding_rag import mcp_web_rag_coding_search
        res = mcp_web_rag_coding_search("LeetCode Two Sum python solution")
        self.assertEqual(res["status"], "success")
        self.assertGreater(res["chunks_retrieved"], 0)


class TestAnswerTypeAndReasoningAgents(unittest.TestCase):
    """Unit tests for Answer-Type Detector, Math Agent, and Physics Agent"""

    def test_answer_type_detection(self):
        from answer_type_agent import detect_answer_type
        r1 = detect_answer_type("Is Asim Munir a Field Marshal?")
        self.assertEqual(r1.answer_type, "YES_NO")

        r2 = detect_answer_type("What is the second derivative of e^(x^2)sin(3x^2+1)?")
        self.assertEqual(r2.answer_type, "CALCULATION")

        r3 = detect_answer_type("An electron is accelerated through 2 MV. Calculate its relativistic velocity.")
        self.assertEqual(r3.answer_type, "CALCULATION")

    def test_math_agent_symbolic_derivation(self):
        from agents.math_agent import solve_math_query
        res = solve_math_query("What is the second derivative of e^(x^2)sin(3x^2+1)?")
        self.assertEqual(res["status"], "success")
        self.assertIn("Symbolic Solver", res["agent"])
        self.assertIn("f''(x)", res["final_answer"])

    def test_physics_agent_relativity(self):
        from agents.physics_agent import solve_physics_query
        res = solve_physics_query("An electron is accelerated through 2 MV. Calculate its relativistic velocity.")
        self.assertEqual(res["status"], "success")
        self.assertAlmostEqual(res["lorentz_gamma"], 4.9139, places=3)
        self.assertAlmostEqual(res["beta"], 0.9791, places=3)

    def test_yes_no_formatting(self):
        from answer_type_agent import format_yes_no_response
        ans = format_yes_no_response("Is Asim Munir a Field Marshal?", "Gen Asim Munir was promoted to Field Marshal in 2025.")
        self.assertTrue(ans.startswith("Yes."))


if __name__ == "__main__":
    unittest.main()
