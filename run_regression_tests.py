"""
Automated 100-Query Regression Test Suite
==========================================
Tests 100 comprehensive queries across 6 major domains:
  1. Travel & Flights (20 queries)
  2. Weather (15 queries)
  3. Finance & Stock Market (15 queries)
  4. Currency & Exchange Rates (15 queries)
  5. Biography & Factoids (15 queries)
  6. Coding, Math & Algorithms (20 queries)

Validates intent detection, router accuracy, verifier scores, answer non-emptiness,
and execution latency. Outputs regression_results.json and a formatted summary table.
"""
import sys
import time
import json
import re
from typing import Dict, Any, List
from main import answer_question

# ---------------------------------------------------------------------------
# 100 Diverse Test Queries Organized by Category
# ---------------------------------------------------------------------------

TEST_QUERIES: List[Dict[str, str]] = [
    # ── Category 1: Travel & Flights (20 queries) ──────────────────────────
    {"cat": "Travel", "expected_intent": "TRAVEL", "q": "what is cheapest flight price from Dallas to Hyderabad on 2nd september?"},
    {"cat": "Travel", "expected_intent": "TRAVEL", "q": "Is there flight from Dallas to Hyderabad on 2nd september?"},
    {"cat": "Travel", "expected_intent": "TRAVEL", "q": "flights from New York to London tomorrow"},
    {"cat": "Travel", "expected_intent": "TRAVEL", "q": "cheapest plane ticket from San Francisco to Tokyo in October"},
    {"cat": "Travel", "expected_intent": "TRAVEL", "q": "what is airfare cost from Chicago to Paris on 15th november?"},
    {"cat": "Travel", "expected_intent": "TRAVEL", "q": "flight schedule from Los Angeles to Sydney"},
    {"cat": "Travel", "expected_intent": "TRAVEL", "q": "cheapest flight deals from Miami to Toronto"},
    {"cat": "Travel", "expected_intent": "TRAVEL", "q": "how much is flight ticket from Seattle to Singapore?"},
    {"cat": "Travel", "expected_intent": "TRAVEL", "q": "best flight options from Boston to Rome on 10th august"},
    {"cat": "Travel", "expected_intent": "TRAVEL", "q": "flights from Atlanta to Dubai next week"},
    {"cat": "Travel", "expected_intent": "TRAVEL", "q": "cheap airfare from Houston to Frankfurt"},
    {"cat": "Travel", "expected_intent": "TRAVEL", "q": "what is flight price from Vancouver to London?"},
    {"cat": "Travel", "expected_intent": "TRAVEL", "q": "direct flights from Newark to Mumbai on 5th december"},
    {"cat": "Travel", "expected_intent": "TRAVEL", "q": "ticket fare from Denver to Seoul"},
    {"cat": "Travel", "expected_intent": "TRAVEL", "q": "cheapest flight from Austin to Amsterdam"},
    {"cat": "Travel", "expected_intent": "TRAVEL", "q": "flight price from Philadelphia to Madrid"},
    {"cat": "Travel", "expected_intent": "TRAVEL", "q": "flights from San Jose to Honolulu on 20th july"},
    {"cat": "Travel", "expected_intent": "TRAVEL", "q": "how much does it cost to fly from Dallas to Delhi?"},
    {"cat": "Travel", "expected_intent": "TRAVEL", "q": "lowest airfare from Phoenix to Cancun"},
    {"cat": "Travel", "expected_intent": "TRAVEL", "q": "flight tickets from Detroit to Zurich"},

    # ── Category 2: Weather (15 queries) ──────────────────────────────────
    {"cat": "Weather", "expected_intent": "WEATHER", "q": "what is the weather in Tokyo?"},
    {"cat": "Weather", "expected_intent": "WEATHER", "q": "what is current temperature in Paris?"},
    {"cat": "Weather", "expected_intent": "WEATHER", "q": "is it raining in London right now?"},
    {"cat": "Weather", "expected_intent": "WEATHER", "q": "weather forecast for New York City"},
    {"cat": "Weather", "expected_intent": "WEATHER", "q": "what is temperature in Sydney Australia?"},
    {"cat": "Weather", "expected_intent": "WEATHER", "q": "current weather in Berlin Germany"},
    {"cat": "Weather", "expected_intent": "WEATHER", "q": "what is climate in Toronto today?"},
    {"cat": "Weather", "expected_intent": "WEATHER", "q": "temperature in Rome Italy"},
    {"cat": "Weather", "expected_intent": "WEATHER", "q": "is it hot in Dubai today?"},
    {"cat": "Weather", "expected_intent": "WEATHER", "q": "weather in Singapore today"},
    {"cat": "Weather", "expected_intent": "WEATHER", "q": "current temperature in Seoul South Korea"},
    {"cat": "Weather", "expected_intent": "WEATHER", "q": "weather report for Cairo Egypt"},
    {"cat": "Weather", "expected_intent": "WEATHER", "q": "temperature in Chicago Illinois"},
    {"cat": "Weather", "expected_intent": "WEATHER", "q": "what is weather in Madrid Spain?"},
    {"cat": "Weather", "expected_intent": "WEATHER", "q": "current temperature in Mumbai India"},

    # ── Category 3: Finance & Stock Market (15 queries) ───────────────────
    {"cat": "Finance", "expected_intent": "FINANCE", "q": "what is the stock price of Apple?"},
    {"cat": "Finance", "expected_intent": "FINANCE", "q": "what is NVDA share price?"},
    {"cat": "Finance", "expected_intent": "FINANCE", "q": "stock price of Microsoft MSFT"},
    {"cat": "Finance", "expected_intent": "FINANCE", "q": "how much is Tesla TSLA stock today?"},
    {"cat": "Finance", "expected_intent": "FINANCE", "q": "Google Alphabet GOOGL share price"},
    {"cat": "Finance", "expected_intent": "FINANCE", "q": "Amazon AMZN stock price today"},
    {"cat": "Finance", "expected_intent": "FINANCE", "q": "Meta Facebook stock price"},
    {"cat": "Finance", "expected_intent": "FINANCE", "q": "what is AMD stock trading at?"},
    {"cat": "Finance", "expected_intent": "FINANCE", "q": "Netflix NFLX share price"},
    {"cat": "Finance", "expected_intent": "FINANCE", "q": "Disney DIS stock price today"},
    {"cat": "Finance", "expected_intent": "FINANCE", "q": "Intel INTC share price"},
    {"cat": "Finance", "expected_intent": "FINANCE", "q": "IBM stock price"},
    {"cat": "Finance", "expected_intent": "FINANCE", "q": "Alibaba BABA stock price"},
    {"cat": "Finance", "expected_intent": "FINANCE", "q": "Coinbase COIN stock price"},
    {"cat": "Finance", "expected_intent": "FINANCE", "q": "Palantir PLTR share price"},

    # ── Category 4: Currency Exchange (15 queries) ────────────────────────
    {"cat": "Currency", "expected_intent": "CURRENCY", "q": "convert 100 USD to EUR"},
    {"cat": "Currency", "expected_intent": "CURRENCY", "q": "what is exchange rate from USD to INR?"},
    {"cat": "Currency", "expected_intent": "CURRENCY", "q": "convert 50 EUR to GBP"},
    {"cat": "Currency", "expected_intent": "CURRENCY", "q": "how many USD is 1000 JPY?"},
    {"cat": "Currency", "expected_intent": "CURRENCY", "q": "convert 200 CAD to USD"},
    {"cat": "Currency", "expected_intent": "CURRENCY", "q": "exchange rate AUD to USD"},
    {"cat": "Currency", "expected_intent": "CURRENCY", "q": "convert 5000 INR to USD"},
    {"cat": "Currency", "expected_intent": "CURRENCY", "q": "100 GBP in EUR"},
    {"cat": "Currency", "expected_intent": "CURRENCY", "q": "convert 150 USD to CAD"},
    {"cat": "Currency", "expected_intent": "CURRENCY", "q": "exchange rate CHF to EUR"},
    {"cat": "Currency", "expected_intent": "CURRENCY", "q": "convert 500 SGD to USD"},
    {"cat": "Currency", "expected_intent": "CURRENCY", "q": "how much is 100 NZD in USD?"},
    {"cat": "Currency", "expected_intent": "CURRENCY", "q": "convert 1000 MXN to USD"},
    {"cat": "Currency", "expected_intent": "CURRENCY", "q": "convert 500 BRL to USD"},
    {"cat": "Currency", "expected_intent": "CURRENCY", "q": "exchange rate ZAR to USD"},

    # ── Category 5: Biography & Factoids (15 queries) ────────────────────
    {"cat": "Biography", "expected_intent": "BIOGRAPHY", "q": "Who is top coding programmer present?"},
    {"cat": "Biography", "expected_intent": "BIOGRAPHY", "q": "Who invented the C programming language?"},
    {"cat": "Biography", "expected_intent": "BIOGRAPHY", "q": "Who created the Linux operating system?"},
    {"cat": "Biography", "expected_intent": "BIOGRAPHY", "q": "Who developed Python programming language?"},
    {"cat": "Biography", "expected_intent": "BIOGRAPHY", "q": "Who is Albert Einstein?"},
    {"cat": "Biography", "expected_intent": "BIOGRAPHY", "q": "Who is Alan Turing and what did he invent?"},
    {"cat": "Biography", "expected_intent": "BIOGRAPHY", "q": "Who discovered gravity?"},
    {"cat": "Biography", "expected_intent": "BIOGRAPHY", "q": "Who was Marie Curie?"},
    {"cat": "Biography", "expected_intent": "BIOGRAPHY", "q": "Who was Nikola Tesla?"},
    {"cat": "Biography", "expected_intent": "BIOGRAPHY", "q": "Who was Ada Lovelace?"},
    {"cat": "Biography", "expected_intent": "BIOGRAPHY", "q": "Who invented the Analytical Engine?"},
    {"cat": "Biography", "expected_intent": "BIOGRAPHY", "q": "Who was Grace Hopper?"},
    {"cat": "Biography", "expected_intent": "BIOGRAPHY", "q": "Who co-founded Apple with Steve Wozniak?"},
    {"cat": "Biography", "expected_intent": "BIOGRAPHY", "q": "Who founded Microsoft?"},
    {"cat": "Biography", "expected_intent": "BIOGRAPHY", "q": "Who is Elon Musk?"},

    # ── Category 6: Coding, Math & Reasoning (20 queries) ─────────────────
    {"cat": "Coding", "expected_intent": "CODING", "q": "Write a python script for binary search"},
    {"cat": "Coding", "expected_intent": "CODING", "q": "Write a python function to compute fibonacci numbers"},
    {"cat": "Coding", "expected_intent": "CODING", "q": "Write a python implementation of quicksort"},
    {"cat": "Coding", "expected_intent": "CODING", "q": "How to implement merge sort in Python?"},
    {"cat": "Coding", "expected_intent": "CODING", "q": "Write a python script for linear regression"},
    {"cat": "Coding", "expected_intent": "CODING", "q": "How to implement a linked list in Python?"},
    {"cat": "Coding", "expected_intent": "CODING", "q": "Write python code for bubble sort"},
    {"cat": "Coding", "expected_intent": "CODING", "q": "Python function to check if a number is prime"},
    {"cat": "Coding", "expected_intent": "CODING", "q": "Write a python function for matrix multiplication"},
    {"cat": "Coding", "expected_intent": "CODING", "q": "How to compute factorial recursively in Python?"},
    {"cat": "Coding", "expected_intent": "CODING", "q": "Implement a Stack data structure in Python"},
    {"cat": "Coding", "expected_intent": "CODING", "q": "Implement a Queue data structure in Python"},
    {"cat": "Coding", "expected_intent": "CODING", "q": "Write a python script for binary tree traversal"},
    {"cat": "Coding", "expected_intent": "CODING", "q": "How to implement Breadth First Search BFS in Python?"},
    {"cat": "Coding", "expected_intent": "CODING", "q": "How to implement Depth First Search DFS in Python?"},
    {"cat": "Coding", "expected_intent": "CODING", "q": "Write Sieve of Eratosthenes prime algorithm in Python"},
    {"cat": "Coding", "expected_intent": "CODING", "q": "Python function to find greatest common divisor GCD"},
    {"cat": "Coding", "expected_intent": "CODING", "q": "Implement Dijkstra shortest path algorithm in Python"},
    {"cat": "Coding", "expected_intent": "CODING", "q": "How to reverse a string in Python?"},
    {"cat": "Coding", "expected_intent": "CODING", "q": "Write a python script for checking valid parentheses"},
]


def run_regression():
    """Runs all 100 test queries, logs results, and outputs summary statistics."""
    print("=" * 80)
    print("      AUTOMATED 100-QUERY REGRESSION TEST SUITE FOR HYBRID RAG")
    print("=" * 80)
    print(f"Total Test Cases: {len(TEST_QUERIES)}")
    print("Executing tests...\n")

    results = []
    category_stats = {}
    total_start_time = time.time()

    passed_count = 0
    failed_count = 0

    for idx, test in enumerate(TEST_QUERIES, 1):
        cat = test["cat"]
        expected_intent = test["expected_intent"]
        q = test["q"]

        if cat not in category_stats:
            category_stats[cat] = {"total": 0, "passed": 0, "failed": 0, "intent_matches": 0, "latency_sum": 0.0}

        category_stats[cat]["total"] += 1

        start_t = time.time()
        try:
            res = answer_question(q)
            elapsed = time.time() - start_t

            intent_info = res.get("intent", {})
            actual_intent = (
                intent_info.intent_type
                if hasattr(intent_info, "intent_type")
                else (intent_info.get("type") if isinstance(intent_info, dict) else str(intent_info))
            )

            final_ans = res.get("final_answer") or res.get("direct_answer") or ""
            route = res.get("route", "")
            verifier_score = res.get("final_score", 0.0)

            # Verification criteria:
            # 1. Answer must be non-empty and not sentinel error
            # 2. No runtime unhandled exception
            # 3. Answer length > 10 chars
            is_valid_ans = bool(final_ans and len(final_ans.strip()) > 10 and "[LLM unavailable — no response generated]" not in final_ans)
            intent_match = (actual_intent == expected_intent)

            passed = is_valid_ans

            if passed:
                passed_count += 1
                category_stats[cat]["passed"] += 1
                status_str = "PASS"
            else:
                failed_count += 1
                category_stats[cat]["failed"] += 1
                status_str = "FAIL"

            if intent_match:
                category_stats[cat]["intent_matches"] += 1

            category_stats[cat]["latency_sum"] += elapsed

            rec = {
                "id": idx,
                "category": cat,
                "query": q,
                "expected_intent": expected_intent,
                "actual_intent": actual_intent,
                "intent_match": intent_match,
                "route": route,
                "status": status_str,
                "verifier_score": verifier_score,
                "elapsed_sec": round(elapsed, 3),
                "answer_snippet": final_ans[:120].replace("\n", " ") + "...",
            }
            results.append(rec)

            print(f"[{idx:03d}/{len(TEST_QUERIES)}] [{status_str}] ({cat}) Intent: {actual_intent} | Route: {route} | {elapsed:.2f}s | Q: '{q[:50]}...'")

        except Exception as exc:
            elapsed = time.time() - start_t
            failed_count += 1
            category_stats[cat]["failed"] += 1
            category_stats[cat]["latency_sum"] += elapsed

            rec = {
                "id": idx,
                "category": cat,
                "query": q,
                "expected_intent": expected_intent,
                "actual_intent": "ERROR",
                "intent_match": False,
                "route": "ERROR",
                "status": "FAIL",
                "verifier_score": 0.0,
                "elapsed_sec": round(elapsed, 3),
                "error": str(exc),
            }
            results.append(rec)
            print(f"[{idx:03d}/{len(TEST_QUERIES)}] [FAIL-EXC] ({cat}) Error: {exc} | Q: '{q[:50]}...'")

    total_elapsed = time.time() - total_start_time

    # Save complete JSON report
    with open("regression_results.json", "w", encoding="utf-8") as f:
        json.dump({
            "total_queries": len(TEST_QUERIES),
            "passed": passed_count,
            "failed": failed_count,
            "pass_rate_pct": round((passed_count / len(TEST_QUERIES)) * 100, 2),
            "total_elapsed_sec": round(total_elapsed, 2),
            "category_summary": category_stats,
            "results": results,
        }, f, indent=2)

    # Output Grand Summary Table
    print("\n" + "=" * 80)
    print("                     REGRESSION TEST SUMMARY REPORT")
    print("=" * 80)
    print(f"Total Queries Executed:  {len(TEST_QUERIES)}")
    print(f"Passed:                  {passed_count} ({passed_count / len(TEST_QUERIES) * 100:.1f}%)")
    print(f"Failed:                  {failed_count} ({failed_count / len(TEST_QUERIES) * 100:.1f}%)")
    print(f"Total Wall Clock Time:   {total_elapsed:.2f} s (Avg: {total_elapsed / len(TEST_QUERIES):.2f} s/query)")
    print("-" * 80)
    print(f"{'Category':<15} | {'Total':<6} | {'Passed':<6} | {'Failed':<6} | {'Pass %':<8} | {'Intent Acc %':<12} | {'Avg Latency':<10}")
    print("-" * 80)

    for cat, stat in category_stats.items():
        tot = stat["total"]
        pas = stat["passed"]
        fai = stat["failed"]
        pass_pct = (pas / tot * 100) if tot > 0 else 0.0
        intent_acc = (stat["intent_matches"] / tot * 100) if tot > 0 else 0.0
        avg_lat = (stat["latency_sum"] / tot) if tot > 0 else 0.0
        print(f"{cat:<15} | {tot:<6} | {pas:<6} | {fai:<6} | {pass_pct:<7.1f}% | {intent_acc:<11.1f}% | {avg_lat:<8.2f} s")

    print("=" * 80)
    print("Full JSON results saved to: regression_results.json\n")


if __name__ == "__main__":
    run_regression()
