"""
Model Context Protocol (MCP) Web RAG Module for Coding Platforms
Supports live Web RAG targeting GeeksforGeeks, LeetCode, Codeforces, and GitHub.
"""

from typing import Any, Dict, List, Optional
import json
import re
from agents.search_tool import google_search, _duckduckgo_search, fetch_page_text, extract_chunks_from_page
from answerability_agent import check_answerability
from reranker import funnel_phase1, funnel_phase2


# ---------------------------------------------------------------------------
# MCP Tool Definitions Schema
# ---------------------------------------------------------------------------

MCP_TOOLS_MANIFEST = [
    {
        "name": "search_leetcode_solution",
        "description": "Fetch LeetCode problem solutions, explanations, and Python/C++ code via Web RAG.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "problem": {"type": "string", "description": "LeetCode problem number or title (e.g. '3', 'Two Sum', 'Longest Substring')"},
                "language": {"type": "string", "default": "python", "description": "Programming language"}
            },
            "required": ["problem"]
        }
    },
    {
        "name": "search_geeksforgeeks_solution",
        "description": "Fetch GeeksforGeeks data structures and algorithm solutions via Web RAG.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "topic": {"type": "string", "description": "GeeksforGeeks article topic or problem title"},
                "language": {"type": "string", "default": "python", "description": "Programming language"}
            },
            "required": ["topic"]
        }
    },
    {
        "name": "search_codeforces_solution",
        "description": "Fetch Codeforces competitive programming solutions via Web RAG.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "problem": {"type": "string", "description": "Codeforces problem ID or title (e.g. '4A Watermelon')"},
                "language": {"type": "string", "default": "cpp", "description": "Programming language"}
            },
            "required": ["problem"]
        }
    },
    {
        "name": "mcp_web_rag_coding_search",
        "description": "Unified MCP Web RAG search targeting GeeksforGeeks, LeetCode, Codeforces, and GitHub.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Coding question or problem title"},
                "platform": {"type": "string", "enum": ["leetcode", "geeksforgeeks", "codeforces", "all"], "default": "all"}
            },
            "required": ["query"]
        }
    }
]


# ---------------------------------------------------------------------------
# MCP Tool Implementations
# ---------------------------------------------------------------------------

def mcp_search_leetcode(problem: str, language: str = "python") -> Dict[str, Any]:
    """MCP Tool: Search LeetCode solution."""
    search_q = f"site:leetcode.com {problem} {language} solution"
    results = google_search(search_q)
    if not results:
        results = _duckduckgo_search(f"leetcode {problem} {language} solution", num_results=5)

    return _process_mcp_results(f"LeetCode: {problem}", results, language)


def mcp_search_geeksforgeeks(topic: str, language: str = "python") -> Dict[str, Any]:
    """MCP Tool: Search GeeksforGeeks solution."""
    search_q = f"site:geeksforgeeks.org {topic} {language} solution"
    results = google_search(search_q)
    if not results:
        results = _duckduckgo_search(f"geeksforgeeks {topic} {language} solution", num_results=5)

    return _process_mcp_results(f"GeeksforGeeks: {topic}", results, language)


def mcp_search_codeforces(problem: str, language: str = "cpp") -> Dict[str, Any]:
    """MCP Tool: Search Codeforces solution."""
    search_q = f"site:codeforces.com {problem} {language} solution"
    results = google_search(search_q)
    if not results:
        results = _duckduckgo_search(f"codeforces {problem} {language} solution", num_results=5)

    return _process_mcp_results(f"Codeforces: {problem}", results, language)


def mcp_web_rag_coding_search(query: str, platform: str = "all") -> Dict[str, Any]:
    """Unified MCP Web RAG search across GeeksforGeeks, LeetCode, and Codeforces."""
    if platform == "leetcode":
        site_filter = "site:leetcode.com"
    elif platform == "geeksforgeeks":
        site_filter = "site:geeksforgeeks.org"
    elif platform == "codeforces":
        site_filter = "site:codeforces.com"
    else:
        site_filter = "site:leetcode.com OR site:geeksforgeeks.org OR site:codeforces.com"

    search_q = f"{query} {site_filter}"
    results = google_search(search_q)
    if not results:
        results = _duckduckgo_search(f"{query} geeksforgeeks leetcode codeforces solution", num_results=8)

    return _process_mcp_results(query, results, "python")


# ---------------------------------------------------------------------------
# Internal MCP Web RAG Content Processor
# ---------------------------------------------------------------------------

def _process_mcp_results(target_topic: str, results: list, language: str) -> Dict[str, Any]:
    """Process raw search results into an MCP Web RAG response object."""
    if not results:
        return {
            "status": "error",
            "message": f"No web search results found for {target_topic}.",
            "target": target_topic,
            "solution": ""
        }

    chunks = []
    sources = []

    for r in results[:5]:
        title = r.title.strip() if hasattr(r, 'title') else r.get('title', '')
        link = r.link.strip() if hasattr(r, 'link') else r.get('link', '')
        snippet = r.snippet.strip() if hasattr(r, 'snippet') else r.get('snippet', '')

        if snippet:
            chunks.append(f"[{title}] ({link}): {snippet}")
            sources.append(link)

    context_str = "\n".join(chunks)

    from llm_client import _format_coding_solution
    solution_code = _format_coding_solution(target_topic, context_str)

    return {
        "status": "success",
        "target": target_topic,
        "sources": sources[:5],
        "chunks_retrieved": len(chunks),
        "solution": solution_code,
        "context_summary": context_str[:600]
    }


# ---------------------------------------------------------------------------
# MCP Server Request Handler (JSON-RPC 2.0 / Standard Protocol)
# ---------------------------------------------------------------------------

def handle_mcp_request(raw_json: str) -> str:
    """Handle incoming MCP protocol requests (tools/list or tools/call)."""
    try:
        req = json.loads(raw_json)
        method = req.get("method", "")
        req_id = req.get("id", 1)

        if method == "tools/list":
            return json.dumps({
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {"tools": MCP_TOOLS_MANIFEST}
            }, indent=2)

        elif method == "tools/call":
            params = req.get("params", {})
            name = params.get("name", "")
            args = params.get("arguments", {})

            if name == "search_leetcode_solution":
                res = mcp_search_leetcode(args.get("problem", ""), args.get("language", "python"))
            elif name == "search_geeksforgeeks_solution":
                res = mcp_search_geeksforgeeks(args.get("topic", ""), args.get("language", "python"))
            elif name == "search_codeforces_solution":
                res = mcp_search_codeforces(args.get("problem", ""), args.get("language", "cpp"))
            elif name == "mcp_web_rag_coding_search":
                res = mcp_web_rag_coding_search(args.get("query", ""), args.get("platform", "all"))
            else:
                return json.dumps({
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "error": {"code": -32601, "message": f"Unknown MCP tool: {name}"}
                })

            return json.dumps({
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {"content": [{"type": "text", "text": json.dumps(res, indent=2)}]}
            }, indent=2)

        else:
            return json.dumps({
                "jsonrpc": "2.0",
                "id": req_id,
                "error": {"code": -32601, "message": f"Method non-existent: {method}"}
            })

    except Exception as e:
        return json.dumps({
            "jsonrpc": "2.0",
            "id": 1,
            "error": {"code": -32603, "message": str(e)}
        })


if __name__ == "__main__":
    # Self-test MCP tools
    print("=== Testing MCP Web RAG Coding Tools ===")
    test_req = json.dumps({
        "jsonrpc": "2.0",
        "id": 1,
        "method": "tools/call",
        "params": {
            "name": "search_leetcode_solution",
            "arguments": {"problem": "Longest Substring Without Repeating Characters", "language": "python"}
        }
    })
    resp = handle_mcp_request(test_req)
    print(resp)
