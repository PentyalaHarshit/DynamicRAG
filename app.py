"""
FastAPI Backend Web Application Server for Dynamic Hybrid RAG Pipeline
Exposes REST APIs for query execution, MCP JSON-RPC tool testing, and pipeline telemetry.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Any, Dict, Optional
import json
import uvicorn

from graph import run_pipeline
from mcp_coding_rag import handle_mcp_request, MCP_TOOLS_MANIFEST


app = FastAPI(
    title="Dynamic Hybrid RAG API",
    description="Hierarchical Multi-Stage Hybrid RAG & SAC Reinforcement Learning Server",
    version="2.0.0"
)

# Enable CORS for React frontend development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class QueryRequest(BaseModel):
    query: str


class MCPRequest(BaseModel):
    raw_json: Optional[str] = None
    tool_name: Optional[str] = None
    arguments: Optional[Dict[str, Any]] = None


@app.get("/api/health")
def health_check():
    """Returns system health and active model configuration."""
    return {
        "status": "online",
        "service": "Dynamic Hybrid RAG Engine",
        "version": "2.0.0",
        "components": {
            "intent_detector": "active",
            "mcp_coding_rag": "active",
            "verifier_agent": "active",
            "sac_rl_policy": "active"
        }
    }


@app.post("/api/query")
def execute_query(req: QueryRequest):
    """Executes full RAG query pipeline and returns structured telemetry and answer."""
    if not req.query or not req.query.strip():
        raise HTTPException(status_code=400, detail="Query string cannot be empty.")

    try:
        res = run_pipeline(req.query.strip())
        intent_obj = res.get("intent")
        intent_type = intent_obj.intent_type if intent_obj else "UNKNOWN"
        intent_conf = intent_obj.confidence if intent_obj else 0.0

        return {
            "status": "success",
            "query": req.query,
            "intent": {
                "type": intent_type,
                "confidence": intent_conf,
                "reasoning": intent_obj.reasoning if intent_obj else ""
            },
            "routing": res.get("route", "web_rag"),
            "route_meta": res.get("route_meta", {}),
            "verification": {
                "score": res.get("final_score", 1.0),
                "dimensions": res.get("verification_dimensions", {}),
                "passed": res.get("passed", True)
            },
            "sac_reward": res.get("sac_reward", 2.0),
            "final_answer": res.get("final_answer", "")
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/mcp/tools")
def get_mcp_tools():
    """Returns standard MCP tool definitions schema."""
    return {
        "jsonrpc": "2.0",
        "result": {"tools": MCP_TOOLS_MANIFEST}
    }


@app.post("/api/mcp/call")
def call_mcp_tool(req: MCPRequest):
    """Executes standard MCP JSON-RPC 2.0 tool calls."""
    if req.raw_json:
        raw_resp = handle_mcp_request(req.raw_json)
        return json.loads(raw_resp)

    if req.tool_name:
        rpc_payload = json.dumps({
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {
                "name": req.tool_name,
                "arguments": req.arguments or {}
            }
        })
        raw_resp = handle_mcp_request(rpc_payload)
        return json.loads(raw_resp)

    raise HTTPException(status_code=400, detail="Provide raw_json or tool_name/arguments.")


if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
