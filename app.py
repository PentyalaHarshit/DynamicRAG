"""
FastAPI Backend Web Application Server for Dynamic Hybrid RAG Pipeline
Exposes REST APIs for query execution, MCP JSON-RPC tool testing, and pipeline telemetry.
"""

import json
import os
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import uvicorn

from graph import run_pipeline
from answer_style_detector import detect_answer_style
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

FRONTEND_DIST = Path(__file__).parent / "frontend" / "dist"

if (FRONTEND_DIST / "assets").exists():
    app.mount("/assets", StaticFiles(directory=str(FRONTEND_DIST / "assets")), name="assets")


@app.get("/")
def read_root():
    """Serves frontend single-page application."""
    index_file = FRONTEND_DIST / "index.html"
    if index_file.exists():
        return FileResponse(str(index_file))
    return {
        "status": "online",
        "service": "Dynamic Hybrid RAG API Server",
        "docs": "/docs",
        "health": "/api/health"
    }


class QueryRequest(BaseModel):
    query: str


class MCPRequest(BaseModel):
    raw_json: Optional[str] = None
    tool_name: Optional[str] = None
    arguments: Optional[Dict[str, Any]] = None


@app.get("/health")
@app.get("/api/health")
def health_check():
    """Returns system health and active model configuration."""
    return {
        "status": "healthy",
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
            "domain":            res.get("domain") or res.get("problem_analysis", {}).get("domain", "GENERAL"),
            "operation_pattern": res.get("problem_analysis", {}).get("operation_pattern", "GENERAL"),
            "answer_style":      res.get("answer_style") or detect_answer_style(req.query.strip()).to_dict(),
            "sport":             res.get("sport", None),
            "entities":          res.get("entities", None),
            "statistic":         res.get("statistic", None),
            "operation":         res.get("operation", None),
            "time_scope":        res.get("time_scope", None),
            "data_source":       res.get("data_source", None),
            "verification": {
                "score": res.get("final_score", 1.0),
                "dimensions": res.get("verification_dimensions", {}),
                "passed": res.get("passed", True)
            },
            "sac_reward": res.get("sac_reward", 2.0),
            "funnel_meta": res.get("funnel_meta", {}),
            "final_answer": res.get("final_answer", "")
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


class MultimodalQueryRequest(BaseModel):
    query: str = ""
    image_base64: Optional[str] = None
    image_path: Optional[str] = None
    pdf_path: Optional[str] = None


@app.post("/api/query/multimodal")
def execute_multimodal_query(req: MultimodalQueryRequest):
    """Executes Multimodal Agent query pipeline (Text + Image + PDF)."""
    if not req.query and not req.image_base64 and not req.image_path and not req.pdf_path:
        raise HTTPException(status_code=400, detail="Provide at least query text or file/image payload.")

    try:
        res = run_pipeline(
            question=req.query or "",
            image_path=req.image_path,
            image_base64=req.image_base64,
            pdf_path=req.pdf_path
        )
        intent_obj = res.get("intent")
        return {
            "status": "success",
            "query": req.query,
            "routing": res.get("route", "multimodal"),
            "domain": res.get("domain", "MULTIMODAL"),
            "intent": {
                "type": intent_obj.intent_type if intent_obj else "MULTIMODAL",
                "confidence": intent_obj.confidence if intent_obj else 0.95
            },
            "verification": {
                "score": res.get("final_score", 1.0),
                "passed": res.get("passed", True)
            },
            "funnel_meta": res.get("funnel_meta", {}),
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


@app.post("/api/modes/research")
def execute_deep_research(req: QueryRequest):
    """Executes multi-source Deep Research Mode and Causal Analysis synthesis."""
    if not req.query or not req.query.strip():
        raise HTTPException(status_code=400, detail="Topic cannot be empty.")
    try:
        from causal_research_engine import execute_causal_research
        report = execute_causal_research(req.query.strip())
        return {
            "status": "success",
            "mode": "CAUSAL_RESEARCH",
            "topic": report.topic,
            "anchor_entity": report.anchor_entity,
            "executive_summary": report.executive_summary,
            "primary_causes": report.primary_causes,
            "contributing_factors": report.contributing_factors,
            "transmission_mechanisms": report.transmission_mechanisms,
            "amplification_mechanisms": report.amplification_mechanisms,
            "triggers_and_catalysts": report.triggers_and_catalysts,
            "systemic_consequences": report.systemic_consequences,
            "causal_graph": [
                {
                    "factor": n.factor_name,
                    "role": n.role.value,
                    "description": n.description,
                    "leads_to": n.leads_to,
                    "evidence": n.evidence_summary
                }
                for n in report.causal_graph
            ],
            "causal_edges": [
                {
                    "source": e.source_factor,
                    "target": e.target_factor,
                    "relationship": e.relationship_label,
                    "evidence": e.evidence_snippet,
                    "source_title": e.supporting_source_title,
                    "confidence": e.confidence
                }
                for e in report.causal_edges
            ],
            "verified_claims": [
                {
                    "claim": c.claim_text,
                    "role": c.causal_role.value,
                    "status": c.support_status,
                    "confidence": c.confidence,
                    "supporting_sources": c.supporting_sources,
                    "contradicting_sources": c.contradicting_sources,
                    "evidence_snippets": c.evidence_snippets,
                    "nuance_note": c.nuance_note
                }
                for c in report.verified_claims
            ],
            "contradiction_analysis": [
                {
                    "claim_a": ca.claim_a,
                    "source_a": ca.source_a,
                    "claim_b": ca.claim_b,
                    "source_b": ca.source_b,
                    "is_contradiction": ca.is_contradiction,
                    "reconciliation_nuance": ca.reconciliation_nuance
                }
                for ca in report.contradiction_analysis
            ],
            "evidence_coverage": {
                "subtopic_coverage": report.evidence_coverage.subtopic_coverage,
                "dimension_coverage": report.evidence_coverage.dimension_coverage,
                "overall_evidence_score": report.evidence_coverage.overall_evidence_score,
                "is_ready_to_answer": report.evidence_coverage.is_ready_to_answer,
                "gap_warnings": report.evidence_coverage.gap_warnings
            },
            "top_5_evidence_chunks": report.top_5_evidence_chunks,
            "tier_1_sources_used": report.tier_1_sources_used,
            "total_sources_analyzed": report.total_sources_analyzed,
            "cross_source_consensus": report.cross_source_consensus
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/modes/graph")
def query_knowledge_graph(req: QueryRequest):
    """Queries entity relations and invariant triples from the Knowledge Graph."""
    if not req.query or not req.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")
    try:
        from knowledge_graph_engine import get_knowledge_graph
        kg = get_knowledge_graph()
        data = kg.query_graph_context(req.query.strip())
        return {
            "status": "success",
            "query": req.query,
            "matched_entities": data.get("matched_entities", []),
            "relational_facts": data.get("relational_facts", []),
            "raw_triples": data.get("raw_triples", [])
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/modes/evaluate")
def get_evaluation_dashboard():
    """Returns aggregated evaluation benchmark metrics and failure diagnostics."""
    try:
        from evaluation_engine import get_evaluation_engine
        eval_engine = get_evaluation_engine()
        return {
            "status": "success",
            "dashboard": eval_engine.get_dashboard_metrics()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/modes/adaptive_quiz")
def get_adaptive_quiz_config(topic: Optional[str] = None):
    """Returns adaptive learning distribution and weak concept targeting."""
    try:
        from agents.quiz_agent import get_user_knowledge_graph
        kg = get_user_knowledge_graph()
        distribution = kg.get_adaptive_quiz_distribution(topic=topic)
        return {
            "status": "success",
            "topic": topic or "All Topics",
            "distribution": distribution
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
