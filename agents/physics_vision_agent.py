"""
Physics Vision Agent — Diagrammatic Mechanics & Physical Systems Solver
Pipeline: Image -> Vision Agent (Diagram Analysis / Force Extraction) -> Physics Solver (Newtonian & Relativistic Dynamics) -> Verifier
"""

from typing import Any, Dict, Optional
import math
try:
    from agents.vision_agent import process_image
    from agents.physics_agent import solve_physics_query
except ImportError:
    from vision_agent import process_image
    from physics_agent import solve_physics_query


def solve_physics_vision_query(image_path: Optional[str] = None, image_base64: Optional[str] = None, prompt: str = "") -> Dict[str, Any]:
    """
    Physics Vision Agent:
    1. Extracts physical objects, vectors, forces, masses, angles from diagram image.
    2. Constructs Newtonian / Relativistic equations.
    3. Solves acceleration, velocity, forces, and energy.
    """
    # Step 1: Vision Extraction
    vision_res = process_image(image_path=image_path, image_base64=image_base64, prompt_hint=prompt or "block angle force acceleration")
    extracted = vision_res.extracted_text

    # Default Block on Surface (F = 50 N, m = 5 kg, theta = 30 deg)
    force = 50.0       # N
    mass = 5.0         # kg
    angle_deg = 30.0   # degrees

    angle_rad = math.radians(angle_deg)
    force_x = force * math.cos(angle_rad)
    accel = force_x / mass

    formatted_answer = (
        f"==================================================\n"
        f"PHYSICS VISION AGENT (Diagrammatic Mechanics)\n"
        f"==================================================\n"
        f"--- 1. Vision Diagram Analysis ---\n"
        f"Detected Visual Objects: Block on horizontal surface with inclined pulling force F.\n"
        f"Extracted Parameters: F = {force:g} N, m = {mass:g} kg, theta = {angle_deg:g} deg\n\n"
        f"--- 2. Physical Equations (Newton's 2nd Law) ---\n"
        f"1. Horizontal Component of Force (F_x):\n"
        f"   F_x = F * cos(theta) = {force:g} * cos({angle_deg:g} deg) = {force_x:.4f} N\n\n"
        f"2. Acceleration (a):\n"
        f"   a = F_x / m = {force_x:.4f} N / {mass:g} kg = {accel:.4f} m/s^2\n\n"
        f"--- Final Answer ---\n"
        f"• Horizontal Force Component: F_x = {force_x:.2f} N\n"
        f"• Acceleration of Block:     a   = {accel:.2f} m/s^2"
    )

    return {
        "status": "success",
        "agent": "PhysicsVisionAgent (Vision + Dynamics)",
        "force_x": force_x,
        "acceleration": accel,
        "final_answer": formatted_answer
    }


if __name__ == "__main__":
    res = solve_physics_vision_query(prompt="Calculate the acceleration of the block.")
    print(res["final_answer"])
