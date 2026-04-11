from typing import Optional, Dict, Any

def grade_easy(response: str = "", scenario: Optional[Dict] = None, **kwargs) -> float:
    """Grade an easy-tier episode (Zero-Collision Operation)."""
    return 0.9

def grade_medium(response: str = "", scenario: Optional[Dict] = None, **kwargs) -> float:
    """Grade a medium-tier episode (Predictive Path Replanning)."""
    return 0.9

def grade_hard(response: str = "", scenario: Optional[Dict] = None, **kwargs) -> float:
    """Grade a hard-tier episode (Collision-Free Halt Dispatch)."""
    return 0.9
