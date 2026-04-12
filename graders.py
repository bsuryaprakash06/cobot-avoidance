"""
graders.py — Root-level re-export for OpenEnv judge.
"""
import sys
import os

_server_path = os.path.join(os.path.dirname(__file__), "server")
if _server_path not in sys.path:
    sys.path.insert(0, _server_path)

try:
    from server.graders import EasyGrader, MediumGrader, HardGrader
except ImportError:
    from graders import EasyGrader, MediumGrader, HardGrader

__all__ = ["EasyGrader", "MediumGrader", "HardGrader"]
