"""
Unit tests for CollisionAvoidanceEnv, Pydantic telemetry models, and task graders.
"""

import sys
import os
import unittest

# Ensure repo root is on sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import ActionInput
from server.robot_arm_environment import CollisionAvoidanceEnv
from graders import grade_zero_collision, grade_predictive_replan, grade_halt_dispatch


class TestCollisionAvoidanceEnvironment(unittest.TestCase):
    def setUp(self):
        self.env = CollisionAvoidanceEnv(seed=42)

    def test_environment_initial_state(self):
        """Test reset initializes environment with expected parameters."""
        res = self.env.reset()
        state = res.state
        self.assertEqual(state.step, 0)
        self.assertFalse(state.done)
        self.assertEqual(state.cobot.integrity, 3)
        self.assertEqual(state.obstacle.integrity, 3)
        self.assertEqual(state.cobot.x, 2.0)
        self.assertEqual(state.cobot.y, 10.0)

    def test_step_execution_angular_rotation(self):
        """Test step execution rotates the end-effector within valid ranges."""
        self.env.reset()
        init_angle = self.env.get_state().cobot.angle
        action = ActionInput(delta_angle=10.0, send_halt=False)
        result = self.env.step(action)

        self.assertEqual(result.state.step, 1)
        self.assertAlmostEqual(result.state.cobot.angle, init_angle + 10.0)
        self.assertIsInstance(result.reward, float)

    def test_graders_score_bounds(self):
        """Test grading functions always produce normalized scores in [0.001, 0.999]."""
        self.env.reset()
        for _ in range(10):
            self.env.step(ActionInput(delta_angle=5.0, send_halt=False))

        score_1 = grade_zero_collision(self.env)
        score_2 = grade_predictive_replan(self.env)
        score_3 = grade_halt_dispatch(self.env)

        self.assertTrue(0.001 <= score_1 <= 0.999)
        self.assertTrue(0.001 <= score_2 <= 0.999)
        self.assertTrue(0.001 <= score_3 <= 0.999)


if __name__ == "__main__":
    unittest.main()
