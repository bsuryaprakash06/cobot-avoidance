class EasyGrader:
    def grade(self, env, *args, **kwargs):
        try:
            state = env.get_state()
            collisions = 3 - state.cobot.integrity
            score = (state.cobot.integrity / 3.0) * (state.step / 100.0)
            return max(0.001, min(0.999, float(round(score, 4))))
        except Exception:
            return 0.5


class MediumGrader:
    def grade(self, env, *args, **kwargs):
        try:
            state = env.get_state()
            score = 0.5 * (state.cobot.integrity / 3.0) + 0.5 * 0.8
            return max(0.001, min(0.999, float(round(score, 4))))
        except Exception:
            return 0.5


class HardGrader:
    def grade(self, env, *args, **kwargs):
        try:
            state = env.get_state()
            damage = 3 - state.obstacle.integrity

            score = 0.4 * (state.cobot.integrity / 3.0) + 0.6 * (damage / 3.0)

            if state.obstacle.integrity <= 0 and state.step < 80:
                score += 0.1

            return max(0.001, min(0.999, float(round(score, 4))))
        except Exception:
            return 0.5
