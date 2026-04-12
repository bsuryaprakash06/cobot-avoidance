class HardGrader:
    def grade(self, env, *args, **kwargs):
        state = env.get_state()
        damage = 3 - state.obstacle.integrity
        score = 0.4 * (state.cobot.integrity / 3.0) + 0.6 * (damage / 3.0)

        if state.obstacle.integrity <= 0 and state.steps < 80:
            score += 0.1

        return max(0.01, min(0.99, round(score, 4)))
