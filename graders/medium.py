class MediumGrader:
    def grade(self, env, *args, **kwargs):
        state = env.get_state()
        score = 0.5 * (state.cobot.integrity / 3.0) + 0.5 * 0.8
        return max(0.01, min(0.99, round(score, 4)))
