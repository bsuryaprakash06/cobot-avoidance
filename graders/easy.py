class EasyGrader:
    def grade(self, env, *args, **kwargs):
        state = env.get_state()
        collisions = 3 - state.cobot.integrity
        score = (state.cobot.integrity / 3.0) * (state.steps / 100.0)
        return max(0.01, min(0.99, round(score, 4)))
