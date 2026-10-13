"""Information Balance Adaptive Multimodal Weight Optimization (IB-AMWO).

The optimizer follows the user-provided Algorithm 1.
The authentication model is supplied through an evaluator callback:
    evaluator(weights, condition) -> accuracy in [0, 1]
where condition is one of: 'nofake', 'fake_face', 'fake_voice', 'fake_odor'.
"""
from dataclasses import dataclass, asdict
from typing import Callable, Dict, List, Tuple
import json
import numpy as np

Weights = Tuple[float, float, float]
Evaluator = Callable[[Weights, str], float]


@dataclass
class IBAMWOConfig:
    initial_weights: Weights = (0.34, 0.33, 0.33)
    alpha: float = 0.5
    epsilon: float = 0.005
    eta: float = 0.1
    theta: float = 0.95
    max_iterations: int = 100


class IBAMWO:
    def __init__(self, evaluator: Evaluator, config: IBAMWOConfig = IBAMWOConfig()):
        self.evaluator = evaluator
        self.cfg = config
        self.history: List[Dict] = []

    @staticmethod
    def enforce_constraints(weights) -> Weights:
        """Enforce w_f,w_v,w_o >= 0 and sum(weights)=1."""
        w = np.asarray(weights, dtype=np.float64)
        w = np.maximum(w, 0.0)
        s = float(w.sum())
        if s <= 0.0:
            w[:] = 1.0 / 3.0
        else:
            w /= s
        return tuple(float(x) for x in w)

    def fake_metrics(self, weights: Weights):
        Ff = float(self.evaluator(weights, "fake_face"))
        Fv = float(self.evaluator(weights, "fake_voice"))
        Fo = float(self.evaluator(weights, "fake_odor"))
        return Ff, Fv, Fo

    def information_quantities(self, Ff: float, Fv: float, Fo: float):
        eps = self.cfg.epsilon
        inv = np.array([
            1.0 / (Ff + eps),
            1.0 / (Fv + eps),
            1.0 / (Fo + eps),
        ], dtype=np.float64)
        info = inv / inv.sum()
        return tuple(float(x) for x in info)

    def objective(self, Ff: float, Fv: float, Fo: float) -> float:
        Favg = (Ff + Fv + Fo) / 3.0
        Fmax = max(Ff, Fv, Fo)
        return self.cfg.alpha * Favg + (1.0 - self.cfg.alpha) * Fmax

    def evaluate_objective(self, weights: Weights):
        Ff, Fv, Fo = self.fake_metrics(weights)
        J = self.objective(Ff, Fv, Fo)
        return J, Ff, Fv, Fo

    def optimize(self) -> Weights:
        W = self.enforce_constraints(self.cfg.initial_weights)
        nofake = float(self.evaluator(W, "nofake"))

        print(f"Initial W={W}, NoFake={nofake:.4f}")
        if nofake < self.cfg.theta:
            print(f"STOP: initial NoFake {nofake:.4f} < theta {self.cfg.theta:.4f}")
            return W

        for iteration in range(1, self.cfg.max_iterations + 1):
            # Steps 4-7: fake accuracies, information quantities, objective.
            J, Ff, Fv, Fo = self.evaluate_objective(W)
            If, Iv, Io = self.information_quantities(Ff, Fv, Fo)

            # Step 8: adaptive step sizes.
            a = self.cfg.eta * If
            b = self.cfg.eta * Iv

            # Steps 9-11: primary candidate (-a, -b, a+b).
            delta_primary = np.array([-a, -b, a + b], dtype=np.float64)
            W_candidate = self.enforce_constraints(np.asarray(W) + delta_primary)

            # Steps 12-13: candidate objective and improvement.
            J_candidate, Ffc, Fvc, Foc = self.evaluate_objective(W_candidate)
            delta_J = J - J_candidate

            used_fallback = False
            if delta_J > 0.0:
                W_new = W_candidate
                J_new = J_candidate
            else:
                # Steps 17-24: fallback direction from Algorithm 1.
                used_fallback = True
                wf, wv, _ = W
                if wf > wv:
                    delta_fallback = np.array([-(a + b), 0.0, a + b])
                else:
                    delta_fallback = np.array([0.0, -(a + b), a + b])
                W_new = self.enforce_constraints(np.asarray(W) + delta_fallback)
                J_new, _, _, _ = self.evaluate_objective(W_new)

            # Steps 26-29: No-Fake constraint.
            nofake_new = float(self.evaluator(W_new, "nofake"))

            record = {
                "iteration": iteration,
                "wf": W[0], "wv": W[1], "wo": W[2],
                "Ff": Ff, "Fv": Fv, "Fo": Fo,
                "If": If, "Iv": Iv, "Io": Io,
                "J": J,
                "candidate_wf": W_new[0],
                "candidate_wv": W_new[1],
                "candidate_wo": W_new[2],
                "candidate_J": J_new,
                "delta_J_primary": delta_J,
                "NoFake_candidate": nofake_new,
                "fallback": used_fallback,
                "accepted": nofake_new >= self.cfg.theta,
            }
            self.history.append(record)

            print(
                f"[{iteration:03d}] W=({W[0]:.4f},{W[1]:.4f},{W[2]:.4f}) "
                f"F=({Ff:.4f},{Fv:.4f},{Fo:.4f}) J={J:.6f} "
                f"I=({If:.4f},{Iv:.4f},{Io:.4f}) -> "
                f"Wnew=({W_new[0]:.4f},{W_new[1]:.4f},{W_new[2]:.4f}) "
                f"NoFake={nofake_new:.4f} fallback={used_fallback}"
            )

            if nofake_new < self.cfg.theta:
                print(f"STOP: NoFake(Wnew)={nofake_new:.4f} < theta={self.cfg.theta:.4f}")
                break

            W = W_new

        print(f"Optimized W*=({W[0]:.6f}, {W[1]:.6f}, {W[2]:.6f})")
        return W

    def save_history(self, path: str = "ib_amwo_history.json"):
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"config": asdict(self.cfg), "history": self.history}, f,
                      ensure_ascii=False, indent=2)


# -----------------------------------------------------------------------------
# Integration example
# Replace this function with validation inference from your multimodal model.
# -----------------------------------------------------------------------------
def model_evaluator(weights: Weights, condition: str) -> float:
    """Template only. Return measured validation accuracy for the condition."""
    raise NotImplementedError(
        "Connect model_evaluator() to your multimodal validation code. "
        "conditions: nofake/fake_face/fake_voice/fake_odor"
    )


if __name__ == "__main__":
    cfg = IBAMWOConfig(
        initial_weights=(0.34, 0.33, 0.33),
        alpha=0.5,
        epsilon=0.005,
        eta=0.1,
        theta=0.95,
        max_iterations=100,
    )

    # Usage after connecting the evaluator:
    # optimizer = IBAMWO(model_evaluator, cfg)
    # best_weights = optimizer.optimize()
    # optimizer.save_history("ib_amwo_history.json")
    # print("Best weights:", best_weights)
