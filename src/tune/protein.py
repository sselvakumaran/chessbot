"Protein (puffer.ai/blog/protein): cost-aware hyperparameter search around the Pareto front of score vs cost"
from dataclasses import dataclass
from typing import Any, Callable
import math

import numpy as np
import torch

Values = dict[str, Any]
SQRT2, SQRT3 = math.sqrt(2), math.sqrt(3)
softplus = torch.nn.functional.softplus
inv_softplus = lambda v: math.log(math.expm1(v))

@dataclass
class Param:
  "one searched setting, mapped to u in [-1, 1]"
  lo: float
  hi: float
  kind: str = "linear" # linear | log | logit (values near 1)
  integer: bool = False
  scale: float = 0.5 # half-width of the candidate box in u

  def _t(self, v: float) -> float:
    if self.kind == "log": return math.log(v)
    if self.kind == "logit": return -math.log(1 - v)
    return v

  def to_u(self, v: float) -> float:
    lo, hi = self._t(self.lo), self._t(self.hi)
    return 2 * (self._t(v) - lo) / (hi - lo) - 1

  def from_u(self, u: float) -> float | int:
    lo, hi = self._t(self.lo), self._t(self.hi)
    t = lo + (u + 1) / 2 * (hi - lo)
    v = math.exp(t) if self.kind == "log" else 1 - math.exp(-t) if self.kind == "logit" else t
    return int(round(v)) if self.integer else float(v)

class GP:
  """PufferLib's GP model: constant mean + scale x (linear + Matern 3/2 with a lengthscale per input), fitted on
  construction by MAP (L-BFGS) with BoTorch's default priors: lengthscale LogNormal(sqrt2 + log(d) / 2 + log 2, sqrt3)
  (Hvarfner et al. 2024; + log 2 because inputs span [-1, 1]), noise LogNormal(-4, 1).
  With a basis H the mean is h(x)' beta instead (Rasmussen & Williams 2006, sec. 2.7): beta by generalised least
  squares under the GP covariance, integrated out of the marginal likelihood (eq. 2.45). beta has the weak prior
  N(0, 1e6 I) (eq. 2.42), so a basis the data cannot pin down yet (a new sweep's first runs) still fits"""

  def __init__(
    self,
    X: np.ndarray,
    y: np.ndarray,
    H: np.ndarray | None = None
  ):
    n, d = X.shape
    X, y = torch.as_tensor(X, dtype=torch.float64), torch.as_tensor(y, dtype=torch.float64)[:, None]
    H = None if H is None else torch.as_tensor(H, dtype=torch.float64)
    eye = torch.eye(n, dtype=torch.float64)
    prior = None if H is None else 1e-6 * torch.eye(H.shape[1], dtype=torch.float64) # inverse covariance of beta
    mu_ell = SQRT2 + 0.5 * math.log(d) + math.log(2)
    raw = lambda shape, v: torch.full(shape, v, dtype=torch.float64, requires_grad=True)
    # start at the priors' modes (exp(mu - sd^2))
    self.mean, self.raw_scale, self.raw_offset = raw((), 0.0), raw((), 0.0), raw((), 0.0)
    self.raw_ell, self.raw_noise = raw((d,), inv_softplus(math.exp(mu_ell - 3))), raw((), inv_softplus(math.exp(-5)))
    log_lognormal = lambda x, mu, sd: (-(torch.log(x) - mu) ** 2 / (2 * sd * sd) - torch.log(x)).sum()
    params = [self.mean, self.raw_scale, self.raw_offset, self.raw_ell, self.raw_noise]
    opt = torch.optim.LBFGS(params, lr=1, max_iter=200, line_search_fn="strong_wolfe")
    best = [math.inf, None] # lowest loss evaluated, its parameters

    def closure() -> torch.Tensor:
      opt.zero_grad()
      L = torch.linalg.cholesky(self._k(X, X) + self._noise() * eye)
      r = y - self.mean if H is None else y
      log_ml = -0.5 * (r * torch.cholesky_solve(r, L)).sum() - torch.log(torch.diagonal(L)).sum()
      if H is not None: # beta integrated out (eq. 2.45)
        LA = torch.linalg.cholesky(H.T @ torch.cholesky_solve(H, L) + prior)
        g = H.T @ torch.cholesky_solve(y, L)
        log_ml = log_ml + 0.5 * (g * torch.cholesky_solve(g, LA)).sum() - torch.log(torch.diagonal(LA)).sum()
      log_prior = (
        log_lognormal(softplus(self.raw_ell).clamp_min(0.025), mu_ell, SQRT3) + log_lognormal(self._noise(), -4, 1)
      )
      loss = -(log_ml + log_prior) / n
      if loss.item() < best[0]: best[:] = loss.item(), [p.detach().clone() for p in params]
      loss.backward()
      return loss

    try:
      opt.step(closure)
    except torch.linalg.LinAlgError: # a line-search step reached a covariance that is not positive definite
      with torch.no_grad():
        for p, value in zip(params, best[1]): p.copy_(value)
    with torch.no_grad():
      self.raw_ell.clamp_(min=inv_softplus(0.025))
      L = torch.linalg.cholesky(self._k(X, X) + self._noise() * eye)
      self.beta = None if H is None else torch.linalg.solve(
        H.T @ torch.cholesky_solve(H, L) + prior, H.T @ torch.cholesky_solve(y, L)
      )
      self.X, self.alpha = X, torch.cholesky_solve(y - (self.mean if H is None else H @ self.beta), L)

  def _noise(self) -> torch.Tensor:
    return softplus(self.raw_noise) + 1e-4

  def _k(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    ell = softplus(self.raw_ell)
    d = torch.cdist(A / ell, B / ell).clamp_min(1e-12) * SQRT3
    return softplus(self.raw_scale) * (A @ B.T + softplus(self.raw_offset) + (1 + d) * torch.exp(-d))

  def predict(self, Z: np.ndarray, Hz: np.ndarray | None = None) -> np.ndarray:
    with torch.no_grad():
      Z = torch.as_tensor(Z, dtype=torch.float64)
      mean = self.mean if self.beta is None else torch.as_tensor(Hz, dtype=torch.float64) @ self.beta
      return (self._k(Z, self.X) @ self.alpha + mean)[:, 0].numpy()

class Protein:
  """Observations in, next settings out (stateless). An observation is {"values", "score" (higher = better), "cost"
  (seconds), "failed"}. As the write-up: two GPs (score scaled 0..1, cost), candidates = `per_pareto` uniform samples
  in the box around each Pareto point, pick = argmax GP_score x (1 - |alpha x U(0, 1) - cost_norm|), means only.
  Failed runs enter the score GP at the lowest score while there are < 100 observations.
  Cost (deviation): basis(values) -> (steps, h); cost = steps x (h' beta + a GP on `cost_dims`)"""

  def __init__(
    self,
    space: dict[str, Param],
    defaults: Values,
    basis: Callable[[Values], tuple[float, list[float]]],
    cost_dims: list[str],
    max_cost: float,
    alpha: float,
    seed: int,
    per_pareto: int = 256
  ):
    self.space, self.defaults, self.basis, self.max_cost = space, defaults, basis, max_cost
    self.alpha, self.seed, self.per_pareto = alpha, seed, per_pareto
    self.cost_idx = [list(space).index(k) for k in cost_dims]
    self.box = np.array([p.scale for p in space.values()])

  def suggest(
    self,
    observations: list[dict],
    fix: Callable[[Values], Values]
  ) -> tuple[Values, float | None, float | None] | None:
    """(settings, predicted seconds, predicted score); None if the first run failed (nothing to model) or every
    candidate costs max_cost or more"""
    if not observations: return fix(dict(self.defaults)), None, None
    ok = [o for o in observations if not o.get("failed")]
    if not ok: return None
    bad = [o for o in observations if o.get("failed")] if len(ok) < 100 else []
    rng = np.random.default_rng([self.seed, len(observations)])
    to_u = lambda values: np.array([p.to_u(values[k]) for k, p in self.space.items()])
    basis = lambda values: [np.array(column, float) for column in zip(*map(self.basis, values))] # steps, H

    X, y = np.array([to_u(o["values"]) for o in ok]), np.array([o["score"] for o in ok])
    Xf = np.array([to_u(o["values"]) for o in bad]).reshape(len(bad), len(self.space))
    cost = np.array([o["cost"] for o in ok])
    log_c = np.log(cost)
    gp_score = GP(np.vstack([Xf, X]), np.concatenate([np.zeros(len(bad)), (y - y.min()) / (np.ptp(y) + 1e-6)]))
    steps, H = basis([o["values"] for o in ok])
    unit = (cost / steps).max()
    gp_cost = GP(X[:, self.cost_idx], cost / steps / unit, H)

    front, best = [], -np.inf
    for i in np.argsort(log_c): # Pareto front: cheapest first, a point is kept only if it beats every cheaper point
      if y[i] <= best + 1e-6: continue
      front.append(i)
      best = y[i]
    centers = X[sorted(front)]
    n = len(centers) * self.per_pareto
    picked = centers[rng.integers(len(centers), size=n)] # draw order matters: same seed, same candidates
    raw = np.clip(picked + rng.uniform(-1, 1, (n, len(self.space))) * self.box, -1, 1)
    # snapped: the GPs see what would run
    cands = [fix({k: p.from_u(u) for (k, p), u in zip(self.space.items(), row)}) for row in raw]
    U = np.array([to_u(v) for v in cands])

    steps, H = basis(cands)
    seconds = steps * np.maximum(gp_cost.predict(U[:, self.cost_idx], H) * unit, 1e-3) # a GP can extrapolate below 0
    cost_norm = (np.log(seconds) - log_c.min()) / (np.ptp(log_c) + 1e-6)
    score_norm = gp_score.predict(U)
    weight = score_norm * (1 - np.abs(self.alpha * rng.uniform() - cost_norm))
    weight[seconds >= self.max_cost] = -np.inf
    if np.isneginf(weight).all(): return None
    i = int(np.argmax(weight))
    return cands[i], float(seconds[i]), float(y.min() + score_norm[i] * (np.ptp(y) + 1e-6))
