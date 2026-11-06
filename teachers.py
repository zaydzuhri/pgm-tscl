"""Teachers for Teacher-Student Curriculum Learning (Matiisen et al., 2017).

We use the *batch* POMDP formulation (Sec 2.2 / Appendix B), the one suited to supervised learning:

  action      a_t : a probability distribution over the N task buckets; the next
                    `teacher_interval` optimizer steps draw their batches from it.
  observation o_t : a vector of per-bucket scores measured after those steps
                    (higher = better, e.g. -val_loss or val accuracy).
  reward      r_t : the change in scores (Online/Sampling) or the slope of the
                    per-bucket learning curve (Naive/Window).

Every teacher implements two methods:
  distribution() -> np.ndarray[N]  the action for the next round
  update(scores)                   observe o_t (the first call passes o_0, measured before training)

Baselines (no observation needed): `proportional` samples buckets by size, which is ordinary
shuffled finetuning; `uniform` gives every bucket equal mass. To add a new teacher, subclass
Teacher and register it in TEACHERS.
"""
from collections import deque

import numpy as np


class Teacher:
    needs_scores = True

    def __init__(self, n_tasks, sizes, seed=0, **unused):
        if unused:
            raise TypeError(f"{type(self).__name__} got unknown kwargs: {sorted(unused)}")
        self.n = n_tasks
        self.sizes = np.asarray(sizes, dtype=np.float64)
        self.rng = np.random.default_rng(seed)

    def distribution(self) -> np.ndarray:
        raise NotImplementedError

    def update(self, scores: np.ndarray) -> None:
        pass

    def state(self) -> dict:
        """Extra info to log every round."""
        return {}


# ----------------------------------------------------------------------------- baselines
class ProportionalTeacher(Teacher):
    """p_i ∝ |bucket_i|, which is plain i.i.d. finetuning on the whole training set."""
    needs_scores = False

    def distribution(self):
        return self.sizes / self.sizes.sum()


class UniformTeacher(Teacher):
    """p_i = 1/N, the 'uniform sampling of subtasks' baseline from the paper."""
    needs_scores = False

    def distribution(self):
        return np.full(self.n, 1.0 / self.n)


# ----------------------------------------------------------------------------- helpers
def _argmax_random_tie(x, rng):
    return int(rng.choice(np.flatnonzero(x == x.max())))


def _eps_greedy(values, eps, rng):
    p = np.full(len(values), eps / len(values))
    p[_argmax_random_tie(values, rng)] += 1.0 - eps
    return p


def _slopes(t, Y):
    """Least-squares slope of each column of Y (m x N) w.r.t. timesteps t (m,)."""
    t = np.asarray(t, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    tc = t - t.mean()
    denom = (tc ** 2).sum()
    return np.zeros(Y.shape[1]) if denom == 0 else tc @ (Y - Y.mean(0)) / denom


class BanditTeacher(Teacher):
    """Shared machinery for Online/Naive/Window: track an expected return Q per task with an
    EMA (Q <- alpha*r + (1-alpha)*Q) and act eps-greedily or by Boltzmann on |Q| (abs counters forgetting)."""

    def __init__(self, n_tasks, sizes, seed=0, alpha=0.1, policy="greedy", eps=0.1, tau=4e-4,
                 use_abs=True, **kw):
        super().__init__(n_tasks, sizes, seed, **kw)
        assert policy in ("greedy", "boltzmann")
        self.alpha, self.policy, self.eps, self.tau, self.use_abs = alpha, policy, eps, tau, use_abs
        self.Q = np.zeros(n_tasks)

    def _policy(self):
        v = np.abs(self.Q) if self.use_abs else self.Q
        if self.policy == "greedy":
            return _eps_greedy(v, self.eps, self.rng)
        z = v / self.tau
        z = np.exp(z - z.max())
        return z / z.sum()

    def _learn(self, r):
        self.Q = self.alpha * np.asarray(r) + (1 - self.alpha) * self.Q

    def distribution(self):
        return self._policy()

    def state(self):
        return {"Q": self.Q.tolist()}


# ----------------------------------------------------------------------------- TSCL teachers
class OnlineTeacher(BanditTeacher):
    """Algorithm 5: r_t = o_t - o_{t-1}."""

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.prev = None

    def update(self, scores):
        scores = np.asarray(scores, dtype=np.float64)
        if self.prev is not None:
            self._learn(scores - self.prev)
        self.prev = scores


class NaiveTeacher(BanditTeacher):
    """Algorithm 6: hold one distribution for K rounds, then use the regression slope of the K scores as r."""

    def __init__(self, *a, K=5, **kw):
        super().__init__(*a, **kw)
        self.K, self.frozen, self.D = K, None, []

    def distribution(self):
        if self.frozen is None:
            self.frozen = self._policy()
        return self.frozen

    def update(self, scores):
        if self.frozen is None:  # o_0 comes before any action was taken, so skip it
            return
        self.D.append(np.asarray(scores, dtype=np.float64))
        if len(self.D) == self.K:
            self._learn(_slopes(np.arange(self.K), self.D))
            self.D, self.frozen = [], None


class WindowTeacher(BanditTeacher):
    """Algorithm 7: FIFO buffer of the last K score vectors; r = regression slope over that window."""

    def __init__(self, *a, K=10, **kw):
        super().__init__(*a, **kw)
        self.buf = deque(maxlen=K)
        self.t = 0

    def update(self, scores):
        self.buf.append((self.t, np.asarray(scores, dtype=np.float64)))
        self.t += 1
        if len(self.buf) >= 2:
            ts, ys = zip(*self.buf)
            self._learn(_slopes(ts, ys))


class SamplingTeacher(Teacher):
    """Algorithm 8 (Thompson-sampling flavoured): keep the last K rewards per task, draw one per task,
    put (1-eps) mass on argmax |r~| and mix in eps-uniform."""

    def __init__(self, n_tasks, sizes, seed=0, K=10, eps=0.1, **kw):
        super().__init__(n_tasks, sizes, seed, **kw)
        self.eps = eps
        self.D = [deque(maxlen=K) for _ in range(n_tasks)]
        self.prev = None

    def distribution(self):
        r = np.array([self.rng.choice(list(d)) if d else 1.0 for d in self.D])
        return _eps_greedy(np.abs(r), self.eps, self.rng)

    def update(self, scores):
        scores = np.asarray(scores, dtype=np.float64)
        if self.prev is not None:
            for d, r in zip(self.D, scores - self.prev):
                d.append(r)
        self.prev = scores


TEACHERS = {
    "proportional": ProportionalTeacher,
    "uniform": UniformTeacher,
    "online": OnlineTeacher,
    "naive": NaiveTeacher,
    "window": WindowTeacher,
    "sampling": SamplingTeacher,
}


def make_teacher(name, n_tasks, sizes, seed=0, **kwargs) -> Teacher:
    return TEACHERS[name](n_tasks, sizes, seed=seed, **kwargs)
