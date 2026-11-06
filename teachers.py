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

Paper notation kept as-is: Q (expected return per task), alpha (EMA learning rate), tau (Boltzmann
temperature), K (buffer/window length), D (buffer of recent rewards).
"""
from collections import deque

import numpy as np


class Teacher:
    needs_scores = True

    def __init__(self, num_tasks, bucket_sizes, seed=0, **unused_kwargs):
        if unused_kwargs:
            raise TypeError(f"{type(self).__name__} got unknown kwargs: {sorted(unused_kwargs)}")
        self.num_tasks = num_tasks
        self.bucket_sizes = np.asarray(bucket_sizes, dtype=np.float64)
        self.random_generator = np.random.default_rng(seed)

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
        return self.bucket_sizes / self.bucket_sizes.sum()


class UniformTeacher(Teacher):
    """p_i = 1/N, the 'uniform sampling of subtasks' baseline from the paper."""
    needs_scores = False

    def distribution(self):
        return np.full(self.num_tasks, 1.0 / self.num_tasks)


# ----------------------------------------------------------------------------- helpers
def _argmax_random_tie(values, random_generator):
    """Index of the largest value; ties are broken at random (at the start all Q are 0)."""
    return int(random_generator.choice(np.flatnonzero(values == values.max())))


def _epsilon_greedy(values, epsilon, random_generator):
    """Distribution with (1 - epsilon) mass on the best task and epsilon spread uniformly over all tasks."""
    probs = np.full(len(values), epsilon / len(values))
    probs[_argmax_random_tie(values, random_generator)] += 1.0 - epsilon
    return probs


def _slopes(timesteps, scores):
    """Linear-regression slope of each task's learning curve.
    timesteps: shape (m,), scores: shape (m, N) -> slopes: shape (N,)."""
    timesteps = np.asarray(timesteps, dtype=np.float64)
    scores = np.asarray(scores, dtype=np.float64)
    centered_timesteps = timesteps - timesteps.mean()
    denominator = (centered_timesteps ** 2).sum()
    if denominator == 0:
        return np.zeros(scores.shape[1])
    return centered_timesteps @ (scores - scores.mean(axis=0)) / denominator


class BanditTeacher(Teacher):
    """Shared machinery for Online/Naive/Window (non-stationary bandit, Sec 3.1):
    track an expected return Q per task with an exponentially weighted moving average
        Q <- alpha * r + (1 - alpha) * Q
    and choose the distribution from |Q| (absolute value, so tasks being forgotten get practiced too)
    with either epsilon-greedy or a Boltzmann distribution."""

    def __init__(self, num_tasks, bucket_sizes, seed=0, alpha=0.1, policy="greedy", epsilon=0.1,
                 tau=4e-4, use_abs=True, **kwargs):
        super().__init__(num_tasks, bucket_sizes, seed, **kwargs)
        assert policy in ("greedy", "boltzmann")
        self.alpha = alpha
        self.policy = policy
        self.epsilon = epsilon
        self.tau = tau
        self.use_abs = use_abs
        self.Q = np.zeros(num_tasks)

    def _policy(self):
        values = np.abs(self.Q) if self.use_abs else self.Q
        if self.policy == "greedy":
            return _epsilon_greedy(values, self.epsilon, self.random_generator)
        # Boltzmann: p(a) = exp(Q(a)/tau) / sum_i exp(Q(i)/tau)
        logits = values / self.tau
        exp_logits = np.exp(logits - logits.max())
        return exp_logits / exp_logits.sum()

    def _learn(self, reward):
        self.Q = self.alpha * np.asarray(reward) + (1 - self.alpha) * self.Q

    def distribution(self):
        return self._policy()

    def state(self):
        return {"Q": self.Q.tolist()}


# ----------------------------------------------------------------------------- TSCL teachers
class OnlineTeacher(BanditTeacher):
    """Algorithm 5: reward r_t = o_t - o_{t-1}."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.previous_scores = None

    def update(self, scores):
        scores = np.asarray(scores, dtype=np.float64)
        if self.previous_scores is not None:
            self._learn(scores - self.previous_scores)
        self.previous_scores = scores


class NaiveTeacher(BanditTeacher):
    """Algorithm 6: keep the same distribution for K rounds, then use the regression slope of
    those K scores as the reward."""

    def __init__(self, *args, K=5, **kwargs):
        super().__init__(*args, **kwargs)
        self.K = K
        self.frozen_distribution = None
        self.D = []   # scores collected under the current frozen distribution

    def distribution(self):
        if self.frozen_distribution is None:
            self.frozen_distribution = self._policy()
        return self.frozen_distribution

    def update(self, scores):
        if self.frozen_distribution is None:  # o_0 comes before any action was taken, so skip it
            return
        self.D.append(np.asarray(scores, dtype=np.float64))
        if len(self.D) == self.K:
            self._learn(_slopes(np.arange(self.K), self.D))
            self.D = []
            self.frozen_distribution = None


class WindowTeacher(BanditTeacher):
    """Algorithm 7: FIFO buffer of the last K score vectors and the timesteps they were recorded at;
    reward = regression slope of the learning curve over that window."""

    def __init__(self, *args, K=10, **kwargs):
        super().__init__(*args, **kwargs)
        self.window = deque(maxlen=K)   # (timestep, scores) pairs
        self.timestep = 0

    def update(self, scores):
        self.window.append((self.timestep, np.asarray(scores, dtype=np.float64)))
        self.timestep += 1
        if len(self.window) >= 2:
            timesteps, past_scores = zip(*self.window)
            self._learn(_slopes(timesteps, past_scores))


class SamplingTeacher(Teacher):
    """Algorithm 8 (inspired by Thompson sampling): keep the last K rewards per task, draw one per task,
    put (1 - epsilon) mass on the task with the largest |sampled reward| and mix in epsilon-uniform."""

    def __init__(self, num_tasks, bucket_sizes, seed=0, K=10, epsilon=0.1, **kwargs):
        super().__init__(num_tasks, bucket_sizes, seed, **kwargs)
        self.epsilon = epsilon
        self.D = [deque(maxlen=K) for _ in range(num_tasks)]   # recent rewards per task
        self.previous_scores = None

    def distribution(self):
        # A task with no rewards yet gets sampled reward 1, so it gets tried (as in the paper).
        sampled_rewards = np.array([self.random_generator.choice(list(rewards)) if rewards else 1.0
                                    for rewards in self.D])
        return _epsilon_greedy(np.abs(sampled_rewards), self.epsilon, self.random_generator)

    def update(self, scores):
        scores = np.asarray(scores, dtype=np.float64)
        if self.previous_scores is not None:
            for rewards, reward in zip(self.D, scores - self.previous_scores):
                rewards.append(reward)
        self.previous_scores = scores


TEACHERS = {
    "proportional": ProportionalTeacher,
    "uniform": UniformTeacher,
    "online": OnlineTeacher,
    "naive": NaiveTeacher,
    "window": WindowTeacher,
    "sampling": SamplingTeacher,
}


def make_teacher(name, num_tasks, bucket_sizes, seed=0, **kwargs) -> Teacher:
    return TEACHERS[name](num_tasks, bucket_sizes, seed=seed, **kwargs)
