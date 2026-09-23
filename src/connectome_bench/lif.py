"""CPU reference. Uniform point neurons; model assumptions are intentionally explicit."""

from dataclasses import dataclass
import math
import numpy as np

from .graph import Graph


@dataclass(frozen=True)
class LIFParameters:
    dt_ms: float = 1.0
    tau_mem_ms: float = 20.0
    tau_syn_ms: float = 5.0
    rest_mv: float = 0.0
    threshold_mv: float = 1.0
    reset_mv: float = 0.0
    refractory_ms: float = 2.0
    delay_ms: float = 1.0

    def __post_init__(self):
        if min(self.dt_ms, self.tau_mem_ms, self.tau_syn_ms, self.delay_ms) <= 0:
            raise ValueError("dt, time constants and delay must be positive")
        if self.refractory_ms < 0 or self.threshold_mv <= self.reset_mv:
            raise ValueError("Invalid threshold/reset/refractory")
        if not all(math.isfinite(v) for v in vars(self).values()):
            raise ValueError("All parameters must be finite")
        if not math.isclose(self.delay_ms / self.dt_ms,
                            round(self.delay_ms / self.dt_ms), abs_tol=1e-9):
            raise ValueError("delay_ms must be an integer number of dt_ms steps")


class LIF:
    """One step: arrivals, g decay, V update, threshold, schedule future arrivals.

    Refractory neurons discard incoming events; the last spike's g is reset.
    `drive_mv` is a sustained voltage-equivalent input, not a pulse each tick.
    Connectivity is W[post,pre]. State must persist across calls/frames.
    """

    def __init__(self, graph: Graph, params: LIFParameters = LIFParameters()):
        self.graph, self.params = graph, params
        self.outgoing = graph.weights.tocsc()
        self.delay_steps = round(params.delay_ms / params.dt_ms)
        self.refractory_steps = math.ceil(params.refractory_ms / params.dt_ms)
        self.a_v = math.exp(-params.dt_ms / params.tau_mem_ms)
        self.a_g = math.exp(-params.dt_ms / params.tau_syn_ms)
        self.reset()

    def reset(self):
        size = len(self.graph.ids)
        self.tick = 0
        self.v = np.full(size, self.params.rest_mv, dtype=np.float64)
        self.g = np.zeros(size, dtype=np.float64)
        self.refractory_until = np.zeros(size, dtype=np.int64)
        self.ring = np.zeros((self.delay_steps + 1, size), dtype=np.float64)

    def step(self, drive_mv=None) -> np.ndarray:
        size = len(self.v)
        drive = (np.zeros(size, dtype=np.float64) if drive_mv is None
                 else np.asarray(drive_mv, dtype=np.float64))
        if drive.shape != (size,) or not np.all(np.isfinite(drive)):
            raise ValueError("drive_mv must be a finite vector of shape (N,)")
        slot = self.tick % len(self.ring)
        eligible = self.tick >= self.refractory_until
        self.g[eligible] = self.a_g * self.g[eligible] + self.ring[slot, eligible]
        self.ring[slot].fill(0)
        self.v[eligible] = (self.params.rest_mv
                            + self.a_v * (self.v[eligible] - self.params.rest_mv)
                            + (1 - self.a_v) * (drive[eligible] + self.g[eligible]))
        spike = eligible & (self.v >= self.params.threshold_mv)
        if np.any(spike):
            active = np.flatnonzero(spike)
            future_slot = (self.tick + self.delay_steps) % len(self.ring)
            for src in active:
                start, stop = self.outgoing.indptr[src:src+2]
                np.add.at(self.ring[future_slot], self.outgoing.indices[start:stop],
                          self.outgoing.data[start:stop])
            self.v[spike] = self.params.reset_mv
            self.g[spike] = 0
            self.refractory_until[spike] = self.tick + self.refractory_steps + 1
        self.tick += 1
        return spike

    def snapshot(self) -> dict:
        return {"tick": self.tick, "v": self.v.copy(), "g": self.g.copy(),
                "refractory_until": self.refractory_until.copy(), "ring": self.ring.copy()}

    def restore(self, state: dict):
        if not isinstance(state["tick"], int) or state["tick"] < 0:
            raise ValueError("Invalid tick")
        for name in ("v", "g", "refractory_until", "ring"):
            if np.shape(state[name]) != np.shape(getattr(self, name)):
                raise ValueError(f"State shape mismatch for {name}")
        self.tick = state["tick"]
        for name in ("v", "g", "refractory_until", "ring"):
            setattr(self, name, np.asarray(state[name]).copy())


class RateReadout:
    """Exponential rate per neuron in Hz; a task may map its selected IDs to actions."""

    def __init__(self, size: int, dt_ms: float, tau_ms: float = 100.0):
        if size < 1 or min(dt_ms, tau_ms) <= 0:
            raise ValueError("Invalid rate filter size or time")
        self.hz = np.zeros(size, dtype=np.float64)
        self.decay = math.exp(-dt_ms / tau_ms)
        self.dt_ms = dt_ms

    def update(self, spikes: np.ndarray) -> np.ndarray:
        spikes = np.asarray(spikes)
        if spikes.shape != self.hz.shape:
            raise ValueError("Spike count does not match filter")
        self.hz = self.decay * self.hz + (1-self.decay) * spikes * (1000/self.dt_ms)
        return self.hz.copy()


def signed_turn(readout: RateReadout, left_index: int, right_index: int,
                gain: float = 1.0, limit: float = 1.0) -> float:
    """An engineered action interface; no natural motor semantics assumed."""
    return float(np.clip(gain * (readout.hz[right_index] - readout.hz[left_index]),
                         -limit, limit))
