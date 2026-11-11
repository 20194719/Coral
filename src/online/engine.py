"""Influence-guided pair verification on acyclic join graphs."""

from dataclasses import dataclass
import numpy as np
from src.online.refinement import Feedback
from src.online.sampling import STRATA, quantile_strata


@dataclass
class Result:
    ce0: float
    ce: float
    calls: int
    sd: float
    refinement: dict


class TreeEngine:
    def __init__(
        self,
        baselines,
        edges,
        filters,
        groups=5,
        prior_strength=1.0,
        *,
        join_prior_strength=None,
        filter_prior_strength=None,
        join_calibration=True,
    ):
        """Initialize model state."""
        self.join_calibration = bool(join_calibration)
        join_strength = (
            prior_strength if join_prior_strength is None else join_prior_strength
        )
        filter_strength = (
            prior_strength if filter_prior_strength is None else filter_prior_strength
        )
        self.edges = list(edges)
        self.sizes = [0] * len(filters)
        self.adj = [[] for _ in filters]
        for e, ((i, j), base) in enumerate(zip(edges, baselines)):
            for node, size in zip((i, j), base.pi.shape):
                if self.sizes[node] not in (0, size):
                    raise ValueError("Inconsistent collection sizes")
                self.sizes[node] = size
            self.adj[i].append((j, e))
            self.adj[j].append((i, e))
        if (
            len(baselines) != len(edges)
            or len(edges) != len(filters) - 1
            or not all(self.sizes)
        ):
            raise ValueError("Expected one connected acyclic join graph")
        visited = set()

        def visit(node, parent):
            """Traverse the join tree."""
            if node in visited:
                raise ValueError("Join graph contains a cycle")
            visited.add(node)
            for other, _ in self.adj[node]:
                if other != parent:
                    visit(other, node)

        visit(0, -1)
        if len(visited) != len(filters):
            raise ValueError("Disconnected join graph")
        self.bands, self.filters = [], []
        for h, size in zip(filters, self.sizes):
            band = (
                quantile_strata(h, groups)
                if h is not None
                else np.zeros(size, dtype=np.int32)
            )
            self.bands.append(band)
            self.filters.append(
                Feedback(h, band, filter_strength) if h is not None else None
            )
        self.pairs, self.regions = [], []
        for (i, j), base in zip(edges, baselines):
            region = (
                self.bands[i][:, None] * (int(self.bands[j].max()) + 1)
                + self.bands[j][None, :]
            )
            self.regions.append(region)
            self.pairs.append(
                Feedback(base.pi, region * len(STRATA) + base.strata, join_strength)
            )
        self.rounds = []

    def probabilities(self):
        """Probabilities."""
        return [
            f.values if f is not None else np.ones(n)
            for f, n in zip(self.filters, self.sizes)
        ]

    def composition(self):
        """Composition."""
        h = self.probabilities()
        messages = {}

        def message(source, target, edge):
            """Compute a join tree message."""
            key = (source, target)
            if key not in messages:
                product = h[source].copy()
                for other, e in self.adj[source]:
                    if other != target:
                        product *= message(other, source, e)
                pi = self.pairs[edge].values
                messages[key] = (
                    product @ pi if self.edges[edge][0] == source else pi @ product
                )
            return messages[key]

        influence = []
        for node in range(len(h)):
            product = np.ones(self.sizes[node])
            for other, e in self.adj[node]:
                product *= message(other, node, e)
            influence.append(product)
        pair_influence = []
        for i, j in self.edges:
            left, right = h[i].copy(), h[j].copy()
            for other, e in self.adj[i]:
                if other != j:
                    left *= message(other, i, e)
            for other, e in self.adj[j]:
                if other != i:
                    right *= message(other, j, e)
            pair_influence.append((left, right))
        return float(h[0] @ influence[0]), influence, pair_influence

    def choose_region(self, influence, pair_influence):
        """Choose region."""
        filter_gains = [
            f.priorities(w)[0] if f is not None else None
            for f, w in zip(self.filters, influence)
        ]
        best = None
        for e, ((i, j), pair, region, (left, right)) in enumerate(
            zip(self.edges, self.pairs, self.regions, pair_influence)
        ):
            gain, _ = pair.priorities(left[:, None] * right[None, :])
            priority = np.where(pair.known, 0.0, gain[pair.groups])
            eligible = ~pair.known.copy()
            for node, axis in ((i, 0), (j, 1)):
                state = self.filters[node]
                if state is None:
                    continue
                unknown = ~state.known
                value = np.where(unknown, filter_gains[node][state.groups], 0.0)
                priority += value[:, None] if axis == 0 else value[None, :]
                eligible |= unknown[:, None] if axis == 0 else unknown[None, :]
            counts = np.bincount(
                region.ravel(),
                weights=eligible.ravel(),
                minlength=int(region.max()) + 1,
            )
            totals = np.bincount(
                region.ravel(),
                weights=np.where(eligible, priority, 0.0).ravel(),
                minlength=len(counts),
            )
            scores = np.divide(
                totals, counts, out=np.full(len(counts), -np.inf), where=counts > 0
            )
            chosen = int(np.argmax(scores))
            if counts[chosen] and (best is None or scores[chosen] > best[0]):
                best = (
                    float(scores[chosen]),
                    e,
                    chosen,
                    np.flatnonzero(eligible & (region == chosen)),
                )
        return best

    def run(self, probes, budget, seed=0, batch=None):
        """Run adaptive query refinement."""
        if len(probes) != len(self.edges) or budget < 0:
            raise ValueError("Expected one judge per edge and a nonnegative budget")
        rng = np.random.default_rng(1000 + seed)
        batch = batch or max(10, budget // 8)
        initial = self.composition()[0]
        spent = 0
        while spent < budget:
            _, influence, pair_influence = self.composition()
            chosen = self.choose_region(influence, pair_influence)
            if chosen is None:
                break
            gain, e, region, eligible = chosen
            flat = rng.choice(
                eligible, size=min(batch, budget - spent, len(eligible)), replace=False
            )
            i, j = self.edges[e]
            width = self.sizes[j]
            pairs = [(int(k // width), int(k % width)) for k in flat]
            requested, pending = [], [set(), set()]
            for index, (r, s) in zip(flat, pairs):
                need = [not bool(self.pairs[e].known.flat[index])]
                for axis, (node, doc) in enumerate(((i, r), (j, s))):
                    state = self.filters[node]
                    missing = (
                        state is not None
                        and not state.known[doc]
                        and doc not in pending[axis]
                    )
                    need.append(bool(missing))
                    if missing:
                        pending[axis].add(doc)
                requested.append(tuple(need))
            answers = probes[e](pairs, requested)
            if len(answers) != len(pairs):
                raise ValueError("Judge returned the wrong number of answers")
            for index, (r, s), need, answer in zip(flat, pairs, requested, answers):
                if len(answer) != 3 or any(
                    flag and y not in (0, 1) for flag, y in zip(need, answer)
                ):
                    raise ValueError("Judge must supply every requested binary label")
                if need[0]:
                    self.pairs[e].observe(index, answer[0])
                for node, doc, flag, label in zip((i, j), (r, s), need[1:], answer[1:]):
                    if flag:
                        self.filters[node].observe(doc, label)
            if self.join_calibration:
                self.pairs[e].recompute()
            for node in (i, j):
                if self.filters[node] is not None:
                    self.filters[node].recompute()
            spent += len(pairs)
            self.rounds.append(
                dict(
                    edge=e,
                    region=region,
                    priority=gain,
                    pairs=pairs,
                    requested=requested,
                    answers=answers,
                    estimate=self.composition()[0],
                )
            )
        estimate, influence, pair_influence = self.composition()
        variance = sum(
            float(f.priorities(w)[1].sum())
            for f, w in zip(self.filters, influence)
            if f is not None
        )
        variance += sum(
            float(p.priorities(a[:, None] * b[None, :])[1].sum())
            for p, (a, b) in zip(self.pairs, pair_influence)
        )
        pair_states = []
        for pair in self.pairs:
            state = pair.state_dict()
            state["calibration_applied"] = self.join_calibration
            if not self.join_calibration:
                state["group_offsets"] = [0.0] * pair.size
            pair_states.append(state)
        return Result(
            initial,
            estimate,
            spent,
            float(np.sqrt(variance)),
            dict(
                policy="additive-distinct-labels",
                join_calibration=self.join_calibration,
                join_probabilities_unchanged=all(
                    np.array_equal(p.values, p.initial) for p in self.pairs
                ),
                edges=self.edges,
                pairs=pair_states,
                filters=[
                    f.state_dict() if f is not None else None for f in self.filters
                ],
                rounds=self.rounds,
            ),
        )
