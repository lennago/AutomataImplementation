import numpy as np
import math
import statistics
import random
from fractions import Fraction
from joblib import Parallel, delayed
import os
import multiprocessing
import time
from nfa import DAG, NFA


class MainFPRAS:

    def __init__(self, nfa: NFA, n: int, epsilon: float):
        self.dag = DAG(nfa, n)
        if self.dag.m:
            self.empty = False
            if len(self.dag.start_states) != 1:
                raise ValueError("DAG must have exactly one start state.")
            if len(self.dag.accept_states) != 1:
                raise ValueError("DAG must have exactly one accept state.")
        else:
            self.empty = True
        self.BATCH_SIZE = max(multiprocessing.cpu_count(), 10000000 // (2**self.dag.m))
        self.epsilon = epsilon
        self.n = n
        self.k = math.ceil((self.n * self.dag.m) / self.epsilon)
        self.num_of_words = 2 * self.k**7
        self.np_alpha = np.empty(self.n + 1, dtype=object)

    def sample(
        self, layer: int, states: np.ndarray, cur_word: np.ndarray, phi: Fraction
    ) -> np.ndarray | None:
        """
        Samples a word from the DAG based on the current layer, states, and phi value.
        Returns the sampled word or None if the sampling fails.
        :param layer: Current layer in the DAG.
        :param states: Current states in the DAG.
        :param cur_word: Current word being constructed.
        :param phi: Probability threshold for sampling.
        """
        if layer == 0:
            if random.random() <= phi:
                return cur_word
            return None
        possible_predecessors = self.dag.states[layer - 1].nonzero()[0]
        set_p_b = [
            np.logical_and(
                self.dag.states[layer - 1],
                states.dot(self.dag.transition_matrices_backwards[b]),
            ).nonzero()[0]
            for b in range(2)
        ]
        np_0 = self.np_alpha[layer - 1][
            sum(
                [
                    2 ** np.where(possible_predecessors == state_index)[0][0]
                    for state_index in set_p_b[0]
                ]
            )
        ]
        np_1 = self.np_alpha[layer - 1][
            sum(
                [
                    2 ** np.where(possible_predecessors == state_index)[0][0]
                    for state_index in set_p_b[1]
                ]
            )
        ]
        p_0 = np_0 / (np_0 + np_1)
        p_b = [p_0, 1 - p_0]
        if random.random() <= np_0 / (np_0 + np_1):
            b = 0
        else:
            b = 1
        cur_word[layer - 1] = b
        return self.sample(
            layer - 1,
            p_b[b],
            cur_word,
            phi / p_b[b],
        )

    def run(self):
        """
        Runs the FPRAS on the given input string and returns whether it accepts or rejects.
        """
        # Implement the logic for running the FPRAS
        if self.empty:
            return 0
        self.np_alpha = np.empty(self.n + 1, dtype=object)
        self.np_alpha[0] = np.array([Fraction(0), Fraction(1)], dtype=object)
        for layer in range(1, self.n):
            layer_states = self.dag.states[layer].nonzero()[0]
            possible_predecessors = self.dag.states[layer - 1].nonzero()[0]
            self.np_alpha[layer] = np.zeros(2 ** len(layer_states), dtype=object)
            for i, state in enumerate(layer_states):
                state_vector = np.zeros(self.dag.m, dtype=bool)
                state_vector[state] = 1
                r_b = [
                    np.logical_and(
                        self.dag.states[layer - 1],
                        state_vector.dot(self.dag.transition_matrices_backwards[b]),
                    ).nonzero()[0]
                    for b in range(2)
                ]
                self.np_alpha[layer][2**i] = (
                    self.np_alpha[layer - 1][
                        sum(
                            [
                                2
                                ** np.where(possible_predecessors == state_index)[0][0]
                                for state_index in r_b[0]
                            ]
                        ),
                    ]
                    + self.np_alpha[layer - 1][
                        sum(
                            [
                                2
                                ** np.where(possible_predecessors == state_index)[0][0]
                                for state_index in r_b[1]
                            ]
                        ),
                    ]
                )
                length_s_state_powerset = np.zeros(
                    (2 ** len(layer_states)), dtype=object
                )
                mask_length = 2 ** len(layer_states)
                start_phi = Fraction(self.epsilon**-5) / self.np_alpha[layer][2**i]
                for batch_start in range(0, self.num_of_words, self.BATCH_SIZE):
                    batch_end = min(batch_start + self.BATCH_SIZE, self.num_of_words)
                    words = Parallel(
                        n_jobs=-1,
                        batch_size="auto",
                    )(
                        delayed(self.sample)(
                            layer,
                            state_vector,
                            np.zeros(layer, dtype=bool),
                            start_phi,
                        )
                        for _ in range(batch_start, batch_end)
                    )
                    if None in words:
                        return 0
                    mask_i_in_powerset = np.zeros(mask_length, dtype=bool)
                    indices_i_in_powerset = (
                        [
                            2**i + k + j * 2 ** (i + 1)
                            for k in range(2 ** (i - 1))
                            for j in range(2 ** (len(layer_states) - (i + 1)))
                        ]
                        if i
                        else [
                            2**i + j * 2 ** (i + 1)
                            for j in range(2 ** (len(layer_states) - (i + 1)))
                        ]
                    )
                    mask_i_in_powerset[indices_i_in_powerset] = True
                    mask_i_in_powerset[2**i] = False  # 2**i is the {i} calculated above
                    if i:
                        reached_states = self.dag.simulate_batch(
                            np.array(words, dtype=bool)
                        )
                        columns_before_i = reached_states[:, : i - 1]
                        mask_i_lowest_in_order = np.all(columns_before_i == 0, axis=1)
                        count = np.sum(mask_i_lowest_in_order)
                    else:
                        count = len(words)
                    length_s_state_powerset[mask_i_in_powerset] += count
                    print(
                        f"Layer: {layer}, State: {i}, Words processed: {batch_start} - {batch_end} of {self.num_of_words}"
                    )
                self.np_alpha[layer] += (
                    length_s_state_powerset
                    * self.np_alpha[layer][2**i]
                    / self.num_of_words
                )
            for index, value in enumerate(self.np_alpha[layer]):
                self.np_alpha[layer][index] = int(value)
        self.np_alpha[self.n] = np.array([Fraction(0), Fraction(0)], dtype=object)
        r_b = [
            np.logical_and(
                self.dag.states[self.n - 1],
                state_vector.dot(self.dag.transition_matrices_backwards[b]),
            ).nonzero()[0]
            for b in range(2)
        ]
        self.np_alpha[self.n][1] = (
            self.np_alpha[self.n - 1][
                sum(
                    [
                        2 ** np.where(possible_predecessors == state_index)[0][0]
                        for state_index in r_b[0]
                    ]
                ),
            ]
            + self.np_alpha[self.n - 1][
                sum(
                    [
                        2 ** np.where(possible_predecessors == state_index)[0][0]
                        for state_index in r_b[1]
                    ]
                ),
            ]
        )
        return self.np_alpha[self.n][1]


class FasterFPRAS:
    def __init__(self, nfa: NFA, n: int, epsilon: float, delta: float):
        self.dag = DAG(nfa, n)
        if self.dag.m:
            self.empty = False
            if len(self.dag.start_states) != 1:
                raise ValueError("DAG must have exactly one start state.")
            if len(self.dag.accept_states) != 1:
                raise ValueError("DAG must have exactly one accept state.")
        else:
            self.empty = True
        self.n = n
        self.beta = Fraction(epsilon / (4 * self.n**2))
        self.eta = Fraction(delta / (2 * self.n * self.dag.m))
        self.ns = Fraction(
            ((4096 * math.e * self.n**4) / epsilon**2)
            * math.log2(
                (4096 * self.dag.m**2 * self.n**2 * math.log2(1 / epsilon**2)) / delta
            )
        )
        self.xns = (
            self.ns * 12 * 1 / (1 - (2 / (3 * math.e**2))) * math.log2(8 / self.eta)
        )

    def app_union(
        self,
        epsilon_sz: Fraction,
        states: np.ndarray,
        delta: float,
        epsilon: float = None,
    ) -> int:
        """
        Approximates the union of the states in the NFA with a given epsilon size.
        :param epsilon_sz:
        :param states: Current states in the DAG.
        :param delta: Delta value for the approximation.
        :param epsilon: Epsilon value for the approximation, defaults to self.beta.
        :return: Approximated size of the union of set of states.
        """
        if epsilon is None:
            epsilon = self.beta
        m = math.ceil(1)
        return 0

    def sample(
        self,
        layer: int,
        states: np.ndarray,
        cur_word: np.ndarray,
        phi: float,
        beta: float,
        eta: float,
    ) -> np.ndarray | None:
        """
        Samples a word from the DAG based on the current layer, states, and phi value.
        Returns the sampled word or None if the sampling fails.
        :param layer: Current layer in the DAG.
        :param states: Current states in the DAG.
        :param cur_word: Current word being constructed.
        :param phi: Probability threshold for sampling.
        :param beta: Beta value for the sampling process.
        :param eta: Eta value for the sampling process.
        """
        return None

    def run(self):
        """
        Runs the Faster FPRAS on the given input string and returns whether it accepts or rejects.
        """
        if self.empty:
            return 0

        for l in range(1, self.n + 1):
            beta_acute = (1 + self.beta) ** (l - 1) - 1
            for q in range(self.dag.m):
                pass
        return 0


class DependentFPRAS:
    def __init__(self, nfa: NFA, n: int, epsilon: float, delta: float):
        self.dag = DAG(nfa, n)
        if not self.dag.m:
            self.empty = True
        else:
            self.empty = False
            k = epsilon / (1 + epsilon)
            self.n = n
            self.n_s = math.ceil(4 * (n + 1) * (((1 + k) ** 2) / (k**2 * (1 - k))))
            self.n_t = math.ceil(8 * math.log(16 * n * self.dag.m))
            self.n_u = math.ceil(8 * math.log(1 / delta))
            self.theta = 16 * self.n_s * self.n_t * n * (1 + k) * self.dag.m
            self.ns_nt = self.n_s * self.n_t
            self.cache_i = np.zeros(self.dag.m, dtype=bool)
            self.s_r = np.zeros((3, self.dag.m, self.ns_nt, 1), dtype=bool)
            self.p = np.zeros((2, self.dag.m), dtype=Fraction)

    def run(self):
        """
        Runs the Dependent FPRAS on the given input string and returns whether it accepts or rejects.
        """
        if self.empty:
            return 0
        # Implement the logic for running the Dependent FPRAS
        est = Parallel(n_jobs=-1)(
            delayed(self.count_nfa_core)() for _ in range(self.n_u)
        )
        return statistics.median(est)

    def compute_cache(self, i: int):
        """
        Computes the cache for the NFA.
        This method should implement the logic to compute the cache for the NFA.
        """
        # Placeholder for actual cache computation logic
        if 0 > i > self.dag.n:
            raise ValueError(f"Index {i} out of bounds for NFA with n={self.dag.n}.")
        if i == 0:
            self.cache_i = np.zeros((1, self.dag.m), dtype=bool)
            self.cache_i[0, self.dag.start_states[0]] = True
        elif i == self.dag.n:
            return
        else:
            states = self.dag.states[i]
            mask_0 = np.logical_or.reduce(
                np.logical_or.reduce(self.s_r[0, states, :], axis=1, dtype=bool),
                axis=0,
                dtype=bool,
            )
            mask_1 = np.logical_or.reduce(
                np.logical_or.reduce(self.s_r[1, states, :], axis=0, dtype=bool),
                axis=0,
                dtype=bool,
            )
            self.cache_i = np.append(
                self.dag.simulate_one(self.cache_i[mask_0], 0, i - 1),
                self.dag.simulate_one(self.cache_i[mask_1], 1, i - 1),
                axis=0,
            )
            s_r_new = np.zeros(
                (3, self.dag.m, self.ns_nt, self.cache_i.shape[0]), dtype=bool
            )
            index_0 = np.zeros(self.cache_i.shape[0], dtype=bool)
            index_1 = np.zeros(self.cache_i.shape[0], dtype=bool)
            index_0[: np.sum(mask_0)] = True
            index_1[np.sum(mask_0) :] = True
            s_r_new[2, :, :, index_0] = self.s_r[0, :, :, mask_0]
            s_r_new[2, :, :, index_1] = self.s_r[1, :, :, mask_1]
            self.s_r = s_r_new

    def union(self, q: int, s: np.ndarray, pred_states: list) -> np.ndarray:
        """
        Computes the union of two states in the NFA.
        This method should implement the logic to compute the union of two states.
        :param q: The first state in the NFA.
        :param layer: The layer of the state.
        :param s: The set from which to compute the union.
        :return: The set s_acute which is the union of set s for state q.
        """
        s_acute = np.zeros((2, self.ns_nt, self.cache_i.shape[0]), dtype=bool)
        for b in range(2):
            b_pred = pred_states[b]
            s_acute_b = np.zeros((self.ns_nt, self.cache_i.shape[0]), dtype=bool)
            for index, j in enumerate(b_pred):
                l = b_pred[:index]
                j_first = np.zeros(self.cache_i.shape[0], dtype=bool)
                j_first[np.where(~self.cache_i[:, l].any(axis=1))[0]] = True
                s_acute_b = np.logical_or(s_acute_b, np.logical_and(s[j], j_first))
            s_acute[b] = s_acute_b
        return s_acute

    def reduce(self, s: np.ndarray, p: Fraction) -> np.ndarray:
        """
        Reduces the set s based on the probability p.
        This method should implement the logic to reduce the set s.
        :param s: The set to be reduced.
        :param p: The probability threshold for reduction.
        :return: The reduced set.
        """
        return np.logical_and(s, np.random.rand(*s.shape) < p)

    def estimate_and_sample(self, q: int, layer: int):
        """
        Estimates and samples the NFA for a given state q.
        This method should implement the logic to estimate and sample the NFA.
        :param q: The state in the NFA to estimate and sample.
        """
        state_vector = np.zeros(self.dag.m, dtype=bool)
        state_vector[q] = True
        pred_states = [
            np.logical_and(
                self.dag.states[layer - 1],
                state_vector.dot(self.dag.transition_matrices_backwards[b]),
            )
            for b in range(2)
        ]
        pred_states_vector = np.logical_or(pred_states[0], pred_states[1])
        p_pred_states = self.p[0][pred_states_vector]
        pred_states = [np.nonzero(pred_states[b])[0] for b in range(2)]
        all_pred_states = np.nonzero(pred_states_vector)[0]
        p_q = np.min(p_pred_states)
        s_r_dach = np.zeros((2, self.ns_nt, self.cache_i.shape[0]), dtype=bool)
        m_j = np.zeros((self.n_t), dtype=Fraction)
        s_r_quer = np.zeros((self.dag.m, self.ns_nt, self.cache_i.shape[0]), dtype=bool)
        for q_pred in all_pred_states:
            s_r_quer[q_pred] = self.reduce(
                np.logical_or(
                    self.s_r[2, q_pred], np.zeros((self.ns_nt, self.cache_i.shape[0]))
                ),
                p_q / self.p[0, q_pred],
            )
        s_r_dach = self.union(q, s_r_quer, pred_states)
        factor = self.n_s * p_q
        factor = Fraction(factor.denominator, factor.numerator)
        m_j = factor * s_r_dach.reshape(
            (2, self.n_t, self.n_s, self.cache_i.shape[0])
        ).sum(axis=(0, 2, 3))
        median_m_j = np.median(m_j)
        self.p[1, q] = min(p_q, Fraction(median_m_j.denominator, median_m_j.numerator))
        self.s_r[:2, q] = self.reduce(s_r_dach, self.p[1, q] / p_q)

    def count_nfa_core(self) -> Fraction:
        """
        Core logic for counting the NFA.
        This method should implement the core counting logic for the NFA.
        """
        self.p = np.zeros((2, self.dag.m), dtype=Fraction)
        self.p[1, self.dag.start_states[0]] = Fraction(1)
        self.compute_cache(0)
        start_state_vector = np.zeros(self.dag.m, dtype=bool)
        start_state_vector[self.dag.start_states[0]] = True
        self.s_r = np.zeros((3, self.dag.m, self.ns_nt, 1), dtype=bool)
        self.s_r[2, self.dag.start_states[0], :] = 1
        sample_count = 0
        for i in range(1, self.n + 1):
            self.p[0] = self.p[1].copy()
            self.p[1] = np.zeros(self.dag.m, dtype=Fraction)
            layer_states = self.dag.states[i].nonzero()[0]
            print(f"Layer {i}, States: {layer_states}")
            for q in layer_states:
                self.estimate_and_sample(q, i)
                sample_count += np.sum(self.s_r[:2, q])
                if sample_count >= self.theta:
                    return 0
            self.compute_cache(i)
        return Fraction(
            numerator=self.p[1, self.dag.accept_states[0]].denominator,
            denominator=self.p[1, self.dag.accept_states[0]].numerator,
        )


class Stupid_Parallel:
    def __init__(self, nfa: NFA, n: int):
        self._dag = DAG(nfa, n)
        self._alphabet = [0, 1]
        self._n = n
        self._words = []

    def binary_generator(self, n):
        """Generator version for large n"""
        for i in range(2**n):
            binary_str = format(i, f"0{n}b")
            yield np.pad(
                np.array([c == "1" for c in binary_str], dtype=bool),
                (0, self._n - n),
                mode="constant",
                constant_values=False,
            )

    def run(self):
        """
        Runs the Stupid algorithm on the given input string and
        returns whether it accepts or rejects.
        """
        # Implement the logic for running the Stupid algorithm
        process_count = min(
            2 ** int(math.log2(multiprocessing.cpu_count())), 2**self._n, 2**64
        )
        start_n = int(math.log2(process_count))
        counts = Parallel(n_jobs=process_count)(
            delayed(self.dfs)(start_str, start_n)
            for start_str in self.binary_generator(start_n)
        )
        return sum(counts)

    def dfs(self, cur_str: np.ndarray, layer: int):
        """
        Depth-first search to traverse the DAG and count the number of accepted strings.
        """
        if layer == self._n:
            if self._dag.is_accepted(cur_str):
                return 1
            return 0
        count = 0
        for symbol in self._alphabet:
            next_str = cur_str.copy()
            next_str[layer] = symbol
            count += self.dfs(next_str, layer + 1)
        return count


class Stupid:
    def __init__(self, nfa: NFA, n: int):
        self._dag = DAG(nfa, n)
        self._alphabet = [0, 1]
        self._n = n
        self._count = 0
        self._words = []

    def run(self):
        """
        Runs the Stupid algorithm on the given input string and
        returns whether it accepts or rejects.
        """
        # Implement the logic for running the Stupid algorithm
        self._count = 0
        self._words = []
        self.dfs(np.zeros(self._n, dtype=bool), 0)
        return self._count

    def dfs(self, cur_str: np.ndarray, layer: int):
        """
        Depth-first search to traverse the DAG and count the number of accepted strings.
        """
        if layer == self._n:
            if self._dag.is_accepted(cur_str):
                self._count += 1
                self._words.append(cur_str.copy())
            return

        for symbol in self._alphabet:
            next_str = cur_str.copy()
            next_str[layer] = symbol
            self.dfs(next_str, layer + 1)


if __name__ == "__main__":
    pass
