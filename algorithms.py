"""Module containing the FPRAS algorithms and Bruteforce algorithm to approximate L^n(NFA)."""

import math
import statistics
import random
from fractions import Fraction
import multiprocessing
from multiprocessing import Process, Queue, shared_memory
from tqdm.auto import tqdm
import numpy as np
from joblib import Parallel, delayed
import psutil
from nfa import DAG


class MainFPRAS:
    """
    FPRAS from Arenas et al., "#NFA admits an FPRAS: Efficient Enumeration, Counting,
    and Uniform Generation for Logspace Classes"
    """

    def __init__(
        self,
        dag: DAG,
        n: int,
        epsilon: Fraction,
        delta: Fraction = Fraction(1 / math.e),
        debug: str = "None",
        progress_bar: bool = True,
    ):
        self.dag = dag
        if self.dag.m:
            self.empty = False
            if len(self.dag.start_states) != 1:
                raise ValueError("DAG must have exactly one start state.")
            if len(self.dag.accept_states) != 1:
                raise ValueError("DAG must have exactly one accept state.")
        else:
            self.empty = True
        self.epsilon = epsilon
        self.n = n
        self.k = math.ceil((self.n * self.dag.m) / self.epsilon * math.log(1 / delta))
        self.max_tries = math.ceil(
            (2 + math.log(4) + (8 * math.log(self.k)))
            / math.log((1 - math.e**-9) ** -1)
        )
        self.debug = debug
        self.progress_bar = progress_bar
        if self.debug != "None":
            tqdm.write(f"Max tries: {self.max_tries}, k: {self.k}")
        self.num_of_words = 2 * self.k**7
        self.np_alpha = np.empty(self.n + 1, dtype=object)
        self.available_processes = multiprocessing.cpu_count() - 1

    def worker(
        self,
        queue: Queue,
        i: int,
        batch_start: int,
        batch_end: int,
        layer: int,
        state_vector: np.ndarray,
        start_phi: Fraction,
        layer_states: np.ndarray,
        batch: int,
        total_batches: int,
        total_layers: int,
        progress_bar: bool = True,
    ):
        """
        Worker function to sample batch of words in a new process,
        to free memory after processing.
        Puts the results in the queue for the main process to collect.
        :param queue: Queue to collect results.
        :param i: Current index in the layer.
        :param batch_start: Start index for the batch of words.
        :param batch_end: End index for the batch of words.
        :param layer: Current layer in the DAG.
        :param state_vector: State vector for the current layer.
        :param start_phi: Starting phi value for sampling.
        :param mask_length: Length of the mask for the powerset.
        :param layer_states: States in the current layer.
        :param batch: Current batch number.
        """

        def sample(
            dag: DAG,
            np_alpha: np.ndarray,
            layer: int,
            states: np.ndarray,
            phi: Fraction,
            max_tries: int,
        ) -> np.ndarray | None:
            """
            Samples a word from the DAG based on the current layer, states, and phi value.
            Returns the sampled word or None if the sampling fails.
            :param dag: The DAG to sample from.
            :param np_alpha: Numpy array containing the alpha values for each layer.
            :param layer: Current layer in the DAG.
            :param states: Current states in the DAG.
            :param phi: Probability threshold for sampling
            :param max_tries: Maximum number of tries for sampling.
            """
            relevant_states_per_layer = dag.states[:layer]
            possible_predecessors = [
                relevant_states_per_layer[l].nonzero()[0] for l in range(layer)
            ]
            for _ in range(int(max_tries)):
                cur_phi = Fraction(phi)
                cur_states = states.copy()
                word = np.empty(layer, dtype=np.uint8)
                for l in range(layer):
                    prev_layer_index = -(l + 1)
                    set_p_0 = np.logical_and(
                        relevant_states_per_layer[prev_layer_index],
                        cur_states @ dag.transition_matrices_backwards[0],
                    )
                    set_p_1 = np.logical_and(
                        relevant_states_per_layer[prev_layer_index],
                        cur_states @ dag.transition_matrices_backwards[1],
                    )
                    index_p_0 = set_p_0.nonzero()[0]
                    index_p_1 = set_p_1.nonzero()[0]
                    np_0 = np_alpha[prev_layer_index][
                        tuple(
                            [
                                1 if idx in index_p_0 else 0
                                for idx in possible_predecessors[prev_layer_index]
                            ]
                        )
                    ]
                    np_1 = np_alpha[prev_layer_index][
                        tuple(
                            [
                                1 if idx in index_p_1 else 0
                                for idx in possible_predecessors[prev_layer_index]
                            ]
                        )
                    ]
                    p_0 = np_0 / (np_0 + np_1)
                    if random.random() <= np_0 / (np_0 + np_1):
                        cur_states = set_p_0
                        cur_phi = Fraction(cur_phi, p_0)
                        word[prev_layer_index] = 0
                    else:
                        cur_states = set_p_1
                        cur_phi = Fraction(cur_phi, 1 - p_0)
                        word[prev_layer_index] = 1
                if random.random() <= cur_phi:
                    return word
            return None

        def sample_one(
            dag: DAG,
            np_alpha: np.ndarray,
            layer: int,
            states: np.ndarray,
            phi: Fraction,
            max_tries: int,
        ) -> np.ndarray | None:
            """
            Returns the states reached by a single sampled word in the DAG or None if sampling failed.
            :param dag: The DAG to sample from.
            :param np_alpha: Numpy array containing the alpha values for each layer.
            :param layer: Current layer in the DAG.
            :param states: Current states in the DAG.
            :param phi: Probability threshold for sampling.
            """
            word = sample(dag, np_alpha, layer, states, phi, max_tries)
            if word is None:
                return None
            return dag.simulate(word).astype(np.uint8)

        def words_to_count(
            generator, batch_size, batch, layer, i, layer_states, debug="None"
        ) -> np.ndarray | None:
            """
            Counts the words in the generator.
            :param generator: Generator yielding words.
            :param batch_size: Number of words in each batch.
            :param i: Current batch index.
            :param layer_states: Current layer states.
            :return: Numpy array of word counts or None if the generator is empty.
            """
            dtype = np.uint64 if batch_size < 2**64 else object
            counts = np.zeros(2**i, dtype=dtype).reshape((2,) * i)
            for word in (
                tqdm(
                    generator,
                    total=batch_size,
                    desc=f"Processing layer {layer} of {total_layers}, batch {batch} of {total_batches}",
                    mininterval=1,
                    leave=False,
                )
                if progress_bar
                else generator
            ):
                if word is None:
                    return None
                order_before = tuple(word[layer_states][:i])
                counts[order_before] += 1
            return counts

        words = Parallel(
            n_jobs=self.available_processes,
            batch_size="auto",
            return_as="generator_unordered",
            temp_folder=None,
            max_nbytes=None,
        )(
            delayed(sample_one)(
                dag=self.dag,
                np_alpha=self.np_alpha[:layer],
                layer=layer,
                states=state_vector,
                phi=start_phi,
                max_tries=self.max_tries,
            )
            for _ in range(batch_start, batch_end)
        )
        counts = words_to_count(
            generator=words,
            batch_size=batch_end - batch_start,
            batch=batch,
            layer=layer,
            i=i,
            layer_states=layer_states,
            debug=self.debug,
        )
        queue.put(counts)

    def run(self):
        """
        Runs the FPRAS on the given input string and returns whether it accepts or rejects.
        """
        if self.empty:
            return 0

        self.np_alpha = np.empty(self.n + 1, dtype=object)
        self.np_alpha[0] = np.array([Fraction(0), Fraction(1)], dtype=Fraction)
        projected_np_alpha_size = (
            self.n + 1
        ) * 8  # 8 Bytes for object array overhead in n+1 layers
        min_size_fraction = (
            48 + 28 * 2
        )  # 48 Bytes for object overhead, atleast 28 Bytes for numerator and denominator each
        for l in range(self.n + 1):
            j = len(self.dag.states[l].nonzero()[0])
            projected_np_alpha_size += (
                8
                + min_size_fraction  # 8 Bytes for object array overhead and minimal fraction size
            ) * 2**j  # 2^j items in powerset for layer l
        if self.debug != "None":
            tqdm.write(
                f"Projected np_alpha size: {projected_np_alpha_size / 1024**2} MB"
            )

        queue = Queue()
        for layer in range(1, self.n + 1):
            layer_states = self.dag.states[layer].nonzero()[0]
            possible_predecessors = self.dag.states[layer - 1].nonzero()[0]
            self.np_alpha[layer] = np.array(
                [Fraction(0) for _ in range(2 ** len(layer_states))], dtype=Fraction
            ).reshape((2,) * len(layer_states))
            for i, state in enumerate(layer_states):
                state_vector = np.zeros(self.dag.m, dtype=bool)
                state_vector[state] = 1
                r_b = [
                    np.logical_and(
                        self.dag.states[layer - 1],
                        state_vector @ self.dag.transition_matrices_backwards[b],
                    )[possible_predecessors].astype(np.uint8)
                    for b in range(2)
                ]
                for b in range(2):
                    if r_b[b].shape[0] > 1:
                        r_b[b] = tuple(r_b[b])
                    else:
                        r_b[b] = r_b[b][0]
                ni_alpha = Fraction(
                    self.np_alpha[layer - 1][r_b[0]] + self.np_alpha[layer - 1][r_b[1]]
                )
                if len(layer_states) == 1:
                    idx = 1
                else:
                    idx = tuple(
                        ([0] * i)
                        + [1]
                        + ([slice(None)] * (len(layer_states) - (i + 1)))
                    )
                if i == 0:
                    # First state in order, therefore all words are used in S(P^alpha) (factor 1)
                    # So no need to sample
                    self.np_alpha[layer][idx] = ni_alpha
                else:
                    start_phi = Fraction(math.e**-5 / ni_alpha)
                    batch_memory_limit = (
                        psutil.virtual_memory().available // 4
                    ) * 3  # 75% of available memory
                    batch_size = batch_memory_limit // (
                        self.dag.m
                    )  # Adjust batch size based on available memory
                    number_of_batches = math.ceil(self.num_of_words / batch_size)
                    if self.debug == "Full":
                        tqdm.write(
                            f"Batch size: {batch_memory_limit} bytes ({(batch_memory_limit / 1024**3):.2f} GB)"
                        )
                        tqdm.write(
                            f"Batch size: {batch_size} words, for m={self.dag.m}, layer={layer}, state={i}, N(q)={ni_alpha}"
                        )
                        tqdm.write(
                            f"Number of words: {self.num_of_words}, results in {number_of_batches} batches"
                        )
                    for batch in range(number_of_batches):
                        batch_start = batch * batch_size
                        batch_end = min(batch_start + batch_size, self.num_of_words)
                        p = Process(
                            target=self.worker,
                            args=(
                                queue,
                                i,
                                batch_start,
                                batch_end,
                                layer,
                                state_vector,
                                start_phi,
                                layer_states,
                                batch + 1,
                                number_of_batches,
                                self.n,
                                self.progress_bar,
                            ),
                        )
                        p.start()
                        counts = queue.get()
                        p.terminate()
                        p.join()
                        if counts is None:
                            # Word generation failed too often, return 0
                            return 0

                        indeces = counts.nonzero()
                        index_rest = (1,) + (slice(None),) * (
                            len(layer_states) - (i + 1)
                        )
                        for index in zip(*indeces):
                            # Create index for np_alpha so that every powerset that contains the state and
                            # doesn't contain the previous states in the order that were included in the count
                            # are being included
                            adjusted_index = tuple(
                                [0 if val else slice(None) for val in index]
                            )
                            np_alpha_index = adjusted_index + index_rest
                            self.np_alpha[layer][np_alpha_index] += Fraction(
                                ni_alpha * counts[index], self.num_of_words
                            )
        if self.debug == "Full":
            for layer in range(self.n + 1):
                tqdm.write(f"\nLayer {layer}:\n{self.np_alpha[layer]}")
        self.np_alpha[self.n][1] = Fraction(
            self.np_alpha[self.n - 1][r_b[0]] + self.np_alpha[self.n - 1][r_b[1]]
        )
        return float(self.np_alpha[self.n][1])


class DependentFPRAS:
    def __init__(
        self,
        dag: DAG,
        n: int,
        epsilon: float,
        delta: float,
        debug: str = "None",
        progress_bar: bool = True,
    ):
        self.dag = dag
        if not self.dag.m:
            self.empty = True
        else:
            self.max_size = 0
            self.empty = False
            k = epsilon / (1 + epsilon)
            self.n = n
            self.n_s = math.ceil(4 * (n + 1) * (((1 + k) ** 2) / (k**2 * (1 - k))))
            self.n_t = math.ceil(8 * math.log(16 * n * self.dag.m))
            self.n_u = math.ceil(8 * math.log(1 / delta))
            self.ns_nt = self.n_s * self.n_t
            self.theta = 16 * self.ns_nt * n * (1 + k) * self.dag.m
            self.p = None
            self.available_processes = min(self.n_t, multiprocessing.cpu_count())
            self.n_t_per_process = [
                a.tolist()
                for a in np.array_split(np.arange(self.n_t), self.available_processes)
            ]
            self.shape_cache = None
            self.dtype_cache = np.bool_
            self.shm_cache = None
            self.cache = None
            self.shape_s_r = None
            self.dtype_s_r = np.uint64
            self.shm_s_r = None
            self.s_r = None
            self.shape_offsets = None
            self.dtype_offsets = np.uint64
            self.shm_offsets = None
            self.offsets = None
            self.s_r_new = None
            self.s_r_new_sizes = None
            self.debug = debug
            self.progress_bar = progress_bar
            if self.debug != "None":
                tqdm.write(f"ns_nt: {self.ns_nt}, n_s: {self.n_s}, n_t: {self.n_t}")

    def __del__(self):
        """Clean up temporary files when object is destroyed."""
        self._cleanup_shm()

    def _cleanup_shm(self):
        """Remove temporary directory and all files."""
        if not self.empty:
            if self.shm_cache is not None:
                self.shm_cache.close()
                self.shm_cache.unlink()
            if self.shm_s_r is not None:
                self.shm_s_r.close()
                self.shm_s_r.unlink()
            if self.shm_offsets is not None:
                self.shm_offsets.close()
                self.shm_offsets.unlink()

    def run(self):
        """
        Runs the Dependent FPRAS on the given input string and returns whether it accepts or rejects.
        """
        if self.empty:
            return 0, 0.0
        # Implement the logic for running the Dependent FPRAS
        if self.debug != "None":
            tqdm.write(f"Times rerun: {self.n_u}")
        est = []
        try:
            for run_number in (
                tqdm(
                    range(self.n_u),
                    desc="RAM S_R Dependent Processing",
                    total=self.n_u,
                    maxinterval=1,
                )
                if self.progress_bar
                else range(self.n_u)
            ):
                est.append(self.count_nfa_core(run_number))
            return statistics.median(est), round(self.max_size / (1024**3), 2)
        finally:
            self._cleanup_shm()

    def count_nfa_core(self, run_number: int) -> Fraction:
        """
        Core logic for counting the NFA.
        This method should implement the core counting logic for the NFA.
        """
        self.p = np.zeros((2, self.dag.m), dtype=Fraction)
        self.p[1, self.dag.start_states[0]] = Fraction(1)
        self.compute_cache(0, run_number)
        sample_count = self.ns_nt
        for i in (
            tqdm(range(1, self.n + 1), desc="Processing layers", leave=False)
            if self.progress_bar
            else range(1, self.n + 1)
        ):
            self.p[0] = self.p[1].copy()
            self.p[1] = np.zeros(self.dag.m, dtype=Fraction)
            layer_states = self.dag.states[i].nonzero()[0]
            self.s_r_new = np.empty((2, layer_states.shape[0] * self.n_t), dtype=object)
            self.s_r_new_sizes = np.zeros(self.s_r_new.shape, dtype=int)
            if self.debug == "Full" and not run_number:
                tqdm.write(f"Layer {i}, States: {layer_states}")

            for idq, q in enumerate(layer_states):
                self.estimate_and_sample(q=q, layer=i, idq=idq)
                sample_count += np.sum(
                    self.s_r_new_sizes[:, idq * self.n_t : (idq + 1) * self.n_t]
                )
                if sample_count >= self.theta:
                    return 0
            self.compute_cache(i, run_number)
        res = Fraction(
            numerator=self.p[1, self.dag.accept_states[0]].denominator,
            denominator=self.p[1, self.dag.accept_states[0]].numerator,
        )
        if self.debug != "None":
            tqdm.write(f"Run {run_number} complete. Result: {float(res)}")
        return res

    def compute_cache(self, i: int, run_number: int):
        """
        Computes the cache for the NFA.
        This method should implement the logic to compute the cache for the NFA.
        """
        if 0 > i > self.dag.n:
            raise ValueError(f"Index {i} out of bounds for NFA with n={self.dag.n}.")
        if i == 0:
            self.shape_cache = (1, self.dag.m)
            self.shm_cache = shared_memory.SharedMemory(
                create=True,
                size=int(
                    np.prod(self.shape_cache) * np.dtype(self.dtype_cache).itemsize
                ),
            )
            self.cache = np.ndarray(
                self.shape_cache, dtype=self.dtype_cache, buffer=self.shm_cache.buf
            )
            self.cache.fill(False)
            self.cache[0, self.dag.start_states[0]] = True

            self.shape_offsets = (self.n_t + 1,)
            self.shm_offsets = shared_memory.SharedMemory(
                create=True,
                size=int(
                    np.prod(self.shape_offsets) * np.dtype(self.dtype_offsets).itemsize
                ),
            )
            self.offsets = np.ndarray(
                self.shape_offsets,
                dtype=self.dtype_offsets,
                buffer=self.shm_offsets.buf,
            )
            self.offsets.fill(0)
            self.offsets[:] = np.arange(self.n_t + 1) * self.n_s

            self.shape_s_r = (self.ns_nt,)
            self.shm_s_r = shared_memory.SharedMemory(
                create=True,
                size=int(np.prod(self.shape_s_r) * np.dtype(self.dtype_s_r).itemsize),
            )
            self.s_r = np.ndarray(
                self.shape_s_r, dtype=self.dtype_s_r, buffer=self.shm_s_r.buf
            )
            self.s_r.fill(0)  # Every set S contains the empty word (word 0 in Cache)

        elif i == self.dag.n:
            if self.debug != "None":
                tqdm.write(f"Max memory used: {self.max_size / (1024**3):.2f} GB")
            return
        else:
            states = self.dag.states[i]
            self.s_r_new = [
                np.concatenate(self.s_r_new[0].tolist()),
                np.concatenate(self.s_r_new[1].tolist()),
            ]
            indices_0, new_s_r_data_0 = np.unique(
                self.s_r_new[0],
                return_inverse=True,
            )
            indices_1, new_s_r_data_1 = np.unique(
                self.s_r_new[1],
                return_inverse=True,
            )
            new_s_r_data_1 = new_s_r_data_1 + indices_0.shape[0]
            new_cache_data = np.append(
                self.dag.simulate_one(self.cache[indices_0], 0, i - 1),
                self.dag.simulate_one(self.cache[indices_1], 1, i - 1),
                axis=0,
            )
            cache_size = new_cache_data.shape[0]
            self.shape_cache = (cache_size, self.dag.m)
            self.shm_cache.close()
            self.shm_cache.unlink()
            self.shm_cache = shared_memory.SharedMemory(
                create=True,
                size=int(
                    np.prod(self.shape_cache) * np.dtype(self.dtype_cache).itemsize
                ),
            )
            self.cache = np.ndarray(
                self.shape_cache, dtype=self.dtype_cache, buffer=self.shm_cache.buf
            )
            self.cache[:] = new_cache_data
            del new_cache_data

            self.shape_s_r = self.s_r_new[0].shape[0] + self.s_r_new[1].shape[0]
            new_s_r_data = np.empty(self.shape_s_r, dtype=self.dtype_s_r)
            index_0 = np.zeros(self.shape_s_r, dtype=bool)
            index_1 = np.zeros(self.shape_s_r, dtype=bool)
            number_of_states = states.nonzero()[0].shape[0]
            new_offsets_data = np.empty(
                (number_of_states * self.n_t + 1), dtype=self.dtype_offsets
            )
            start_0 = 0
            for q in range(number_of_states):
                for k in range(self.n_t):
                    new_offsets_data[q * self.n_t + k] = start_0
                    start_1 = start_0 + self.s_r_new_sizes[0, q * self.n_t + k]
                    index_0[start_0:start_1] = True
                    start_0 = start_1 + self.s_r_new_sizes[1, q * self.n_t + k]
                    index_1[start_1:start_0] = True
            new_offsets_data[-1] = start_0
            self.shape_offsets = (number_of_states * self.n_t + 1,)
            self.shm_offsets.close()
            self.shm_offsets.unlink()
            self.shm_offsets = shared_memory.SharedMemory(
                create=True,
                size=int(
                    np.prod(self.shape_offsets) * np.dtype(self.dtype_offsets).itemsize
                ),
            )
            self.offsets = np.ndarray(
                self.shape_offsets,
                dtype=self.dtype_offsets,
                buffer=self.shm_offsets.buf,
            )
            self.offsets[:] = new_offsets_data
            del new_offsets_data

            new_s_r_data[index_0] = new_s_r_data_0
            new_s_r_data[index_1] = new_s_r_data_1
            self.shm_s_r.close()
            self.shm_s_r.unlink()
            self.shm_s_r = shared_memory.SharedMemory(
                create=True,
                size=int(np.prod(self.shape_s_r) * np.dtype(self.dtype_s_r).itemsize),
            )
            self.s_r = np.ndarray(
                self.shape_s_r, dtype=self.dtype_s_r, buffer=self.shm_s_r.buf
            )
            self.s_r[:] = new_s_r_data
            del new_s_r_data

            if self.debug == "Full":
                tqdm.write(
                    f"Cache size for layer {i}: {self.cache.nbytes / (1024 * 1024 * 1024):.2f} GB"
                )
                tqdm.write(
                    f"s_r size for layer {i}: {self.s_r.nbytes / (1024 * 1024 * 1024):.2f} GB"
                )
                tqdm.write(
                    f"Offsets size for layer {i}: {self.offsets.nbytes / (1024 * 1024 * 1024):.2f} GB"
                )
                tqdm.write(
                    f"Total size for layer {i}: {(self.cache.nbytes + self.s_r.nbytes + self.offsets.nbytes) / (1024 * 1024 * 1024):.2f} GB"
                )
            elif self.debug == "Minimal":
                if self.max_size < (
                    self.cache.nbytes + self.s_r.nbytes + self.offsets.nbytes
                ):
                    self.max_size = (
                        self.cache.nbytes + self.s_r.nbytes + self.offsets.nbytes
                    )

    def estimate_and_sample(
        self,
        q: int,
        layer: int,
        idq: int,
    ):
        state_vector = np.zeros(self.dag.m, dtype=bool)
        state_vector[q] = True
        pred_states = [
            np.logical_and(
                self.dag.states[layer - 1],
                state_vector @ self.dag.transition_matrices_backwards[b],
            )
            for b in range(2)
        ]
        pred_states_vector = np.logical_or(pred_states[0], pred_states[1])
        p_pred_states = self.p[0, pred_states_vector]
        pred_states = [np.nonzero(pred_states[b])[0] for b in range(2)]
        all_pred_states = np.nonzero(pred_states_vector)[0]
        pred_states_mapping = {
            pred_state: np.where(self.dag.states[layer - 1].nonzero()[0] == pred_state)[
                0
            ][0]
            for pred_state in all_pred_states
        }
        p_q = np.min(p_pred_states)
        ret = Parallel(
            n_jobs=self.available_processes,
            batch_size="auto",
            return_as="list",
            temp_folder=None,
            max_nbytes=None,
        )(
            delayed(self._estimate_and_sample_process)(
                shm_cache_name=self.shm_cache.name,
                shape_cache=self.shape_cache,
                dtype_cache=np.dtype(self.dtype_cache).str,
                shm_s_r_name=self.shm_s_r.name,
                shape_s_r=self.shape_s_r,
                dtype_s_r=np.dtype(self.dtype_s_r).str,
                shm_offsets_name=self.shm_offsets.name,
                shape_offsets=self.shape_offsets,
                dtype_offsets=np.dtype(self.dtype_offsets).str,
                n_t=self.n_t,
                indices_nt=indices_nt,
                pred_states=pred_states,
                all_pred_states=all_pred_states,
                pred_states_mapping=pred_states_mapping,
                p_q=p_q,
                p=self.p[0],
            )
            for indices_nt in self.n_t_per_process
        )
        factor = self.n_s * p_q
        factor = Fraction(factor.denominator, factor.numerator)
        s_r_dach = np.empty((2, self.n_t), dtype=object)
        m_j = np.zeros((self.n_t), dtype=Fraction)
        idx = 0
        for s_nt_dach_list in ret:
            for s_nt_dach in s_nt_dach_list:
                s_r_dach[:, idx] = s_nt_dach
                m_j[idx] = (s_nt_dach[0].shape[0] + s_nt_dach[1].shape[0]) * factor
                idx += 1
        median_m_j = np.median(m_j)
        self.p[1, q] = min(p_q, Fraction(median_m_j.denominator, median_m_j.numerator))
        for idx in range(self.n_t):
            s = s_r_dach[:, idx]
            res_0 = s[0][np.random.rand(s[0].shape[0]) < self.p[1, q] / p_q]
            res_1 = s[1][np.random.rand(s[1].shape[0]) < self.p[1, q] / p_q]
            self.s_r_new[0, idq * self.n_t + idx] = res_0
            self.s_r_new[1, idq * self.n_t + idx] = res_1
            self.s_r_new_sizes[0, idq * self.n_t + idx] = res_0.shape[0]
            self.s_r_new_sizes[1, idq * self.n_t + idx] = res_1.shape[0]

    @staticmethod
    def _estimate_and_sample_process(
        shm_cache_name: str,
        shape_cache: tuple,
        dtype_cache: str,
        shm_s_r_name: str,
        shape_s_r: tuple,
        dtype_s_r: str,
        shm_offsets_name: str,
        shape_offsets: tuple,
        dtype_offsets: str,
        n_t: int,
        indices_nt: list[int],
        pred_states: list,
        all_pred_states: np.ndarray,
        pred_states_mapping: dict,
        p_q: Fraction,
        p: np.ndarray,
    ) -> Fraction:
        """
        Estimates and samples the NFA for a given state q.
        This method should implement the logic to estimate and sample the NFA.
        :param q: The state in the NFA to estimate and sample.
        """

        def union_local(
            s: np.ndarray, pred_states: list, cache: np.ndarray
        ) -> np.ndarray:
            """
            Computes the union of two states in the NFA.
            This method should implement the logic to compute the union of two states.
            :param q: The first state in the NFA.
            :param layer: The layer of the state.
            :param s: The set from which to compute the union.
            :return: The set s_acute which is the union of set s for state q.
            """

            s_acute = np.empty((2), dtype=object)
            for b in range(2):
                b_pred = pred_states[b]
                if b_pred.shape[0] != 0:
                    s_acute_b = np.empty(b_pred.shape[0], dtype=object)
                    for index, j in enumerate(b_pred):
                        l = b_pred[:index]
                        j_first = np.where(~cache[:, l].any(axis=1))[0]
                        temp = s[np.where(all_pred_states == j)[0]][0]
                        mask = np.isin(temp, j_first)
                        s_acute_b[index] = temp[mask]
                    s_acute[b] = np.concatenate(s_acute_b.tolist())
                else:
                    s_acute[b] = np.array([], dtype=np.uint64)
            return s_acute

        shm_cache = shared_memory.SharedMemory(name=shm_cache_name)
        cache = np.ndarray(
            shape_cache, dtype=np.dtype(dtype_cache), buffer=shm_cache.buf
        )
        shm_s_r = shared_memory.SharedMemory(name=shm_s_r_name)
        s_r = np.ndarray(shape_s_r, dtype=np.dtype(dtype_s_r), buffer=shm_s_r.buf)
        shm_offsets = shared_memory.SharedMemory(name=shm_offsets_name)
        offsets = np.ndarray(
            shape_offsets, dtype=np.dtype(dtype_offsets), buffer=shm_offsets.buf
        )
        s_r_dach = []
        for idx_nt in indices_nt:
            s_r_quer = np.empty(all_pred_states.shape[0], dtype=object)
            for idx, q_pred in enumerate(all_pred_states):
                s = s_r[
                    offsets[pred_states_mapping[q_pred] * n_t + idx_nt] : offsets[
                        pred_states_mapping[q_pred] * n_t + idx_nt + 1
                    ]
                ]
                s_r_quer[idx] = s[np.random.rand(s.shape[0]) < p_q / p[q_pred]]
            s_r_dach.append(union_local(s_r_quer, pred_states, cache))
        shm_s_r.close()
        shm_cache.close()
        shm_offsets.close()
        return s_r_dach


class BruteForce:
    def __init__(
        self, dag: DAG, n: int, debug: str = "None", progress_bar: bool = True
    ):
        self._dag = dag
        self._empty = True if not self._dag.m else False
        self._alphabet = [0, 1]
        self._n = n
        self._words = []

    def binary_generator(self, n):
        """Generator version for large n"""
        for i in range(2**n):
            binary_str = format(i, f"0{n}b")
            yield np.pad(
                np.array([c == "1" for c in binary_str], dtype=np.uint8),
                (0, self._n - n),
                mode="constant",
                constant_values=False,
            )

    def run(self):
        """
        Runs the BruteForce algorithm on the given input string and
        returns whether it accepts or rejects.
        """
        # Implement the logic for running the BruteForce algorithm
        if self._empty:
            return 0
        process_count = min(
            2 ** int(math.log2(multiprocessing.cpu_count())), 2**self._n, 2**64
        )
        start_n = int(math.log2(process_count))
        counts = Parallel(n_jobs=process_count)(
            delayed(self.dfs)(start_str, start_n)
            for start_str in self.binary_generator(start_n)
        )
        return sum(counts)

    def dfs(self, cur_str: np.ndarray, layer: int) -> int:
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


class BruteForcePowerset:
    def __init__(
        self, dag: DAG, n: int, debug: str = "None", progress_bar: bool = True
    ):
        self._dag = dag
        self._empty = True if not self._dag.m else False
        self._n = n
        self._debug = debug
        self._progress_bar = progress_bar

    def run(self):
        """
        Runs the BruteForcePowerset algorithm on the given input string and
        returns whether it accepts or rejects.
        """
        if self._empty:
            return 0, 0.0
        cur_states_set = self._dag.states[0].reshape(1, -1)
        cur_counts = np.ones(1, dtype=object)
        max_size = cur_states_set.nbytes
        for layer in range(self._n):
            next_states_set = self._dag.simulate_one(cur_states_set, 0, layer)
            next_states_set = np.append(
                next_states_set,
                self._dag.simulate_one(cur_states_set, 1, layer),
                axis=0,
            )
            prev_counts = np.append(cur_counts, cur_counts, axis=0)
            unique_next_states_set, indices = np.unique(
                next_states_set, axis=0, return_inverse=True
            )
            cur_states_set = unique_next_states_set
            cur_counts = np.zeros(unique_next_states_set.shape[0], dtype=object)
            for count_idx, idx in enumerate(indices):
                cur_counts[idx] += prev_counts[count_idx]
            cur_size = cur_states_set.nbytes
            if max_size < cur_size:
                max_size = cur_size
        idx_accept = np.where(cur_states_set[:, self._dag.accept_states[0]])[0]
        if idx_accept.size == 0:
            return 0, 0.0
        return cur_counts[idx_accept[0]], round(max_size / (1024**3), 4)


if __name__ == "__main__":
    pass
