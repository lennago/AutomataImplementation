import numpy as np
import scipy.sparse as sp


class NFA:
    def __init__(
        self,
        num_states: int,
        transitions: list[tuple[int, int | str, int]],
        start_states: list[int],
        accept_states: list[int],
        debug: bool = True,
    ):
        self.num_states = num_states + 2
        self.start_states = np.array(start_states, dtype=int)
        self.accept_states = np.array(accept_states, dtype=int)
        self.debug = debug
        self.alphabet = [0, 1, 2]  # 0 and 1 for symbols, 2 for epsilon
        self.transition_matrices = np.zeros(
            (3, self.num_states, self.num_states), dtype=bool
        )
        self.transition_matrices[2] = np.identity(self.num_states, dtype=bool)
        self._initialize_transitions(transitions)
        self._create_single_start_and_accept()
        density = np.count_nonzero(self.transition_matrices) / (
            self.num_states * self.num_states * 2
        )
        self.sparse = density < 0.05
        if self.sparse:
            self.transition_matrices = np.array(
                [sp.csr_array(mat, dtype=bool) for mat in self.transition_matrices],
                dtype=object,
            )
            self.transition_matrices_backwards = np.array(
                [mat.T.tocsr() for mat in self.transition_matrices[:2]], dtype=object
            )
        else:
            self.transition_matrices_backwards = np.array(
                [mat.T for mat in self.transition_matrices[:2]], dtype=bool
            )
        self._remove_unreachable_and_not_accepting_states()

    def _initialize_transitions(
        self, transitions: list[tuple[int, int | str, int]]
    ) -> None:
        """
        Initializes the transition matrices based on the provided transitions.
        Each transition is a tuple of (source state, symbol, destination state).
        """
        if not transitions:
            return
        transitions_array = np.array(transitions)
        for symbol in self.alphabet:
            mask = transitions_array[:, 1] == symbol
            if mask.any():
                valid_transitions = transitions_array[mask]
                src = valid_transitions[:, 0]
                dst = valid_transitions[:, 2]
                self.transition_matrices[symbol][src, dst] = True

    def _remove_unreachable_and_not_accepting_states(self) -> None:
        """
        Removes unreachable and not accepting states from the NFA.
        This is done by iteratively marking reachable and accepting states until no new states are found.
        """
        n = self.num_states
        reachable = np.zeros(n, dtype=bool)
        accepting = np.zeros(n, dtype=bool)
        reachable[self.start_states] = True
        accepting[self.accept_states] = True
        if self.sparse:
            forwards_transitions = (
                self.transition_matrices[0] + self.transition_matrices[1]
            )
            backwards_transitions = (
                self.transition_matrices_backwards[0]
                + self.transition_matrices_backwards[1]
            )
        else:
            forwards_transitions = (
                self.transition_matrices[0] | self.transition_matrices[1]
            )
            backwards_transitions = (
                self.transition_matrices_backwards[0]
                | self.transition_matrices_backwards[1]
            )
        for _ in range(n):
            old_reachable = reachable.copy()
            old_accepting = accepting.copy()
            reachable = reachable @ forwards_transitions
            accepting = accepting @ backwards_transitions
            reachable |= old_reachable
            accepting |= old_accepting
            if np.array_equal(reachable, old_reachable) and np.array_equal(
                accepting, old_accepting
            ):
                break
        remaining_states = reachable & accepting
        self.num_states = int(remaining_states.sum())
        if not self.num_states:
            return
        self.transition_matrices = np.array(
            [
                mat[np.ix_(remaining_states, remaining_states)]
                for mat in self.transition_matrices
            ]
        )
        if self.sparse:
            self.transition_matrices_backwards = np.array(
                [mat.T.tocsr() for mat in self.transition_matrices], dtype=object
            )
        else:
            self.transition_matrices_backwards = np.array(
                [mat.T for mat in self.transition_matrices], dtype=bool
            )
        self.start_states = np.array([self.num_states - 2], dtype=int)
        self.accept_states = np.array([self.num_states - 1], dtype=int)

    def _create_single_start_and_accept(self) -> None:
        """
        Creates a single start and accept state for the NFA.
        This is done by adding two new states: one for the start state and one for the accept state.
        The new start state has epsilon transitions to all original start states,
        and all original accept states have epsilon transitions to the new accept state.
        The epsilon transitions are then eliminated.
        """
        new_start_state = self.num_states - 2
        new_accept_state = self.num_states - 1
        for old_start_state in self.start_states:
            self.transition_matrices[2][new_start_state, old_start_state] = True
        for old_accept_state in self.accept_states:
            self.transition_matrices[2][old_accept_state, new_accept_state] = True
        self.start_states = np.array([new_start_state], dtype=int)
        self.accept_states = np.array([new_accept_state], dtype=int)
        self._eliminate_epsilon_transitions()

    def _eliminate_epsilon_transitions(self) -> None:
        """
        Eliminates epsilon transitions from the NFA.
        """
        density = np.count_nonzero(self.transition_matrices[2]) / (
            self.num_states * self.num_states
        )
        sparse = density < 0.05
        epsilon_closure = self._get_epsilon_closure(sparse=sparse)
        for i in range(2):
            cur_matrix = self.transition_matrices[i]
            self.transition_matrices[i] = epsilon_closure @ cur_matrix @ epsilon_closure
        self.transition_matrices = self.transition_matrices[
            :2
        ]  # Keep only the non-epsilon matrices
        self.alphabet.remove(2)  # Remove the epsilon transition matrix

    def _get_epsilon_closure(self, sparse: bool = False) -> np.ndarray:
        """
        Computes the epsilon closure for each state in the NFA.
        """
        epsilon_matrix = self.transition_matrices[2]
        n = self.num_states
        if sparse:
            epsilon_sparse = sp.csr_array(epsilon_matrix)
            closure = sp.eye_array(n, dtype=bool, format="csr")
            power = epsilon_sparse.copy()
            max_iterations = int(np.ceil(np.log2(n))) + 1
            for _ in range(max_iterations):
                closure = closure + power
                new_power = power @ power
                if (new_power != power).nnz == 0:
                    break
                power = new_power
            return closure
        closure = np.identity(n, dtype=bool)
        power = epsilon_matrix.copy()
        for _ in range(int(np.ceil(np.log2(n))) + 1):
            closure |= power
            new_power = power @ power
            if np.array_equal(new_power, power):
                break
            power = new_power
        return closure

    def _reduce_equals(self, outgoing: bool = True) -> bool:
        """
        Merges equivalent states in the NFA.
        Two states are considered equivalent if they have identical ingoing or outgoing
        transition behavior.
        Returns True if any states were merged, False otherwise.
        This method modifies the NFA in place.
        """
        if self.num_states < 2:
            # Can't reduce less than 2 states
            return False
        if self.sparse:
            self.transition_matrices = np.array(
                [mat.todense() for mat in self.transition_matrices], dtype=bool
            )
        if outgoing:
            helper = self.transition_matrices.transpose(1, 0, 2).reshape(
                self.num_states, -1
            )
        else:
            helper = self.transition_matrices.transpose(2, 0, 1).reshape(
                self.num_states, -1
            )

        unique_transitions, inverse_indices, counts = np.unique(
            helper, axis=0, return_inverse=True, return_counts=True
        )
        if len(unique_transitions) == self.num_states:
            # No equivalent states found
            if self.sparse:
                # Return to sparse format
                self.transition_matrices = np.array(
                    [sp.csr_array(mat, dtype=bool) for mat in self.transition_matrices],
                    dtype=object,
                )
            return False
        unmerged_duplicates = [i for i in np.nonzero(counts > 1)[0]]
        states_to_merge = [
            np.nonzero(inverse_indices == i)[0] for i in unmerged_duplicates
        ]
        filtered_states_to_merge = []
        for states in states_to_merge:
            # Check if start or accept states are in this group
            has_start = np.any(np.isin(states, self.start_states))
            has_accept = np.any(np.isin(states, self.accept_states))

            if not (has_start or has_accept):
                filtered_states_to_merge.append(states)

        states_to_merge = filtered_states_to_merge
        if not states_to_merge:
            if self.sparse:
                # Return to sparse format
                self.transition_matrices = np.array(
                    [sp.csr_array(mat, dtype=bool) for mat in self.transition_matrices],
                    dtype=object,
                )
            return False
        if self.debug:
            print(f"Found {len(states_to_merge)} sets of equivalent states to merge.")
            print(f"States to merge: {states_to_merge}")
        states_to_keep = {i for i in range(self.num_states)}
        for states in states_to_merge:
            new_transition = np.zeros((2, self.num_states), dtype=bool)
            for state in states:
                new_transition[0] |= (
                    self.transition_matrices[0][:, state]
                    if outgoing
                    else self.transition_matrices[0][state]
                )
                new_transition[1] |= (
                    self.transition_matrices[1][:, state]
                    if outgoing
                    else self.transition_matrices[1][state]
                )
            if outgoing:
                self.transition_matrices[0][:, states[-1]] = new_transition[0]
                self.transition_matrices[1][:, states[-1]] = new_transition[1]
            else:
                self.transition_matrices[0][states[-1]] = new_transition[0]
                self.transition_matrices[1][states[-1]] = new_transition[1]
            states_to_keep -= set(states[:-1])
        states_to_keep = np.array(sorted(states_to_keep), dtype=int)
        self.transition_matrices = np.array(
            [
                mat[states_to_keep, :][:, states_to_keep]
                for mat in self.transition_matrices
            ]
        )
        density = np.count_nonzero(self.transition_matrices) / (
            self.num_states * self.num_states * 2
        )
        self.sparse = density < 0.05
        if self.sparse:
            self.transition_matrices = np.array(
                [sp.csr_array(mat, dtype=bool) for mat in self.transition_matrices],
                dtype=object,
            )
            self.transition_matrices_backwards = np.array(
                [mat.T.tocsr() for mat in self.transition_matrices], dtype=object
            )
        else:
            self.transition_matrices_backwards = np.array(
                [mat.T for mat in self.transition_matrices], dtype=bool
            )
        self.num_states = len(states_to_keep)
        self.accept_states = np.array([self.num_states - 1], dtype=int)
        self.start_states = np.array([self.num_states - 2], dtype=int)
        return True

    def minimize(self) -> None:
        """
        Minimizes the NFA by merging equivalent states.
        This is done by repeatedly applying the _reduce_equals method.
        """
        changed = True
        while changed:
            changed = self._reduce_equals(outgoing=True)
            changed |= self._reduce_equals(outgoing=False)

    def simulate(self, input_string: np.ndarray) -> np.ndarray:
        reached_states = np.zeros(self.num_states, dtype=bool)
        reached_states[self.start_states] = True
        for symbol in input_string:
            if symbol not in self.alphabet:
                raise ValueError(f"Symbol {symbol} not in alphabet {self.alphabet}.")
            reached_states = reached_states @ self.transition_matrices[symbol]
        return reached_states

    def is_accepted(self, input_string: np.ndarray) -> bool:
        return self.simulate(input_string)[self.accept_states].any()


class DAG:
    def __init__(self, nfa: NFA, n: int):
        self.transition_matrices = nfa.transition_matrices
        self.transition_matrices_backwards = nfa.transition_matrices_backwards
        self.sparse = nfa.sparse
        self.alphabet = [0, 1]
        self.m = nfa.num_states
        self.n = n
        if n < 1:
            raise ValueError("n must be at least 1.")
        if self.m == 0:
            self.num_states = 0
            self.start_states = np.array([], dtype=int)
            self.accept_states = np.array([], dtype=int)
            self.states = np.zeros((self.n + 1, self.m), dtype=bool)
            return
        self.num_states = self.m * (n + 1)
        self.start_states = nfa.start_states
        self.accept_states = np.array([self.m - 1], dtype=int)
        self.states = np.ones((self.n + 1, self.m), dtype=bool)
        self.states[0] = np.zeros(self.m, dtype=bool)
        self.states[0][self.start_states] = True
        self.states[self.n] = np.zeros(self.m, dtype=bool)
        self.states[self.n][-1] = True
        if n == 1:
            if not (
                np.logical_and(
                    self.states[0] @ self.transition_matrices[0], self.states[-1]
                ).any()
                or np.logical_and(
                    self.states[0] @ self.transition_matrices[1], self.states[-1]
                ).any()
            ):
                # Set m to 0 since L^n is empty
                self.m = 0
        else:
            self._compute_layers()

    def _compute_layers(self):
        if self.sparse:
            forwards_transitions = (
                self.transition_matrices[0] + self.transition_matrices[1]
            )
            backwards_transitions = (
                self.transition_matrices_backwards[0]
                + self.transition_matrices_backwards[1]
            )
        else:
            forwards_transitions = (
                self.transition_matrices[0] | self.transition_matrices[1]
            )
            backwards_transitions = (
                self.transition_matrices_backwards[0]
                | self.transition_matrices_backwards[1]
            )
        for i in range(1, self.n + 1):
            reachable = self.states[i - 1] @ forwards_transitions
            accepting = self.states[self.n + 1 - i] @ backwards_transitions
            if not (np.any(reachable) and np.any(accepting)):
                # Set m to 0 since L^n is empty
                self.m = 0
                return
            self.states[i] &= reachable
            self.states[self.n - i] &= accepting
        relevant_states = self.states.any(axis=0)
        self.states = self.states[:, relevant_states]
        new_m = int(relevant_states.sum())
        self.start_states = np.array([self.start_states[0] - self.m + new_m], dtype=int)
        self.accept_states = np.array([new_m - 1], dtype=int)
        self.m = new_m

        self.transition_matrices = np.array(
            [
                mat[relevant_states, :][:, relevant_states]
                for mat in self.transition_matrices
            ]
        )
        if self.sparse:
            self.transition_matrices_backwards = np.array(
                [mat.T.tocsr() for mat in self.transition_matrices], dtype=object
            )
        else:
            self.transition_matrices_backwards = np.array(
                [mat.T for mat in self.transition_matrices], dtype=bool
            )

    def simulate(self, input_string: np.ndarray) -> np.ndarray:
        """
        Simulates the DAG on the given input string and returns the final state.
        The input string is expected to be a numpy array of symbols from the alphabet.
        """
        if len(input_string) > self.n:
            raise ValueError(
                f"Input strings must have length <= {self.n}, got {len(input_string)}."
            )
        current_states = self.states[0].copy()
        for symbol in input_string:
            if symbol not in self.alphabet:
                raise ValueError(f"Symbol {symbol} not in alphabet {self.alphabet}.")
            current_states = current_states @ self.transition_matrices[symbol]
        return np.logical_and(current_states, self.states[len(input_string)])

    def simulate_one(self, states: np.ndarray, symbol: int, layer: int) -> np.ndarray:
        """
        Simulates the DAG on the given states with a single symbol and returns the next states.
        The input states are expected to be a numpy array of boolean values representing the current states.
        """
        if symbol not in self.alphabet:
            raise ValueError(f"Symbol {symbol} not in alphabet {self.alphabet}.")
        return np.logical_and(
            states @ self.transition_matrices[symbol], self.states[layer + 1]
        )

    def simulate_batch(self, input_batch: np.ndarray) -> np.ndarray:
        """
        Simulates the DAG on a batch of input strings and returns the final states.
        The input batch is expected to be a 2D numpy array where each row is an input string.
        """
        batch_size, n = input_batch.shape
        if n > self.n:
            raise ValueError(f"Input strings must have length <= {self.n}, got {n}.")
        current_states = np.zeros((batch_size, self.m), dtype=bool)
        current_states[:, self.start_states] = True
        for i in range(n):
            symbols = input_batch[:, i]
            mask_0 = symbols == 0
            mask_1 = symbols == 1
            current_states[mask_0] = (
                current_states[mask_0] @ self.transition_matrices[0]
            )
            current_states[mask_1] = (
                current_states[mask_1] @ self.transition_matrices[1]
            )
        return np.logical_and(current_states, self.states[n])

    def is_accepted(self, input_string: np.ndarray) -> bool:
        """
        Checks if the input string is accepted by the DAG.
        The input string is expected to be a numpy array of symbols from the alphabet.
        """
        if len(input_string) != self.n:
            return False
        reached_states = self.simulate(input_string)
        return reached_states[-1]

    def is_accepted_batch(self, input_batch: np.ndarray) -> np.ndarray:
        """
        Checks if each input string in the batch is accepted by the DAG.
        The input batch is expected to be a 2D numpy array where each row is an input string.
        """
        if input_batch.shape[1] != self.n:
            return np.zeros(input_batch.shape[0], dtype=bool)
        reached_states = self.simulate_batch(input_batch)
        return reached_states[:, -1]
