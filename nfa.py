import numpy as np

try:
    import cupy as cp
except ImportError:
    cp = None


class NFA:
    def __init__(
        self,
        num_states: int,
        transitions: list[tuple[int, int | str, int]],
        start_states: list[int],
        accept_states: list[int],
    ):
        self.num_states = num_states + 2
        self.start_states = list(start_states)
        self.accept_states = list(accept_states)
        self.alphabet = [0, 1, ""]
        self.transition_matrices = np.zeros(
            (3, num_states + 2, num_states + 2), dtype=bool
        )
        self.transition_matrices[2] = np.identity(num_states + 2, dtype=bool)

        for src, symbol, dst in transitions:
            if symbol in self.alphabet:
                idx = self.alphabet.index(symbol)
                self.transition_matrices[idx][src, dst] = True

        self._create_single_start_and_accept()
        self.transition_matrices_backwards = [
            self.transition_matrices[0].T,
            self.transition_matrices[1].T,
        ]
        self._remove_unreachable_and_not_accepting_states()

    def _remove_unreachable_and_not_accepting_states(self) -> None:
        """
        Removes unreachable and not accepting states from the NFA.
        This is done by iteratively marking reachable and accepting states until no new states are found.
        """
        reachable = np.zeros(self.num_states, dtype=bool)
        accepting = np.zeros(self.num_states, dtype=bool)
        remaining_states = np.zeros(self.num_states, dtype=bool)
        reachable[self.start_states] = True
        accepting[self.accept_states] = True
        for _ in range(self.num_states):
            for b in range(2):
                reachable = np.logical_or(
                    reachable, reachable.dot(self.transition_matrices[b])
                )
                accepting = np.logical_or(
                    accepting, accepting.dot(self.transition_matrices_backwards[b])
                )
        remaining_states = np.logical_and(reachable, accepting)
        self.transition_matrices = [
            mat[remaining_states][:, remaining_states]
            for mat in self.transition_matrices
        ]
        self.transition_matrices_backwards = [mat.T for mat in self.transition_matrices]
        self.num_states = remaining_states.sum()
        if self.num_states:
            self.start_states = [self.num_states - 2]
            self.accept_states = [self.num_states - 1]

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
        self.start_states = [new_start_state]
        self.accept_states = [new_accept_state]
        self._eliminate_epsilon_transitions()

    def _eliminate_epsilon_transitions(self) -> None:
        """
        Eliminates epsilon transitions from the NFA.
        """
        epsilon_closure = self._get_epsilon_closure()
        for i in range(2):
            cur_matrix = self.transition_matrices[i]
            self.transition_matrices[i] = epsilon_closure.dot(cur_matrix).dot(
                epsilon_closure
            )
        self.transition_matrices = self.transition_matrices[
            :2
        ]  # Keep only the non-epsilon matrices
        self.alphabet.remove("")  # Remove the epsilon transition matrix

    def _get_epsilon_closure(self) -> np.ndarray:
        """
        Computes the epsilon closure for each state in the NFA.
        """
        cur_epsilon_matrix = self.transition_matrices[2]
        next_epsilon_matrix = np.ones((self.num_states, self.num_states), dtype=bool)
        for _ in range(self.num_states):
            next_epsilon_matrix = cur_epsilon_matrix.dot(cur_epsilon_matrix)
            if np.array_equal(next_epsilon_matrix, cur_epsilon_matrix):
                break
            cur_epsilon_matrix = next_epsilon_matrix
        return cur_epsilon_matrix

    def simulate(self, input_string: np.ndarray) -> np.ndarray:
        reached_states = np.zeros(self.num_states, dtype=bool)
        reached_states[self.start_states] = True
        for symbol in input_string:
            if symbol not in self.alphabet:
                raise ValueError(f"Symbol {symbol} not in alphabet {self.alphabet}.")
            reached_states = reached_states.dot(
                self.transition_matrices[self.alphabet.index(symbol)]
            )
        return reached_states

    def is_accepted(self, input_string: np.ndarray) -> bool:
        return self.simulate(input_string)[self.accept_states].any()


class DAG:
    def __init__(self, nfa: NFA, n: int):
        self.transition_matrices = nfa.transition_matrices
        self.transition_matrices_backwards = nfa.transition_matrices_backwards
        self.alphabet = [0, 1]
        self.m = nfa.num_states
        self.n = n
        self.num_states = self.m * (n + 1)
        self.start_states = nfa.start_states
        self.accept_states = [self.m - 1]
        self.states = np.ones((self.n + 1, self.m), dtype=bool)
        self.states[0] = np.zeros(self.m, dtype=bool)
        self.states[0][self.start_states] = True
        self.states[self.n] = np.zeros(self.m, dtype=bool)
        self.states[self.n][-1] = True
        if n == 1:
            reachable = np.zeros(self.m, dtype=bool)
            if not (
                np.logical_and(
                    self.states[0].dot(self.transition_matrices[0]), self.states[-1]
                ).any()
                or np.logical_and(
                    self.states[0].dot(self.transition_matrices[1]), self.states[-1]
                ).any()
            ):
                # Set m to 0 since L^n is empty
                self.m = 0
        else:
            for i in range(1, self.n + 1):
                reachable = np.zeros(self.m, dtype=bool)
                accepting = np.zeros(self.m, dtype=bool)
                for transition_matrix in self.transition_matrices:
                    reachable = np.logical_or(
                        reachable, self.states[i - 1].dot(transition_matrix)
                    )
                for transition_matrix in self.transition_matrices_backwards:
                    accepting = np.logical_or(
                        accepting, self.states[self.n + 1 - i].dot(transition_matrix)
                    )
                if not np.any(reachable) or not np.any(accepting):
                    # Set m to 0 since L^n is empty
                    self.m = 0
                    break
                self.states[i] = np.logical_and(reachable, self.states[i])
                self.states[self.n - i] = np.logical_and(
                    accepting, self.states[self.n - i]
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
            current_states = current_states.dot(self.transition_matrices[symbol])
        return np.logical_and(current_states, self.states[len(input_string)])

    def simulate_one(self, states: np.ndarray, symbol: int, layer: int) -> np.ndarray:
        """
        Simulates the DAG on the given states with a single symbol and returns the next states.
        The input states are expected to be a numpy array of boolean values representing the current states.
        """
        if symbol not in self.alphabet:
            raise ValueError(f"Symbol {symbol} not in alphabet {self.alphabet}.")
        return np.logical_and(
            states.dot(self.transition_matrices[symbol]), self.states[layer + 1]
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
            current_states[mask_0] = current_states[mask_0].dot(
                self.transition_matrices[0]
            )
            current_states[mask_1] = current_states[mask_1].dot(
                self.transition_matrices[1]
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
