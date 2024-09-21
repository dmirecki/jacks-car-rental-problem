import random
from abc import abstractmethod
from collections import defaultdict
from copy import copy
from dataclasses import dataclass
from itertools import product
from typing import List

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

OUTSIDE_TRACK: int = 0
BOARD_PLACE: int = 1
START_LINE: int = 3
FINISH_LINE: int = 2
PLAYER: int = 4

BOARD_1 = np.array([
    [0, 0, 1, 1, 1, 1, 2],
    [0, 1, 1, 1, 1, 1, 2],
    [1, 1, 1, 1, 1, 1, 2],
    [1, 1, 1, 1, 1, 0, 0],
    [1, 1, 1, 1, 1, 0, 0],
    [0, 1, 1, 1, 1, 0, 0],
    [0, 1, 1, 1, 1, 0, 0],
    [0, 0, 3, 3, 3, 0, 0],
])

BOARD_2 = np.array([
    [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2],
    [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2],
    [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 2],
    [0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0],
    [0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0],
    [0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0],
    [0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
    [3, 3, 3, 3, 3, 3, 3, 3, 3, 0, 0],
])

BOARD: np.ndarray


@dataclass
class Action:
    velocity_change_horizontal: int = 0  # only -1, 0, 1
    velocity_change_vertical: int = 0  # only -1, 0, 1

    def __hash__(self):
        return hash((self.velocity_change_vertical, self.velocity_change_horizontal))


@dataclass
class State:
    velocity_horizontal: int = 0  # [0; 5)
    velocity_vertical: int = 0  # [0; 5)
    position_horizontal: int = -1
    position_vertical: int = -1
    finished: bool = False

    def __post_init__(self):
        self.reset()

    def __hash__(self):
        return hash(
            (self.velocity_horizontal, self.velocity_vertical, self.position_horizontal, self.position_vertical))

    def reset(self):
        self.initialize_random_position()
        self.velocity_horizontal = 0
        self.velocity_vertical = 0
        self.finished = False

    def initialize_random_position(self):
        start_line_positions = np.argwhere(BOARD == START_LINE)
        self.position_vertical, self.position_horizontal = \
            start_line_positions[np.random.randint(start_line_positions.shape[0]), :]

    def is_position_outside_track(self) -> bool:
        if self.position_vertical < 0 or self.position_horizontal < 0:
            return True

        if self.position_vertical >= BOARD.shape[0] or self.position_horizontal >= BOARD.shape[1]:
            return True

        return BOARD[self.position_vertical, self.position_horizontal] == OUTSIDE_TRACK

    def is_finish_position(self) -> bool:
        return BOARD[self.position_vertical, self.position_horizontal] == FINISH_LINE

    def set_finished(self):
        self.finished = True

    def plot(self):
        _b = BOARD.copy()
        _b[self.position_vertical, self.position_horizontal] = PLAYER
        plt.imshow(_b)
        plt.show()


class Policy:
    @abstractmethod
    def get_action(self, s: State, possible_actions: List[Action]) -> Action:
        ...


class TargetPolicy(Policy):
    def __init__(self):
        self.optimal_actions = defaultdict(lambda: Action())

    def get_action(self, s: State, possible_actions: List[Action]) -> Action:
        if s in self.optimal_actions:
            return self.optimal_actions[s]

        self.set_action(s, possible_actions[0])
        return self.optimal_actions[s]

    def set_action(self, state, action):
        self.optimal_actions[state] = action


class BehaviorPolicy(Policy):
    def __init__(self, target_policy: Policy, epsilon: float = 0.1) -> None:
        self.epsilon = epsilon
        self.target_policy = target_policy

        super().__init__()

    def get_action(self, s: State, possible_actions: List[Action]) -> Action:
        if random.random() < self.epsilon:
            return random.choice(possible_actions)

        return self.target_policy.get_action(s, possible_actions)

    def get_probability_of_choosing_action(self, action: Action, state: State, possible_actions: List[Action]) -> float:
        if action not in possible_actions:
            return 0

        if action == self.target_policy.get_action(state, possible_actions):
            return (1 - self.epsilon) + self.epsilon * len(possible_actions)

        return self.epsilon * len(possible_actions)


class Environment:
    def __init__(self, randomization_enabled: bool = True) -> None:
        self.state = State()
        self.randomization_enabled: bool = randomization_enabled

    def run_action(self, action) -> int:
        if self.randomization_enabled and (random.random() < 0.1):
            pass  # action is 0
        else:
            self.state.velocity_vertical += action.velocity_change_vertical
            self.state.velocity_horizontal += action.velocity_change_horizontal

        self.state.velocity_vertical = np.clip(self.state.velocity_vertical, 0, 4)
        self.state.velocity_horizontal = np.clip(self.state.velocity_horizontal, 0, 4)

        self._make_move()

        if not self.state.finished:
            return -1

        return 0

    def _make_move(self):
        # The assumption here is that the player is moving first vertical, then horizontal, not sidelong.
        for _ in range(self.state.velocity_vertical):
            self.state.position_vertical -= 1

            if self.state.is_position_outside_track():
                self.state.reset()
                return

            if self.state.is_finish_position():
                self.state.set_finished()
                return

        for _ in range(self.state.velocity_horizontal):
            self.state.position_horizontal += 1

            if self.state.is_position_outside_track():
                self.state.reset()
                return

            if self.state.is_finish_position():
                self.state.set_finished()
                return

    @staticmethod
    def get_possible_actions_in_state(state) -> List[Action]:
        possible_horizontal_actions = [0]
        if state.velocity_horizontal > 0:
            possible_horizontal_actions.append(-1)
        if state.velocity_horizontal < 4:
            possible_horizontal_actions.append(1)

        possible_vertical_actions = [0]
        if state.velocity_vertical > 0:
            possible_vertical_actions.append(-1)
        if state.velocity_vertical < 4:
            possible_vertical_actions.append(1)

        actions = [Action(h, v) for h, v in product(possible_horizontal_actions, possible_vertical_actions)]

        try:
            actions.remove(Action(-state.velocity_horizontal, -state.velocity_vertical))
        except ValueError:
            pass

        return actions


def generate_episode(env: Environment, policy: Policy, plot: bool = False) -> list[tuple[State, Action, float]]:
    rewards: List[float] = []
    actions: List[Action] = []
    states: List[State] = []

    env.state.reset()
    while not env.state.finished:
        states.append(copy(env.state))

        possible_actions = env.get_possible_actions_in_state(env.state)

        action = policy.get_action(env.state, possible_actions)
        reward = env.run_action(action)

        actions.append(action)
        rewards.append(reward)

        if plot:
            env.state.plot()

    return list(zip(states, actions, rewards))


# %%

BOARD = BOARD_2

def run_monte_carlo_control():
    GAMMA = 1
    env = Environment(False)
    C = defaultdict(lambda: defaultdict(int))
    Q = defaultdict(lambda: defaultdict(int))

    target_policy = TargetPolicy()
    behaviour_policy = BehaviorPolicy(target_policy, epsilon=0.1)

    history = []
    for _ in tqdm(range(5000)):
        episode = generate_episode(env, behaviour_policy)
        G = 0
        W = 1

        history.append((len(episode), sum(e[2] for e in episode)))

        for state, action, reward in reversed(episode):
            possible_actions = env.get_possible_actions_in_state(state)
            G = GAMMA * G + reward
            C[state][action] += W
            Q[state][action] += W / C[state][action] * (G - Q[state][action])
            target_policy.set_action(state, max(Q[state].keys(), key=Q[state].get))
            if action != target_policy.get_action(state, possible_actions):
                break
            W *= 1 / behaviour_policy.get_probability_of_choosing_action(action, state, possible_actions)

    return history, target_policy

history, target_policy = run_monte_carlo_control()

#%%

import pandas as pd

plt.plot(pd.Series(list(zip(*history))[1]).rolling(50).mean())
plt.show()

# %%

env = Environment(False)
generate_episode(env, target_policy, plot=True)
