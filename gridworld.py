import random
from collections import defaultdict
from copy import copy
from dataclasses import dataclass
from enum import Enum, unique, auto
from typing import List, Dict

import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import numpy as np
from tqdm import trange


@unique
class Action(Enum):
    UP = auto()
    DOWN = auto()
    LEFT = auto()
    RIGHT = auto()
    UP_LEFT = auto()
    UP_RIGHT = auto()
    DOWN_LEFT = auto()
    DOWN_RIGHT = auto()
    NONE = auto()


@dataclass
class State:
    position_horizontal: int = 0
    position_vertical: int = 0

    def __hash__(self):
        return hash((self.position_horizontal, self.position_vertical))


@dataclass
class Board:
    width: int
    height: int
    start_position_horizontal: int
    start_position_vertical: int
    goal_position_horizontal: int
    goal_position_vertical: int
    wind: np.ndarray
    allowed_actions: List[Action]
    is_stochastic: bool = False


class Policy:
    def __init__(self, epsilon: float):
        self.epsilon: float = epsilon
        self.Q: Dict[State, Dict[Action, float]] = defaultdict(lambda: defaultdict(float))

    def get_action(self, state: State, possible_actions: List[Action]) -> Action:
        if random.random() < self.epsilon:
            return random.choice(possible_actions)

        max_value = max(self.Q[state][a] for a in possible_actions)
        return random.choice([a for a in possible_actions if self.Q[state][a] == max_value])


class Environment:
    def __init__(self, board: Board):
        self.state = State(
            position_vertical=board.start_position_vertical,
            position_horizontal=board.start_position_horizontal
        )
        self.board = board

        assert len(self.board.wind) == self.board.width

    def is_in_terminal_state(self) -> bool:
        return (self.board.goal_position_horizontal == self.state.position_horizontal
                and self.board.goal_position_vertical == self.state.position_vertical)

    def run_action(self, action: Action):
        if action == Action.UP:
            self.state.position_vertical += 1
        elif action == Action.DOWN:
            self.state.position_vertical -= 1
        elif action == Action.LEFT:
            self.state.position_horizontal -= 1
        elif action == Action.RIGHT:
            self.state.position_horizontal += 1
        elif action == Action.UP_LEFT:
            self.state.position_vertical += 1
            self.state.position_horizontal -= 1
        elif action == Action.UP_RIGHT:
            self.state.position_vertical += 1
            self.state.position_horizontal += 1
        elif action == Action.DOWN_LEFT:
            self.state.position_vertical -= 1
            self.state.position_horizontal -= 1
        elif action == Action.DOWN_RIGHT:
            self.state.position_vertical -= 1
            self.state.position_horizontal += 1
        elif action == Action.NONE:
            pass
        else:
            raise ValueError(f'Invalid action: {action}')

        self.state.position_horizontal = np.clip(self.state.position_horizontal, 0, self.board.width - 1)
        self.state.position_vertical = np.clip(self.state.position_vertical, 0, self.board.height - 1)

        # Apply wind
        self.state.position_vertical += self.board.wind[self.state.position_horizontal]
        if self.board.is_stochastic and (self.board.wind[self.state.position_horizontal] > 0):
            randomness = np.random.choice([-1, 0, 1])
            self.state.position_vertical += randomness
        self.state.position_vertical = np.clip(self.state.position_vertical, 0, self.board.height - 1)

        return -1

    def get_possible_actions(self) -> List[Action]:
        actions = []

        can_go_right = self.state.position_horizontal < self.board.width - 1
        can_go_left = self.state.position_horizontal > 0
        can_go_up = self.state.position_vertical < self.board.height - 1
        can_go_down = self.state.position_vertical > 0

        if Action.NONE in self.board.allowed_actions:
            actions.append(Action.NONE)
        if can_go_right:
            actions.append(Action.RIGHT)
        if can_go_left:
            actions.append(Action.LEFT)
        if can_go_up:
            actions.append(Action.UP)
        if can_go_down:
            actions.append(Action.DOWN)
        if can_go_up and can_go_right and (Action.UP_RIGHT in self.board.allowed_actions):
            actions.append(Action.UP_RIGHT)
        if can_go_up and can_go_left and (Action.UP_LEFT in self.board.allowed_actions):
            actions.append(Action.UP_LEFT)
        if can_go_down and can_go_right and (Action.DOWN_RIGHT in self.board.allowed_actions):
            actions.append(Action.DOWN_RIGHT)
        if can_go_down and can_go_left and (Action.DOWN_LEFT in self.board.allowed_actions):
            actions.append(Action.DOWN_LEFT)

        return actions


def run_sarsa(board: Board, alpha: float = 0.5, delta: float = 1, max_episode_length: int = 500):
    policy = Policy(0.1)

    i = 0
    history = []
    for n_episode in trange(200):
        env = Environment(board)
        reward = 0

        state_0 = copy(env.state)
        action_0 = policy.get_action(env.state, possible_actions=env.get_possible_actions())
        for _ in range(max_episode_length):
            if env.is_in_terminal_state():
                break

            reward += env.run_action(action_0)
            state_1 = copy(env.state)
            action_1 = policy.get_action(env.state, possible_actions=env.get_possible_actions())

            policy.Q[state_0][action_0] += alpha * (
                    reward + delta * policy.Q[state_1][action_1]
                    - policy.Q[state_0][action_0]
            )

            action_0 = action_1
            state_0 = copy(state_1)

            history.append((i, n_episode))
            i += 1

    return history, policy


board = Board(
    width=10,
    height=7,
    start_position_vertical=3,
    start_position_horizontal=0,
    goal_position_horizontal=7,
    goal_position_vertical=3,
    wind=np.array([0, 0, 0, 1, 1, 1, 2, 2, 1, 0]),
    allowed_actions=[
        Action.UP,
        Action.DOWN,
        Action.LEFT,
        Action.RIGHT,
    ]
)


board_kings_moves = Board(
    width=10,
    height=7,
    start_position_vertical=3,
    start_position_horizontal=0,
    goal_position_horizontal=7,
    goal_position_vertical=3,
    wind=np.array([0, 0, 0, 1, 1, 1, 2, 2, 1, 0]),
    allowed_actions=[
        Action.UP,
        Action.DOWN,
        Action.LEFT,
        Action.RIGHT,
        Action.UP_RIGHT,
        Action.DOWN_RIGHT,
        Action.UP_LEFT,
        Action.DOWN_LEFT,
    ]
)


board_kings_moves_with_none = Board(
    width=10,
    height=7,
    start_position_vertical=3,
    start_position_horizontal=0,
    goal_position_horizontal=7,
    goal_position_vertical=3,
    wind=np.array([0, 0, 0, 1, 1, 1, 2, 2, 1, 0]),
    allowed_actions=[
        Action.UP,
        Action.DOWN,
        Action.LEFT,
        Action.RIGHT,
        Action.UP_RIGHT,
        Action.DOWN_RIGHT,
        Action.UP_LEFT,
        Action.DOWN_LEFT,
        Action.NONE
    ]
)


board_kings_moves_stochastic = Board(
    width=10,
    height=7,
    start_position_vertical=3,
    start_position_horizontal=0,
    goal_position_horizontal=7,
    goal_position_vertical=3,
    wind=np.array([0, 0, 0, 1, 1, 1, 2, 2, 1, 0]),
    allowed_actions=[
        Action.UP,
        Action.DOWN,
        Action.LEFT,
        Action.RIGHT,
        Action.UP_RIGHT,
        Action.DOWN_RIGHT,
        Action.UP_LEFT,
        Action.DOWN_LEFT,
    ],
    is_stochastic=True
)

history_standard_board, policy_standard_board = run_sarsa(board)
history_kings_moves, policy_kings_moves = run_sarsa(board_kings_moves)
history_kings_moves_with_none, policy_kings_moves_with_none = run_sarsa(board_kings_moves_with_none)
history_kings_moves_stochastic, policy_kings_moves_stochastic = run_sarsa(board_kings_moves_stochastic)

fig = px.line(pd.DataFrame(history_standard_board, columns=['iteration', 'episode']), x='iteration', y='episode')
fig.show()

fig = px.line(pd.DataFrame(history_kings_moves, columns=['iteration', 'episode']), x='iteration', y='episode')
fig.show()

fig = px.line(pd.DataFrame(history_kings_moves_with_none, columns=['iteration', 'episode']), x='iteration', y='episode')
fig.show()

fig = px.line(pd.DataFrame(history_kings_moves_stochastic, columns=['iteration', 'episode']), x='iteration', y='episode')
fig.show()

def plot(env: Environment):
    _b = np.zeros((env.board.height, env.board.width))
    _b[env.board.height - 1 - env.board.goal_position_vertical, env.board.goal_position_horizontal] = 1
    _b[env.board.height - 1 - env.board.start_position_vertical, env.board.start_position_horizontal] = 2
    _b[env.board.height - 1 - env.state.position_vertical, env.state.position_horizontal] = 3
    plt.imshow(_b)
    plt.show()


def plot_game(b, p):
    env = Environment(b)
    i = 0
    while not env.is_in_terminal_state():
        action = p.get_action(env.state, possible_actions=env.get_possible_actions())
        env.run_action(action)

        plot(env)
        i += 1
    print(i)


plot_game(board, policy_standard_board)
plot_game(board_kings_moves, policy_kings_moves)
plot_game(board_kings_moves_with_none, policy_kings_moves_with_none)
plot_game(board_kings_moves_stochastic, policy_kings_moves_stochastic)