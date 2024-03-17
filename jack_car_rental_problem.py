import itertools
from copy import deepcopy
from typing import Optional, List
from dataclasses import dataclass
from tqdm import tqdm

import scipy
import numpy as np

EVALUATE_POLICY_EPSILON = 0.01

DISCOUNT_RATE = 0.9
DAY_RENTING_INCOME = 10
MOVING_CAR_REWARD = -2
EXTRA_PARKING_REWARD = -4
MAX_CARS = 20
MAX_NUMBER_OF_MOVING_CARS = 5
LAMBDA_RENTAL_REQUEST_FIRST_LOCATION = 3
LAMBDA_RENTAL_REQUEST_SECOND_LOCATION = 4
LAMBDA_RETURN_FIRST_LOCATION = 3
LAMBDA_RETURN_SECOND_LOCATION = 2


@dataclass
class State:
    cars_in_first_location: int
    cars_in_second_location: int


@dataclass
class Action:
    # Number of cars to move from first location to second location.
    cars_to_move: int = 0


class Policy:
    def __init__(self):
        self._matrix = np.zeros((MAX_CARS + 1, MAX_CARS + 1), int)

    def get_action(self, s: State) -> Action:
        # noinspection PyTypeChecker
        return Action(cars_to_move=self._matrix[s.cars_in_first_location, s.cars_in_second_location])

    def set_action(self, state: State, action: Action):
        self._matrix[state.cars_in_first_location, state.cars_in_second_location] = action.cars_to_move

    def __eq__(self, other: 'Policy'):
        return np.all(np.equal(self._matrix, other._matrix))

    def get_matrix(self):
        return self._matrix


class ValueFunction:
    def __init__(self, matrix=None):
        self._matrix = matrix

        if self._matrix is None:
            self._matrix = np.zeros((MAX_CARS + 1, MAX_CARS + 1), float)

    # noinspection PyTypeChecker
    def get_value(self, state: State, action: Optional[Action] = None) -> float:
        if action is None:
            return self._matrix[state.cars_in_first_location, state.cars_in_second_location]

        return self._matrix[
            min(state.cars_in_first_location - action.cars_to_move, MAX_CARS),
            min(state.cars_in_second_location + action.cars_to_move, MAX_CARS)
        ]

    def set_value(self, state: State, value: float):
        self._matrix[state.cars_in_first_location, state.cars_in_second_location] = value

    def get_diff(self, other: 'ValueFunction') -> float:
        return np.abs(self.get_matrix() - other.get_matrix()).sum()

    def get_matrix(self):
        return self._matrix

    def copy(self):
        return ValueFunction(self._matrix.copy())


class EnvironmentModelJacksCarRental:
    def __init__(self):
        self.renting_income = self._calculate_renting_income()

        self.first_location_rental_request_prob = self._calculate_rental_requests_probability(
            LAMBDA_RENTAL_REQUEST_FIRST_LOCATION
        )
        self.second_location_rental_request_prob = self._calculate_rental_requests_probability(
            LAMBDA_RENTAL_REQUEST_SECOND_LOCATION
        )

        self.first_location_return_prob = self._calculate_return_probability(LAMBDA_RETURN_FIRST_LOCATION)
        self.second_location_return_prob = self._calculate_return_probability(LAMBDA_RETURN_SECOND_LOCATION)

        self.first_location_state_change_prob = self.first_location_rental_request_prob @ self.first_location_return_prob
        self.second_location_state_change_prob = self.second_location_rental_request_prob @ self.second_location_return_prob

        assert np.allclose(self.first_location_state_change_prob.sum(axis=1), 1)
        assert np.allclose(self.second_location_state_change_prob.sum(axis=1), 1)

    @staticmethod
    def _calculate_renting_income():
        """
        Returns matrix N x N where the element [n, m] is the renting income if
        there will be m cars in a location when there were n cars before the "renting stage".
        """

        renting_income = np.zeros(shape=(MAX_CARS + 1, MAX_CARS + 1), dtype=np.float32)

        for were_cars in range(MAX_CARS + 1):
            for will_be_cars in range(MAX_CARS + 1):
                renting_income[were_cars, will_be_cars] = max(0, were_cars - will_be_cars) * DAY_RENTING_INCOME

        return renting_income

    @staticmethod
    def _calculate_return_probability(mu):
        """
        Returns matrix N x N where the element [n, m] is the probability that
        there will be m cars in a location when there were n cars before the "returning stage".
        """

        returned_cars_prob = np.zeros(shape=(MAX_CARS + 1, MAX_CARS + 1), dtype=np.float32)

        for were_cars in range(MAX_CARS + 1):
            for will_be_cars in range(were_cars, MAX_CARS):
                number_of_returns = will_be_cars - were_cars
                returned_cars_prob[were_cars, will_be_cars] = scipy.stats.poisson.pmf(mu=mu, k=number_of_returns)

        # The last column absorbs the situation when that are more returns that possible cars in a location.
        returned_cars_prob[:, -1] = 1 - returned_cars_prob.sum(axis=1)

        return returned_cars_prob

    @staticmethod
    def _calculate_rental_requests_probability(mu):
        """
        Returns matrix N x N where the element [n, m] is the probability that
        there will be m cars in a location when there were n cars in a location before the "renting stage".
        """

        rent_cars_prob = np.zeros(shape=(MAX_CARS + 1, MAX_CARS + 1), dtype=np.float32)

        for were_cars in range(MAX_CARS + 1):
            for will_be_cars in range(1, were_cars + 1):
                number_of_clients = max(0, were_cars - will_be_cars)
                rent_cars_prob[were_cars, will_be_cars] = scipy.stats.poisson.pmf(mu=mu, k=number_of_clients)

        # The first column absorbs the situation when there are more clients than available cars.
        rent_cars_prob[:, 0] = 1 - np.sum(rent_cars_prob, axis=1)

        return rent_cars_prob

    def calculate_expected_value(self, state: State, action: Action, value_function: ValueFunction) -> float:
        cars_in_first_location_after_action = min(state.cars_in_first_location - action.cars_to_move, MAX_CARS)
        cars_in_second_location_after_action = min(state.cars_in_second_location + action.cars_to_move, MAX_CARS)

        assert 0 <= cars_in_first_location_after_action <= MAX_CARS
        assert 0 <= cars_in_second_location_after_action <= MAX_CARS

        new_state_probability_matrix = np.outer(
            self.first_location_state_change_prob[cars_in_first_location_after_action, :],
            self.second_location_state_change_prob[cars_in_second_location_after_action, :]
        )

        expected_renting_reward_first_location = np.multiply(
            self.first_location_rental_request_prob[cars_in_first_location_after_action, :],
            self.renting_income[cars_in_first_location_after_action, :]
        ).sum()

        expected_renting_reward_second_location = np.multiply(
            self.second_location_rental_request_prob[cars_in_second_location_after_action, :],
            self.renting_income[cars_in_second_location_after_action, :]
        ).sum()

        expected_value = \
            abs(action.cars_to_move) * MOVING_CAR_REWARD \
            + expected_renting_reward_first_location \
            + expected_renting_reward_second_location \
            + DISCOUNT_RATE * np.multiply(new_state_probability_matrix, value_function.get_matrix()).sum()

        return expected_value

    @staticmethod
    def get_states_space() -> List[State]:
        states_params = itertools.product(range(MAX_CARS + 1), repeat=2)

        all_states = list(
            (State(cars_in_first_location, cars_in_second_location)
             for cars_in_first_location, cars_in_second_location in states_params)
        )

        return list(all_states)

    @staticmethod
    def get_possible_actions_in_state(state: State) -> List[Action]:
        possible_actions = [
            Action(cars_to_move=c)
            for c in range(-min(MAX_NUMBER_OF_MOVING_CARS, state.cars_in_second_location),
                           min(MAX_NUMBER_OF_MOVING_CARS, state.cars_in_first_location) + 1)
        ]

        return possible_actions


class EnvironmentModelJacksCarRentalModifiedProblem(EnvironmentModelJacksCarRental):
    def calculate_expected_value(self, state: State, action: Action, value_function: ValueFunction) -> float:
        cars_in_first_location_after_action = min(state.cars_in_first_location - action.cars_to_move, MAX_CARS)
        cars_in_second_location_after_action = min(state.cars_in_second_location + action.cars_to_move, MAX_CARS)

        assert 0 <= cars_in_first_location_after_action <= MAX_CARS
        assert 0 <= cars_in_second_location_after_action <= MAX_CARS

        new_state_probability_matrix = np.outer(
            self.first_location_state_change_prob[cars_in_first_location_after_action, :],
            self.second_location_state_change_prob[cars_in_second_location_after_action, :]
        )

        expected_renting_reward_first_location = np.multiply(
            self.first_location_rental_request_prob[cars_in_first_location_after_action, :],
            self.renting_income[cars_in_first_location_after_action, :]
        ).sum()

        expected_renting_reward_second_location = np.multiply(
            self.second_location_rental_request_prob[cars_in_second_location_after_action, :],
            self.renting_income[cars_in_second_location_after_action, :]
        ).sum()

        if action.cars_to_move >= 1:
            moving_cars_reward = (action.cars_to_move - 1) * MOVING_CAR_REWARD
        else:
            moving_cars_reward = abs(action.cars_to_move) * MOVING_CAR_REWARD

        if cars_in_first_location_after_action > 10:
            moving_cars_reward += EXTRA_PARKING_REWARD

        if cars_in_second_location_after_action > 10:
            moving_cars_reward += EXTRA_PARKING_REWARD

        expected_value = \
            moving_cars_reward \
            + expected_renting_reward_first_location \
            + expected_renting_reward_second_location \
            + DISCOUNT_RATE * np.multiply(new_state_probability_matrix, value_function.get_matrix()).sum()

        return expected_value




def evaluate_policy(policy: Policy, env) -> ValueFunction:
    value_function_old = ValueFunction()
    value_function_new = ValueFunction()

    all_states = env.get_states_space()

    with tqdm(desc='Evaluating policy') as pbar:
        while True:
            for state in all_states:
                action = policy.get_action(state)

                expected_value = env.calculate_expected_value(state, action, value_function_new)
                value_function_new.set_value(state, expected_value)

            if value_function_old.get_diff(value_function_new) < EVALUATE_POLICY_EPSILON:
                break

            value_function_old = value_function_new
            value_function_new = value_function_new.copy()

            pbar.update()

    return value_function_new


def improve_policy(current_policy: Policy, value_function: ValueFunction, env) -> Policy:
    new_policy = deepcopy(current_policy)
    for state in env.get_states_space():
        possible_actions = env.get_possible_actions_in_state(state)

        actions_with_values = [
            (action, env.calculate_expected_value(state, action, value_function))
            for action in possible_actions
        ]
        best_action = max(actions_with_values, key=lambda x: x[1])[0]

        new_policy.set_action(state, best_action)

    return new_policy


def policy_iteration(env):
    policy = Policy()

    print(f'Policy no. 0:\n {np.flip(policy.get_matrix(), axis=0)}')
    i = 1
    while True:
        value_function = evaluate_policy(policy, env)
        new_policy = improve_policy(policy, value_function, env)

        print(f'Policy no. {i}:\n {np.flip(new_policy.get_matrix(), axis=0)}')
        if new_policy == policy:
            break

        policy = new_policy
        i += 1


policy_iteration(EnvironmentModelJacksCarRental())
policy_iteration(EnvironmentModelJacksCarRentalModifiedProblem())
