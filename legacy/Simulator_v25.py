## deep net bi-lstm + taxis + Q-lstm ## dopamine + self model ## category trade ## v25.3 ##pacman ## negative reward % ## multi states ## rerun with viz

import os
import numpy as np
import pygame
import random
import math
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Bidirectional
from tensorflow.keras.optimizers import Adam
import warnings
import joblib

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Constants
GRID_SIZE = 22
CELL_SIZE = 20
SCREEN_SIZE = GRID_SIZE * CELL_SIZE
TIME_STEPS = 2000
N_AGENTS = 2
COLORS = ['cyan', 'yellow']
CONSTANT_FOOD_COUNT = 50
FOOD_ENERGY_RANGE = (100, 200)
MAX_ENERGY = 1000
ENERGY_LOSS_MOVE = 1.0
ENERGY_LOSS_STATIONARY = 0.1
ENERGY_GAIN_FOOD = 20
PERCEPTION_WINDOW = 180
PERCEPTION_RANGE = 8
COLLISION_PENALTY = 1.0
M3_SEQUENCE_LENGTH = 50
GAMMA = 0.99

# Maze layout (1 for walls, 0 for paths)
MAZE = np.array([
    [1,1,1,1,1,1,1,1,1,1,1,1,0,1,1,1,1,1,1,1,1,1],
    [1,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,1],
    [1,0,1,1,1,1,0,0,0,0,0,0,0,1,1,0,1,1,1,1,0,1],
    [1,0,1,1,1,1,0,1,1,1,1,1,0,1,1,0,1,1,1,1,0,1],
    [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1],
    [1,0,1,1,1,1,0,1,1,0,1,1,1,1,1,1,1,1,0,1,0,1],
    [1,0,0,0,0,0,0,1,1,0,0,0,0,1,1,0,0,0,0,1,0,1],
    [1,1,1,1,1,0,0,1,1,1,1,1,0,1,1,0,1,1,1,1,0,1],
    [1,1,1,1,1,0,0,1,1,0,0,0,0,0,0,0,0,0,0,1,0,1],
    [1,1,1,1,1,0,0,1,1,0,0,0,0,0,0,0,0,0,0,1,0,1],
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [1,1,1,1,1,0,0,1,1,0,0,0,0,0,0,0,0,0,0,1,0,1],
    [1,1,1,1,1,0,0,1,1,0,0,0,0,0,0,0,0,0,0,1,0,1],
    [1,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,0,0,0,1],
    [1,0,1,1,1,0,0,1,1,1,1,1,0,1,1,0,1,1,1,1,0,1],
    [1,0,0,0,1,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,1],
    [1,1,1,0,1,0,0,1,1,0,1,1,1,1,1,1,1,1,0,1,0,1],
    [1,0,0,0,0,0,0,1,1,0,0,0,0,1,1,0,0,0,0,1,0,1],
    [1,0,1,1,1,1,0,1,1,1,1,1,0,1,1,0,1,1,1,1,0,1],
    [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1],
    [1,1,1,1,1,1,1,1,1,1,1,1,0,1,1,1,1,1,1,1,1,1]
])

class SharedCategorySpace:
    def __init__(self):
        self.shared_categories = {}

    def update(self, agent_id, categories):
        self.shared_categories[agent_id] = categories

    def get_shared_categories(self):
        return self.shared_categories

class Agent:
    def __init__(self, x, y, color):
        self.pos = np.array([x, y])
        self.energy = MAX_ENERGY / 2
        self.initial_energy = self.energy
        self.previous_energy = self.energy
        self.angle = random.uniform(0, 360)
        self.color = color
        self.energy_tracking = [0] * 10
        self.movement_tracking = [np.zeros(2)] * 10
        self.angle_tracking = [0] * 10
        self.perception = np.zeros((PERCEPTION_RANGE, PERCEPTION_WINDOW))
        self.raw_perception = np.zeros((PERCEPTION_RANGE, PERCEPTION_WINDOW))
        self.previous_raw_perception = np.zeros((PERCEPTION_RANGE, PERCEPTION_WINDOW))
        self.energy_gradient = np.zeros(360)
        self.raw_energy_gradient = np.zeros(360)
        self.predicted_energy_gradient = np.zeros(360)
        self.generated_perception_M0 = np.zeros((PERCEPTION_RANGE, PERCEPTION_WINDOW))
        self.future_perception = np.zeros((PERCEPTION_RANGE, PERCEPTION_WINDOW))
        
        self.M0 = self._create_bidirectional_lstm_model((10, PERCEPTION_RANGE * PERCEPTION_WINDOW), PERCEPTION_RANGE * PERCEPTION_WINDOW)
        total_input_size = (PERCEPTION_RANGE * PERCEPTION_WINDOW) + 360 + (2 * 10) + 10 + 10
        self.M1 = self._create_bidirectional_lstm_model((10, total_input_size), PERCEPTION_RANGE * PERCEPTION_WINDOW)
        m2_input_size = PERCEPTION_RANGE * PERCEPTION_WINDOW * 2
        self.M2 = self._create_bidirectional_lstm_model((10, m2_input_size), 360)
        
        self.M0_errors = []
        self.M1_errors = []
        self.M2_errors = []
        self.M3_errors = []
        
        self.categories = None
        self.received_categories = None
        
        self.rotation_speed = 60
        self.locomotion_steps = 0
        self.max_locomotion_steps = 10
        self.min_observation_steps = 1
        self.observation_steps = 0

        self.energy_loss_move = ENERGY_LOSS_MOVE
        self.energy_loss_stationary = ENERGY_LOSS_STATIONARY
        self.energy_gain_food = ENERGY_GAIN_FOOD

        self.stuck_counter = 0
        self.last_position = self.pos.copy()
        self.is_randomized = False

        self.step_count = 0
        self.cumulative_efficiency = 0

        self.M3_sequence_length = M3_SEQUENCE_LENGTH
        self.M3_state_size = PERCEPTION_RANGE * PERCEPTION_WINDOW + 360 + 1
        self.M3_action_size = 20
        self.M3 = self._create_bidirectional_lstm_q_model()
        self.M3_sequence = [np.zeros(self.M3_state_size)] * self.M3_sequence_length

        self.M0_sequence = []
        self.M1_sequence = []
        self.M2_sequence = []

        self.last_action_M3 = None
        self.gradient_choices = [0] * self.M3_action_size
        
        self.epsilon = 0.5
        self.epsilon_min = 0.1
        self.epsilon_max = 1.0
        self.epsilon_learning_rate = 0.01
        self.energy_window = []
        self.energy_window_size = 10
        self.min_window_size = 5
        self.max_window_size = 50
        self.epsilon_model = self._create_epsilon_model()
        
        self.window_size_model = self._create_window_size_model()
        self.loss_window = []
        self.loss_window_size = 10

        self.efficiency_score = 0
        self.min_efficiency = float('inf')
        self.max_efficiency = float('-inf')

    def _create_bidirectional_lstm_model(self, input_shape, output_size):
        model = Sequential([
            Bidirectional(LSTM(16, return_sequences=True), input_shape=input_shape),
            Bidirectional(LSTM(16, return_sequences=True)),
            Bidirectional(LSTM(16)),
            Dense(output_size)
        ])
        model.compile(optimizer=Adam(learning_rate=0.05), loss='mse')
        return model

    def _create_bidirectional_lstm_q_model(self):
        model = Sequential([
            Bidirectional(LSTM(128, return_sequences=True, input_shape=(self.M3_sequence_length, self.M3_state_size))),
            Bidirectional(LSTM(64, return_sequences=True)),
            Bidirectional(LSTM(32, return_sequences=True)),
            Bidirectional(LSTM(16)),
            Dense(self.M3_action_size)
        ])
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
        return model

    def _create_epsilon_model(self):
        model = Sequential([
            Dense(32, activation='relu', input_shape=(2,)),
            Dense(16, activation='relu'),
            Dense(1, activation='sigmoid')
        ])
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
        return model

    def _create_window_size_model(self):
        model = Sequential([
            Dense(16, activation='relu', input_shape=(3,)),
            Dense(16, activation='relu'),
            Dense(1, activation='linear')
        ])
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
        return model

    def _get_m3_state(self):
        return np.concatenate([
            self.perception.flatten(),
            self.energy_gradient,
            [self.energy]
        ])

    def perceive(self, food_positions, agents, shared_category_space):
        self.previous_raw_perception = self.raw_perception.copy()
        self.raw_perception = np.zeros((PERCEPTION_RANGE, PERCEPTION_WINDOW))
        self.perception = np.zeros((PERCEPTION_RANGE, PERCEPTION_WINDOW))
        self.received_categories = np.zeros((PERCEPTION_RANGE, PERCEPTION_WINDOW))
        
        for angle in range(PERCEPTION_WINDOW):
            for distance in range(1, PERCEPTION_RANGE + 1):
                x = int(self.pos[0] + distance * math.cos(math.radians(self.angle + angle - PERCEPTION_WINDOW/2)))
                y = int(self.pos[1] + distance * math.sin(math.radians(self.angle + angle - PERCEPTION_WINDOW/2)))
                x, y = x % GRID_SIZE, y % GRID_SIZE
                
                if MAZE[y, x] == 1:
                    self.raw_perception[distance-1, angle] = -1
                    self.perception[distance-1, angle] = -1
                    break
                
                if (x, y) in food_positions:
                    self.raw_perception[distance-1, angle] = food_positions[(x, y)] / distance
                    self.perception[distance-1, angle] = food_positions[(x, y)] / distance
                
                for agent in agents:
                    if agent != self and np.array_equal(agent.pos, [x, y]):
                        self.raw_perception[distance-1, angle] = -agent.energy / distance
                        self.perception[distance-1, angle] = -agent.energy / distance
                        if agent.color in shared_category_space.get_shared_categories():
                            self.received_categories[distance-1, angle] = shared_category_space.get_shared_categories()[agent.color][distance-1 * PERCEPTION_WINDOW + angle]
        
        self.raw_energy_gradient = self._calculate_energy_gradient(food_positions)
        if not self.is_randomized:
            self.energy_gradient = self.raw_energy_gradient.copy()

        self.M0_sequence.append(self.perception.flatten())
        if len(self.M0_sequence) > 10:
            self.M0_sequence.pop(0)

        if len(self.M0_sequence) == 10:
            X_M0 = np.array(self.M0_sequence).reshape(1, 10, -1)
            self.categories = self.M0.predict(X_M0, verbose=0)[0]
        else:
            self.categories = np.zeros(PERCEPTION_RANGE * PERCEPTION_WINDOW)

        shared_category_space.update(self.color, self.categories)

        M1_input = np.hstack((
            self.raw_perception.flatten(),
            self.raw_energy_gradient,
            np.array(self.movement_tracking).flatten(),
            np.array(self.energy_tracking),
            np.array(self.angle_tracking)
        ))
        self.M1_sequence.append(M1_input)
        if len(self.M1_sequence) > 10:
            self.M1_sequence.pop(0)

        if len(self.M1_sequence) == 10:
            X_M1 = np.array(self.M1_sequence).reshape(1, 10, -1)
            self.future_perception = self.M1.predict(X_M1, verbose=0)[0]
        else:
            self.future_perception = np.zeros(PERCEPTION_RANGE * PERCEPTION_WINDOW)

        M2_input = np.hstack((self.raw_perception.flatten(), self.future_perception))
        self.M2_sequence.append(M2_input)
        if len(self.M2_sequence) > 10:
            self.M2_sequence.pop(0)

        if len(self.M2_sequence) == 10:
            X_M2 = np.array(self.M2_sequence).reshape(1, 10, -1)
            self.predicted_energy_gradient = self.M2.predict(X_M2, verbose=0)[0]
        else:
            self.predicted_energy_gradient = np.zeros(360)

        m3_state = self._get_m3_state()
        self.M3_sequence.append(m3_state)
        if len(self.M3_sequence) > self.M3_sequence_length:
            self.M3_sequence.pop(0)

        return self.perception, self.future_perception

    def _calculate_energy_gradient(self, food_positions):
        energy_gradient = np.zeros(360)
        for angle in range(360):
            energy = 0
            for distance in range(1, PERCEPTION_RANGE + 1):
                x = int(self.pos[0] + distance * math.cos(math.radians(angle)))
                y = int(self.pos[1] + distance * math.sin(math.radians(angle)))
                x, y = x % GRID_SIZE, y % GRID_SIZE
                
                if MAZE[y, x] == 1:
                    break
                
                if (x, y) in food_positions:
                    energy += food_positions[(x, y)] / distance
            
            energy_gradient[angle] = energy
        return energy_gradient

    def move(self, food_positions, agents, shared_category_space):
        self.step_count += 1
        previous_pos = np.array(self.pos)
        previous_angle = self.angle
        self.previous_energy = self.energy

        self.perceive(food_positions, agents, shared_category_space)

        action_M3 = self._choose_action()

        if self.locomotion_steps >= self.max_locomotion_steps:
            energy_expenditure = self._pause_and_observe()
            self.observation_steps += 1
            if self.observation_steps >= self.min_observation_steps:
                self.locomotion_steps = 0
        else:
            self.locomotion_steps += 1
            self.observation_steps = 0
            
            energy_expenditure = self._locomotion_taxis(self.energy_gradient)
            self.gradient_choices[action_M3] += 1

        if tuple(self.pos) in food_positions:
            energy_gain = self.energy_gain_food
            self.energy += energy_gain
            del food_positions[tuple(self.pos)]
            if self.is_randomized:
                self.is_randomized = False
                self.stuck_counter = 0

        self.energy -= energy_expenditure

        reward = self._calculate_reward()

        movement = self.pos - previous_pos
        angle_change = (self.angle - previous_angle + 180) % 360 - 180

        self.track_energy_movement_angle(energy_expenditure, movement, angle_change)

        self.last_action_M3 = action_M3

        if np.array_equal(self.pos, self.last_position):
            self.stuck_counter += 1
            if self.stuck_counter >= 20:
                self._randomize_energy_gradient()
                self.is_randomized = True
        else:
            if not self.is_randomized:
                self.stuck_counter = 0

        self.last_position = self.pos.copy()

        self.update_efficiency_score()
        self.update_efficiency_bounds()
        self.epsilon = self.generate_epsilon()

    def _choose_action(self):
        if random.random() < self.epsilon:
            return random.randint(0, self.M3_action_size - 1)
        else:
            if len(self.M3_sequence) == self.M3_sequence_length:
                m3_input = np.array(self.M3_sequence).reshape(1, self.M3_sequence_length, self.M3_state_size)
                q_values = self.M3.predict(m3_input, verbose=0)[0]
                return np.argmax(q_values)
            else:
                return random.randint(0, self.M3_action_size - 1)

    def _locomotion_taxis(self, energy_gradient):
        m3_input = np.array(self.M3_sequence).reshape(1, self.M3_sequence_length, self.M3_state_size)
        q_values = self.M3.predict(m3_input, verbose=0)[0]
        
        normalized_q_values = (q_values - np.min(q_values)) / (np.max(q_values) - np.min(q_values) + 1e-8)
        
        random_gradient = np.random.rand(360)
        
        combined_gradient = (
            energy_gradient * (1 - self.epsilon) * 0.4 +
            random_gradient * self.epsilon * 0.6 +
            self._q_values_to_gradient(normalized_q_values) * (1 - self.epsilon) * 0.6
        )
        
        target_angle = np.argmax(combined_gradient)
        
        angle_diff = (target_angle - self.angle + 180) % 360 - 180
        adjustment = min(abs(angle_diff), self.rotation_speed) * np.sign(angle_diff)
        self.angle = (self.angle + adjustment) % 360
        
        move_x = int(round(math.cos(math.radians(self.angle))))
        move_y = int(round(math.sin(math.radians(self.angle))))
        
        return self._apply_movement(move_x, move_y)

    def _q_values_to_gradient(self, q_values):
        gradient = np.zeros(360)
        for i, q in enumerate(q_values):
            angle = (i / len(q_values)) * 360
            gradient[int(angle)] = q
        return gradient

    def _pause_and_observe(self):
        target_angle = np.argmax(self.energy_gradient)
        angle_diff = (target_angle - self.angle + 180) % 360 - 180
        adjustment = min(abs(angle_diff), self.rotation_speed) * np.sign(angle_diff)
        self.angle = (self.angle + adjustment) % 360
        return self.energy_loss_stationary

    def _apply_movement(self, move_x, move_y):
        energy_expenditure = 0
        steps = max(abs(move_x), abs(move_y))
        for _ in range(steps):
            step_x = np.sign(move_x) if move_x != 0 else 0
            step_y = np.sign(move_y) if move_y != 0 else 0
            new_x = int((self.pos[0] + step_x) % GRID_SIZE)
            new_y = int((self.pos[1] + step_y) % GRID_SIZE)
            if MAZE[new_y, new_x] == 0:  # Check if the new position is not a wall
                self.pos = np.array([new_x, new_y])
                energy_expenditure += self.energy_loss_move
                move_x -= step_x
                move_y -= step_y
            else:
                # If it's a wall, add collision penalty and stop movement
                energy_expenditure += self.energy_loss_stationary + COLLISION_PENALTY
                break
        
        return energy_expenditure

    def track_energy_movement_angle(self, energy_expenditure, movement, angle_change):
        self.energy_tracking.append(-energy_expenditure)  # Negative value for expenditure
        self.energy_tracking = self.energy_tracking[-10:]
        
        self.movement_tracking.append(movement)
        self.movement_tracking = self.movement_tracking[-10:]
        
        self.angle_tracking.append(angle_change)
        self.angle_tracking = self.angle_tracking[-10:]

    def _randomize_energy_gradient(self):
        self.energy_gradient = np.random.rand(360)
        self.predicted_energy_gradient = self.energy_gradient.copy()

    def update_efficiency_score(self):
        energy_change = self.energy - self.initial_energy
        self.efficiency_score = energy_change / max(1, self.step_count)

    def get_efficiency_score(self):
        return self.efficiency_score

    def update_efficiency_bounds(self):
        self.min_efficiency = min(self.min_efficiency, self.efficiency_score)
        self.max_efficiency = max(self.max_efficiency, self.efficiency_score)

    def generate_epsilon(self):
        normalized_score = (self.efficiency_score - self.min_efficiency) / (self.max_efficiency - self.min_efficiency + 1e-8)
        new_epsilon = 1 - normalized_score
        return max(self.epsilon_min, min(self.epsilon_max, new_epsilon))

    def optimize_epsilon(self, food_positions, agents, shared_category_space):
        current_efficiency = self.get_efficiency_score()
        
        # Try slightly higher epsilon
        original_epsilon = self.epsilon
        self.epsilon = min(self.epsilon * 1.1, self.epsilon_max)
        self.move(food_positions, agents, shared_category_space)
        higher_efficiency = self.get_efficiency_score()
        
        # Reset and try slightly lower epsilon
        self.epsilon = original_epsilon
        self.move(food_positions, agents, shared_category_space)  # Reset state
        self.epsilon = max(self.epsilon * 0.9, self.epsilon_min)
        self.move(food_positions, agents, shared_category_space)
        lower_efficiency = self.get_efficiency_score()
        
        # Choose the best epsilon
        if higher_efficiency > current_efficiency and higher_efficiency > lower_efficiency:
            self.epsilon = min(self.epsilon * 1.1, self.epsilon_max)
        elif lower_efficiency > current_efficiency and lower_efficiency > higher_efficiency:
            self.epsilon = max(self.epsilon * 0.9, self.epsilon_min)
        else:
            self.epsilon = original_epsilon

    def train_models(self, food_positions, agents, shared_category_space):
        current_losses = []

        if len(self.M0_sequence) == 10:
            X_M0 = np.array(self.M0_sequence).reshape(1, 10, PERCEPTION_RANGE * PERCEPTION_WINDOW)
            y_M0 = self.received_categories.flatten().reshape(1, -1)
            history = self.M0.fit(X_M0, y_M0, epochs=1, verbose=0)
            self.M0_errors.append(history.history['loss'][0])
            current_losses.append(history.history['loss'][0])

        if len(self.M1_sequence) == 10:
            X_M1 = np.array(self.M1_sequence).reshape(1, 10, -1)
            y_M1 = self.raw_perception.flatten().reshape(1, -1)
            history = self.M1.fit(X_M1, y_M1, epochs=1, verbose=0)
            self.M1_errors.append(history.history['loss'][0])
            current_losses.append(history.history['loss'][0])

        if len(self.M2_sequence) == 10:
            X_M2 = np.array(self.M2_sequence).reshape(1, 10, -1)
            y_M2 = self.raw_energy_gradient.reshape(1, -1)
            history = self.M2.fit(X_M2, y_M2, epochs=1, verbose=0)
            self.M2_errors.append(history.history['loss'][0])
            current_losses.append(history.history['loss'][0])

        if len(self.M3_sequence) == self.M3_sequence_length:
            m3_input = np.array(self.M3_sequence).reshape(1, self.M3_sequence_length, self.M3_state_size)
            m3_target = self.M3.predict(m3_input, verbose=0)
            
            reward = self._calculate_reward()
            next_m3_input = np.array(self.M3_sequence[1:] + [self.M3_sequence[-1]]).reshape(1, self.M3_sequence_length, self.M3_state_size)
            next_q_values = self.M3.predict(next_m3_input, verbose=0)[0]
            
            m3_target[0, self.last_action_M3] = reward + GAMMA * np.max(next_q_values)
            
            history = self.M3.fit(m3_input, m3_target, epochs=1, verbose=0)
            self.M3_errors.append(history.history['loss'][0])

        # Update loss window and train window size model
        if len(current_losses) == 3:  # We have losses for M0, M1, and M2
            self.loss_window.append(current_losses)
            if len(self.loss_window) > self.loss_window_size:
                self.loss_window.pop(0)

            self._update_energy_window_size()
            self._train_window_size_model()

        # Train epsilon model
        self._train_epsilon_model()

    def _calculate_reward(self):
        energy_change = self.energy - self.previous_energy
        reward = energy_change * 0.2
        if energy_change > 0:
            reward += 2
        elif energy_change < 0:
            reward -= 0.5
        
        return reward

    def _update_energy_window_size(self):
        if len(self.loss_window) < self.loss_window_size:
            return

        loss_changes = np.diff(self.loss_window, axis=0)
        avg_loss_changes = np.mean(loss_changes, axis=0)

        X = np.array([avg_loss_changes])
        window_size_change = self.window_size_model.predict(X, verbose=0)[0][0]

        self.energy_window_size = int(max(self.min_window_size, 
                                          min(self.max_window_size, 
                                              self.energy_window_size + window_size_change)))

        while len(self.energy_window) > self.energy_window_size:
            self.energy_window.pop(0)

    def _train_epsilon_model(self):
        if len(self.energy_window) == self.energy_window_size:
            energy_trend = np.mean(np.diff(self.energy_window))
            normalized_trend = (energy_trend - self.energy_loss_move) / (self.energy_gain_food - self.energy_loss_move)
            
            m3_input = np.array(self.M3_sequence).reshape(1, self.M3_sequence_length, self.M3_state_size)
            predicted_future_value = np.max(self.M3.predict(m3_input, verbose=0)[0])
            
            X = np.array([[normalized_trend, predicted_future_value]])
            y = np.array([[self.epsilon]])
            
            self.epsilon_model.fit(X, y, epochs=1, verbose=0)

    def _train_window_size_model(self):
        if len(self.loss_window) < self.loss_window_size:
            return

        loss_changes = np.diff(self.loss_window, axis=0)
        if len(loss_changes) < 2:  # Need at least 2 samples to calculate a meaningful diff
            return

        X = loss_changes
        
        window_sizes = [len(self.energy_window)] * (len(loss_changes) + 1)
        y = np.diff(window_sizes)

        assert len(X) == len(y), "X and y must have the same number of samples"

        self.window_size_model.fit(X, y, epochs=1, verbose=0)

    def get_energy_score(self):
        return (self.energy - self.initial_energy) / max(1, self.step_count)

    def get_gradient_choice_percentages(self):
        total_choices = sum(self.gradient_choices)
        if total_choices == 0:
            return [0] * self.M3_action_size
        return [100 * choices / total_choices for choices in self.gradient_choices]

    def save_models(self):
        self.M0.save(f'M0_{self.color}_model.h5')
        self.M1.save(f'M1_{self.color}_model.h5')
        self.M2.save(f'M2_{self.color}_model.h5')
        self.M3.save(f'M3_{self.color}_model.h5')
        self.epsilon_model.save(f'epsilon_{self.color}_model.h5')
        self.window_size_model.save(f'window_size_{self.color}_model.h5')

    def load_models(self):
        try:
            self.M0 = tf.keras.models.load_model(f'M0_{self.color}_model.h5')
            self.M1 = tf.keras.models.load_model(f'M1_{self.color}_model.h5')
            self.M2 = tf.keras.models.load_model(f'M2_{self.color}_model.h5')
            self.M3 = tf.keras.models.load_model(f'M3_{self.color}_model.h5')
            self.epsilon_model = tf.keras.models.load_model(f'epsilon_{self.color}_model.h5')
            self.window_size_model = tf.keras.models.load_model(f'window_size_{self.color}_model.h5')
            return True
        except (OSError, IOError):
            return False

    def reset_efficiency_tracking(self):
        self.initial_energy = self.energy
        self.step_count = 0
        self.cumulative_efficiency = 0

    def get_model_errors(self):
        return {
            'M0': self.M0_errors[-1] if self.M0_errors else None,
            'M1': self.M1_errors[-1] if self.M1_errors else None,
            'M2': self.M2_errors[-1] if self.M2_errors else None,
            'M3': self.M3_errors[-1] if self.M3_errors else None
        }

    def get_epsilon(self):
        return self.epsilon

    def get_window_size(self):
        return self.energy_window_size
        
def draw_arrow(screen, color, start, end):
    pygame.draw.line(screen, color, start, end, 2)
    rotation = math.degrees(math.atan2(start[1]-end[1], end[0]-start[0]))+90
    pygame.draw.polygon(screen, color, ((end[0]+5*math.sin(math.radians(rotation)), end[1]+5*math.cos(math.radians(rotation))), (end[0]+5*math.sin(math.radians(rotation-120)), end[1]+5*math.cos(math.radians(rotation-120))), (end[0]+5*math.sin(math.radians(rotation+120)), end[1]+5*math.cos(math.radians(rotation+120)))))

def draw_energy_gradient(screen, agent):
    center = (int(agent.pos[0] * CELL_SIZE + CELL_SIZE/2), int(agent.pos[1] * CELL_SIZE + CELL_SIZE/2))
    
    # Use the combined gradient from the _locomotion_taxis method
    m3_input = np.array(agent.M3_sequence).reshape(1, agent.M3_sequence_length, agent.M3_state_size)
    q_values = agent.M3.predict(m3_input, verbose=0)[0]
    normalized_q_values = (q_values - np.min(q_values)) / (np.max(q_values) - np.min(q_values) + 1e-8)
    
    random_gradient = np.random.rand(360)
    
    combined_gradient = (
        agent.energy_gradient * (1 - agent.epsilon) * 0.4 +
        random_gradient * agent.epsilon * 0.6 +
        agent._q_values_to_gradient(normalized_q_values) * (1 - agent.epsilon) * 0.6
    )
    
    max_energy = np.max(combined_gradient)
    if max_energy > 0:
        normalized_gradient = combined_gradient / max_energy
        base_color = pygame.Color(agent.color)
        
        for angle in range(0, 360, 10):  # Draw every 10 degrees for better performance
            energy = normalized_gradient[angle]
            end_x = center[0] + int(50 * energy * math.cos(math.radians(angle)))
            end_y = center[1] + int(50 * energy * math.sin(math.radians(angle)))
            color = (max(0, min(255, int(base_color.r * energy))), 
                     max(0, min(255, int(base_color.g * energy))), 
                     max(0, min(255, int(base_color.b * energy))))
            pygame.draw.line(screen, color, center, (end_x, end_y), 1)
            
def main():
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_SIZE, SCREEN_SIZE))
    clock = pygame.time.Clock()

    agents = []
    shared_category_space = SharedCategorySpace()
    for i, color in enumerate(COLORS[:N_AGENTS]):
        while True:
            x, y = random.randint(0, GRID_SIZE-1), random.randint(0, GRID_SIZE-1)
            if MAZE[y, x] == 0:
                agent = Agent(x, y, color)
                if agent.load_models():
                    print(f"Loaded existing models for Agent {i+1} ({color})")
                else:
                    print(f"No existing models found for Agent {i+1} ({color}). Starting with new models.")
                agents.append(agent)
                break

    # Reset efficiency tracking for all agents
    for agent in agents:
        agent.reset_efficiency_tracking()

    food_positions = {}
    
    def add_food(count):
        for _ in range(count):
            while True:
                x, y = random.randint(0, GRID_SIZE-1), random.randint(0, GRID_SIZE-1)
                if MAZE[y, x] == 0 and (x, y) not in food_positions:
                    food_positions[(x, y)] = random.randint(FOOD_ENERGY_RANGE[0], FOOD_ENERGY_RANGE[1])
                    break

    # Initialize food
    add_food(CONSTANT_FOOD_COUNT)

    for t in range(TIME_STEPS):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return

        screen.fill((12, 12, 12))

        # Draw maze
        for y in range(GRID_SIZE):
            for x in range(GRID_SIZE):
                if MAZE[y, x] == 1:
                    pygame.draw.rect(screen, (32, 32, 32), (x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE))

        for agent in agents:
            agent.perceive(food_positions, agents, shared_category_space)
            agent.move(food_positions, agents, shared_category_space)
            agent.train_models(food_positions, agents, shared_category_space)
            
            # Optimize epsilon every 100 steps
            if t % 100 == 0:
                agent.optimize_epsilon(food_positions, agents, shared_category_space)

        # Maintain constant food count
        food_deficit = CONSTANT_FOOD_COUNT - len(food_positions)
        if food_deficit > 0:
            add_food(food_deficit)

        # Draw food
        for (x, y), energy in food_positions.items():
            pygame.draw.circle(screen, (128, 128, 128), (int(x * CELL_SIZE + CELL_SIZE/2), int(y * CELL_SIZE + CELL_SIZE/2)), 3)

        # Draw agents and their energy gradients
        for agent in agents:
            draw_energy_gradient(screen, agent)
            start_pos = (int(agent.pos[0] * CELL_SIZE + CELL_SIZE/2), int(agent.pos[1] * CELL_SIZE + CELL_SIZE/2))
            end_pos = (int(start_pos[0] + 10 * math.cos(math.radians(agent.angle))),
                       int(start_pos[1] + 10 * math.sin(math.radians(agent.angle))))
            draw_arrow(screen, pygame.Color(agent.color), start_pos, end_pos)

        pygame.display.flip()
        clock.tick(24)

        # Print learning progression every 100 time steps
        if (t + 1) % 10 == 0:
            print(f"\nTime step: {t+1}")
            for i, agent in enumerate(agents):
                efficiency_score = agent.get_efficiency_score()
                gradient_percentages = agent.get_gradient_choice_percentages()
                model_errors = agent.get_model_errors()
                print(f"Agent {i+1} ({agent.color}):")
                print(f"  Efficiency Score: {efficiency_score:.4f}")
                print(f"  Epsilon: {agent.get_epsilon():.4f}")
                print(f"  Energy Window Size: {agent.get_window_size()}")
                print(f"  Gradient Choices:")
                for j, percentage in enumerate(gradient_percentages):
                    print(f"    State {j}: {percentage:.2f}%")
                
                print("  Model Losses:")
                for model, error in model_errors.items():
                    print(f"    {model} Loss: {error if error is not None else 'N/A'}")

    # Print final learning progression and save models
    print("\nFinal learning progression:")
    for i, agent in enumerate(agents):
        efficiency_score = agent.get_efficiency_score()
        gradient_percentages = agent.get_gradient_choice_percentages()
        model_errors = agent.get_model_errors()
        print(f"Agent {i+1} ({agent.color}):")
        print(f"  Final Efficiency Score: {efficiency_score:.4f}")
        print(f"  Final Epsilon: {agent.get_epsilon():.4f}")
        print(f"  Final Energy Window Size: {agent.get_window_size()}")
        print(f"  Final Gradient Choices:")
        for j, percentage in enumerate(gradient_percentages):
            print(f"    State {j}: {percentage:.2f}%")
        
        print("  Final Model Losses:")
        for model, error in model_errors.items():
            print(f"    {model} Loss: {error if error is not None else 'N/A'}")
        
        # Save models
        agent.save_models()
        print(f"  Models saved for Agent {i+1} ({agent.color})")

    pygame.quit()

if __name__ == "__main__":
    main()
