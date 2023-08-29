import pycodestyle
import time
import numpy as np
import pygame as pg
import matplotlib.pyplot as plt

from class_board import Board
from neural_network_snake import NeuralNetwork1, Memory


class NNTrainer:
    """ Class to train a Neural Network that tries to play Snake """

    def __init__(self, num_row, num_col, num_players, len_snakes, max_steps=100000, max_steps_per_food=200,
                 learning_rate=0.001, discount_factor=0.9, epsilon_range=(0.001, 1), epsilon_decay=0.0001,
                 batch_size=32, target_freq=1000, epochs=4, save_file='test', load_weights=True, training=True):

        # Make instance variables for the game environment, NNs, and memories
        self.training = training
        self.save_file = save_file
        num_players *= (2 * int(self.training) - 1)

        self.game_data = [num_row, num_col, num_players, len_snakes, self.training, self.save_file]
        self.snakes = Board(*self.game_data).snakes_list

        self.NN_list = [NeuralNetwork1(learning_rate, epochs) for _ in self.snakes]
        self.memory_list = [Memory() for _ in self.snakes]

        # Training parameters
        self.gamma = discount_factor
        self.batch_size = batch_size
        self.target_freq = target_freq

        # Epsilon greedy policy
        self.min_eps, self.max_eps = epsilon_range

        if self.training:
            self.eps = self.max_eps
            self.decay = epsilon_decay
        else:
            self.eps = 0
            self.decay = 0

        # SARSA
        self.state = None
        self.action = None
        self.reward = None
        self.next_state = None

        # Info about the training process
        self.step = 0
        self.max_steps = max_steps
        self.step_per_food = 0
        self.max_steps_per_food = int(max(max_steps_per_food, num_row + num_col))
        self.game_over = False
        self.measure = [0, 0, 0]  # Good, Neutral, Bad
        self.high_scores = []

        # load the weights if indicated so
        if load_weights:
            for i, NN in enumerate(self.NN_list):
                NN.load_NN(self.save_file + f'_{i+1}')

        # Start training loop
        print('\nStarting with training!\n')
        self.training_loop()

    def training_loop(self):
        # Keep training until the maximum number of steps is reached
        while self.step < self.max_steps:
            if not self.training:
                time.sleep(0.15)

                # press esc to quit PyGame
                for evt in pg.event.get():
                    if evt.type == pg.KEYDOWN and evt.__dict__['key'] == 27:
                        self.step = self.max_steps
                        pg.quit()

            if self.training and (self.step + 1) % 5000 == 0:
                print('Step:', self.step + 1)

            # Create a new game environment once the previous game is finished
            if self.game_over:
                self.high_scores.append([[len(s.snake), self.step] for s in self.snakes])
                self.step_per_food = 0
                self.game_over = False

                if self.training:
                    self.snakes = Board(*self.game_data).snakes_list
                else:
                    self.step = self.max_steps

            # Batch train the NNs using a batch size of states and actions after every [batch_size // 8] steps
            if self.training and self.step % (self.batch_size // 8) == 0 and self.step > 2000:
                [NN.train_batch(*memo.get_batch(self.batch_size)) for NN, memo in zip(self.NN_list, self.memory_list)]

            # Update the weights of the target NN
            if self.training and self.step % self.target_freq == 0 and self.step > 2000:
                [NN.update_target_model() for NN in self.NN_list]

            # Update the state to the current board
            for snake, NN, memory in zip(self.snakes, self.NN_list, self.memory_list):
                self.state = np.array(snake.list_of_board())

                # Determine the action, and perform that action
                self.choose_and_perform_action(snake, NN)

                # Calculate the reward, and update the NN based on this reward
                self.calculate_reward(snake)
                if self.training:
                    self.update_NN_weights(NN, memory)

            # Update step count
            self.step += 1
            self.step_per_food += 1

            # Exponentially decay epsilon for the epsilon-greedy policy
            if self.training:
                self.eps = max(self.min_eps, self.eps - self.decay)

        # Save the NNs at the end of the training
        if self.training:
            [NN.save_NN(self.save_file + f'_{i+1}') for i, NN in enumerate(self.NN_list)]

    def choose_and_perform_action(self, snake, NN):
        # Epsilon greedy policy
        if np.random.random() < self.eps:
            self.action = np.random.randint(0, NN.size_actions)

        else:
            prediction = NN.predict_one(self.state)
            self.action = np.argmax(prediction)

        # Move the snake
        snake.new_direction = snake.keys_direction[snake.direction][self.action]
        snake.move()

        # Remember the next state
        if not snake.check_outside_board():
            self.next_state = np.array(snake.list_of_board())

    def calculate_reward(self, snake):  # Do NOT have a reward equal to zero!!!
        # Bad reward if the snake hits itself/other snake or went out of the board, or if max_steps_per_food is reached
        if snake.check_hit_snake() or snake.check_outside_board() or self.step_per_food == self.max_steps_per_food:
            self.reward = -10
            self.game_over = True
            self.measure[2] += 1

        # Good reward if the snake ate food
        elif snake.check_eat():
            self.reward = 10
            self.step_per_food = 0
            self.measure[0] += 1

            # Remove the eaten food and add new food
            snake.board.food.remove(snake.head())
            snake.board.place_food()

        # Good reward if the snake filled the board completely
        elif snake.check_filled_board():
            self.reward = 10
            self.game_over = True
            self.measure[0] += 1

        # Otherwise, give zero reward
        else:
            self.reward = 0
            self.measure[1] += 1

    def update_NN_weights(self, NN, memory):
        # Get the predicted quality of each action for the current and next state
        q_s_a = np.array(NN.predict_one(self.state))
        q_s_a_next = np.array(NN.target_predict_one(self.next_state))

        # Update the quality of the current action
        if not self.game_over:
            q_s_a[self.action] = self.reward + self.gamma * np.max(q_s_a_next)
        else:
            q_s_a[self.action] = self.reward

        # Add the state and desired q_s_a to the memory
        memory.add_state(self.state)
        memory.add_qsa(q_s_a)


if __name__ == '__main__':
    # Run PEP 8 check on this file
    pycodestyle.StyleGuide().check_files([__file__])

    # Train a NN for Snake
    rows = 10
    columns = 10
    number_of_snakes = 1
    initial_length_snake = 5

    start_time = time.time()
    T = NNTrainer(rows, columns, number_of_snakes, initial_length_snake, max_steps=100000, max_steps_per_food=200,
                  learning_rate=0.0001, discount_factor=0.9, epsilon_range=(0.001, 1), epsilon_decay=0.0003,
                  batch_size=32, target_freq=1, epochs=4, save_file='optimal', load_weights=False, training=True)

    # Show information about the training
    print(f'\nRun-time:  {round((time.time() - start_time) / 60, 3)} minutes\n')
    print(f'[Good, Neutral, Bad]:  {T.measure}\n')

    high_score = np.array(T.high_scores)[:, 0, 0]
    high_score = np.convolve(high_score, np.ones(50) / 50, mode='valid')

    num_steps = np.array(T.high_scores)[:, 0, 1]
    num_steps -= np.array([0] + list(num_steps[:-1]))
    num_steps = np.convolve(num_steps, np.ones(50) / 50, mode='valid')

    # Plot
    fig, ax1 = plt.subplots()
    ax1.set_xlabel('Number of Games')

    ax1.set_ylabel('Average Score per Game', color='tab:blue')
    ax1.plot(np.arange(len(high_score), dtype=int), high_score, color='tab:blue')
    ax1.tick_params(axis='y', labelcolor='tab:blue')

    ax2 = ax1.twinx()

    ax2.set_ylabel('Average Steps per Game', color='tab:orange')
    ax2.plot(np.arange(len(high_score), dtype=int), num_steps, color='tab:orange')
    ax2.tick_params(axis='y', labelcolor='tab:orange')

    plt.title("Training Results using the Optimal Parameters")
    fig.tight_layout()
    plt.savefig('plots/' + T.save_file + '.png', bbox_inches='tight')
    plt.show()
