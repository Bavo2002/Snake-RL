import pycodestyle
from numpy import zeros


class Snake:
    def __init__(self, board, length, index_snakes_list, num_players):
        # Assign variables
        self.board = board
        self.index = index_snakes_list

        # Determine the starting direction and position
        self.direction = 2
        self.new_direction = self.direction

        y_coord = (self.index + 1) * self.board.num_row // (num_players + 1)
        x_coord = max(self.board.num_col // 2, length - 1)

        # Make initial snake(s)
        self.snake = [(x_coord - a, y_coord) for a in range(length)]

        # Make dictionaries containing controls for the snake(s)
        if self.board.training or self.board.NN_playing:
            # Make a dictionary mapping direction of snake, and then action of NN, to new direction of snake
            self.keys_direction = {-2: {0: -2, 1: 1, 2: -1}, -1: {0: -1, 1: -2, 2: 2},
                                   1: {0: 1, 1: 2, 2: -2}, 2: {0: 2, 1: -1, 2: 1}}

        else:
            # Map keys of the keyboard to the correct action
            if self.index == 0:
                self.keys_direction = {'276': -2, '275': 2, '273': -1, '303': -1, '274': 1}  # 303 = shift, up = 273

            elif self.index == 1:  # Add more control dictionaries in case more than 2 players play
                self.keys_direction = {'97': -2, '100': 2, '119': -1, '115': 1}

        # Assign colours to the snake(s)
        if self.index == 0:
            self.colours = [(0, 160, 0), (0, 250, 0)]

        elif self.index >= 1:  # Add more colours in case more than 2 players play
            self.colours = [(0, 0, 200), (0, 230, 255)]

        # Draw initial snake
        if not self.board.training:
            for block in self.snake:
                if self.snake[0] == block:
                    self.board.draw_tile(block, self.colours[0])

                else:
                    self.board.draw_tile(block, self.colours[1])

    def __str__(self):
        return ', '.join([str(block) for block in self.snake])

    def head(self):
        return self.snake[0]

    def check_eat(self):
        return self.head() in self.board.food

    def check_outside_board(self):
        return not (0 <= self.head()[0] < self.board.num_col and 0 <= self.head()[1] < self.board.num_row)

    def check_filled_board(self):
        return len(self.snake) == self.board.num_row * self.board.num_col

    def check_hit_snake(self):
        # check if you hit yourself
        if self.head() in self.snake[1:]:
            return 1

        # check if you hit another snake
        for snake in self.board.snakes_list:
            if snake != self and self.head() in snake.snake:
                return 2

        return False

    def move(self):
        """ direction: left = -2, up = -1, down = 1, right = 2 """

        # Move snake in correct direction by taking the last direction inputted by the user
        if not self.direction == -1 * self.new_direction:
            self.direction = self.new_direction

        # Determine next position of head (making use of the 'weird' values for direction)
        new_x = self.head()[0] + round(self.direction / 2)
        new_y = self.head()[1] + self.direction * (self.direction % 2)

        # Remember previous tail, so you can add it to snake if you eat food
        prev_tail = self.snake[-1]

        # Move snake
        self.snake = [(new_x, new_y)] + self.snake[:-1]

        # If you did not go out of the self.board
        if not self.check_outside_board():

            # Add previous tail to snake if you ate food
            if self.check_eat():
                self.snake.append(prev_tail)

            # Remove previous tail from screen if you did not eat
            elif not self.board.training:
                self.board.draw_tile(prev_tail, (255, 255, 255))

            # Move the head of the snake forward
            if not self.board.training:
                self.board.draw_tile(self.head(), self.colours[0])
                self.board.draw_tile(self.snake[1], self.colours[1])

    def list_of_board(self):
        # Return 2D list of board with snake(s) and food on it
        list_board = zeros((self.board.num_row, self.board.num_col), dtype=int)

        # Add the complete snake for each snake with value -1
        for i, snake in enumerate(self.board.snakes_list):
            for coord in snake.snake:
                list_board[coord[1], coord[0]] = -1

        # For THIS snake, overwrite its head with value 2
        list_board[self.head()[1], self.head()[0]] = 2

        # Add the food with again a different value (5)
        for apple in self.board.food:
            list_board[apple[1], apple[0]] = 5

        # Add a border of walls around the board with a value of -1
        border = zeros((self.board.num_row + 2, self.board.num_col + 2), dtype=int) - 1
        border[1:-1, 1:-1] = list_board

        # Append the direction to the list: [left? (0 or 1), right? (0 or 1), up? (0 or 1), down? (0 or 1)]
        list_state = []
        list_state += {-2: [1, 0, 0, 0], 2: [0, 1, 0, 0], -1: [0, 0, 1, 0], 1: [0, 0, 0, 1]}[self.direction]

        # Append the x-direction in which the food is
        if list(self.board.food)[0][0] < self.head()[0]:
            list_state += [1, 0]
        elif list(self.board.food)[0][0] > self.head()[0]:
            list_state += [0, 1]
        else:
            list_state += [0, 0]

        # Append the y-direction in which the food is
        if list(self.board.food)[0][1] < self.head()[1]:
            list_state += [1, 0]
        elif list(self.board.food)[0][1] > self.head()[1]:
            list_state += [0, 1]
        else:
            list_state += [0, 0]

        # Append direction in which there is immediate danger
        danger_left  = border[self.head()[1] + 1, self.head()[0] - 0] == -1
        danger_right = border[self.head()[1] + 1, self.head()[0] + 2] == -1
        danger_up    = border[self.head()[1] - 0, self.head()[0] + 1] == -1
        danger_down  = border[self.head()[1] + 2, self.head()[0] + 1] == -1

        list_state += [int(danger_left), int(danger_right), int(danger_up), int(danger_down)]

        return list_state


if __name__ == '__main__':
    # Run PEP 8 check on this file
    pycodestyle.StyleGuide().check_files([__file__])
