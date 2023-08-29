import pygame as pg
import numpy as np
import pycodestyle


# function to get input from the user
def get_input(snakes_list, playing, pause, t_start, t_dif, NN_playing=False):
    for evt in pg.event.get():
        if evt.type == pg.KEYDOWN:
            key = evt.__dict__['key']

            # press esc to quit PyGame
            if key == 27:
                playing = False
                print('Game was stopped by pressing ESC.')
                pg.quit()

            # pause button (space bar)
            if key == 32:
                if not pause:
                    pause = True

                else:
                    pause = False
                    t_start = 0.001 * pg.time.get_ticks() - t_dif

            # change direction of the snake
            if not (pause or NN_playing):  # prevent changing directions when game is paused or when NN is playing
                for snake in snakes_list:
                    if str(key) in snake.keys_direction:  # check if the pressed key impacts this snake's movement
                        if snake.keys_direction[str(key)] != -1 * snake.direction:  # snake can't do 180 degree turn
                            snake.new_direction = snake.keys_direction[str(key)]

    return playing, pause, t_start


def get_NN_input(board):
    for snake, NN in zip(board.snakes_list, board.NN_list):
        # Get the current state of the board
        state = np.array(snake.list_of_board())

        # Make a prediction and move the snake
        action = np.argmax(NN.predict_one(state))
        snake.new_direction = snake.keys_direction[snake.direction][action]


if __name__ == '__main__':
    # Run PEP 8 check on this file
    pycodestyle.StyleGuide().check_files([__file__])
