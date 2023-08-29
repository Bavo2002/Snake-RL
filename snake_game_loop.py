import pycodestyle
import pygame as pg

from keyboard_input import get_input, get_NN_input


# main game loop
def game_loop(board, t_dif):
    # start values
    t_start = 0.001*pg.time.get_ticks()

    playing = True
    pause = False

    while playing:
        # get input from the user
        playing, pause, t_start = get_input(board.snakes_list, playing, pause, t_start, t_dif, board.NN_playing)

        # Also get input from the NN if it is playing
        if board.NN_playing:
            get_NN_input(board)

        # get current time
        t = 0.001*pg.time.get_ticks()

        # if a specified amount of time (= t_dif) has passed, move the snake one position
        if t >= t_start + t_dif and playing and not pause:
            t_start = t_start + t_dif

            for snake in board.snakes_list:
                # get the index of the player and move the snake of this player
                player = snake.index + 1
                snake.move()

                # if you went out of the board
                if snake.check_outside_board():
                    playing = False
                    print(f'Snake {player} went out of the board!')

                # check if you hit yourself
                elif snake.check_hit_snake():
                    playing = False

                    if snake.check_hit_snake() == 1:
                        print(f'Snake {player} hit itself!')

                    elif snake.check_hit_snake() == 2:
                        print(f'Snake {player} hit another snake!')

                # place new food if you ate food, and remove the food you ate
                elif snake.check_eat() and not snake.check_filled_board():
                    board.food.remove(snake.head())
                    board.place_food()

                    # make the snake go faster when you eat food
                    t_dif -= 0.001

                    if t_dif <= 0.11:
                        t_dif -= 0.0005

                # when you filled the board completely with your snake (pretty impossible)
                elif snake.check_filled_board():
                    playing = False
                    print('You completely filled the board with your snake!')

    # close PyGame
    pg.quit()


if __name__ == '__main__':
    # Run PEP 8 check on this file
    pycodestyle.StyleGuide().check_files([__file__])
