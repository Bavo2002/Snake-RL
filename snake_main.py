import pycodestyle

from class_board import Board
from snake_game_loop import game_loop
from questions import play_again, choose_mode


# main program
def main(num_row=20, num_col=30, t_dif=0.15, len_snake=5, file='test'):
    assert num_col > len_snake
    print('\nWelcome to Snake!')
    again = True

    while again:
        # create board and list of snakes
        num_players = choose_mode()

        my_board = Board(num_row, num_col, num_players, len_snake=len_snake, save_file=file)
        print('\n')

        # main game loop
        game_loop(my_board, t_dif)

        # ask if user wants to play again
        again = play_again()


if __name__ == '__main__':
    # Different starting variables
    rows = 10
    cols = 10
    time_dif = 0.2
    length_snake = 5
    save_file = 'optimal'

    # Run main program
    # main()
    main(rows, cols, time_dif, length_snake, save_file)

    # Run PEP 8 check on this file
    pycodestyle.StyleGuide().check_files([__file__])
