import win32gui
import win32con
import pycodestyle
import random
import pygame as pg
from ctypes import windll

from class_snake import Snake
from neural_network_snake import NeuralNetwork1


class Board:
    def __init__(self, num_row, num_col, num_players, len_snake=5, training=False, save_file='test'):
        # set the width and height of the board
        self.num_row = num_row
        self.num_col = num_col
        self.training = training
        self.NN_playing = False
        self.food = set()

        # draw screen without a snake or food in PyGame, only when the neural network is NOT being trained
        self.square = None
        self.x_left = None
        self.y_top  = None
        self.scr = None

        if not self.training:
            self.draw_empty_scr()

        # make set of all coordinates to be able to place food
        self.all_coords = {(x, y) for y in range(self.num_row) for x in range(self.num_col)}

        # check if a NN is playing
        if num_players < 0:
            num_players *= -1
            self.NN_playing = True
            self.NN_list = [NeuralNetwork1(1, 1, save_file + f'_{i+1}') for i in range(num_players)]

        # make snake(s) and place the first food
        self.snakes_list = [Snake(self, len_snake, index, num_players) for index in range(num_players)]
        self.place_food()

    def draw_empty_scr(self):
        # start PyGame
        pg.init()

        # set the pygame window in fullscreen
        windll.user32.SetProcessDPIAware()
        true_res = (windll.user32.GetSystemMetrics(0), windll.user32.GetSystemMetrics(1))
        self.scr = pg.display.set_mode(true_res, pg.FULLSCREEN)

        # set the PyGame window in focus
        hwnd = pg.display.get_wm_info()['window']
        win32gui.ShowWindow(hwnd, win32con.SW_MINIMIZE)
        win32gui.ShowWindow(hwnd, win32con.SW_RESTORE)
        win32gui.SetForegroundWindow(hwnd)

        # get the dimensions of the screen
        x_max = pg.Surface.get_width(self.scr)
        y_max = pg.Surface.get_height(self.scr)

        # make the complete screen white
        self.scr.fill((255, 255, 255))

        # find size of 1 tile
        self.square = min((x_max - 100)//self.num_col, (y_max - 70)//self.num_row)

        # find corners of the board
        self.x_left = int((x_max - self.num_col*self.square)/2)
        self.y_top  = int((y_max - self.num_row*self.square)/2)

        x_right = int(self.x_left + self.num_col*self.square)
        y_bottom = int(self.y_top + self.num_row*self.square)

        # draw the lines
        for i in range(self.num_row+1):
            pg.draw.line(self.scr, (0, 0, 0), (self.x_left, self.y_top + i*self.square),
                                              (x_right, self.y_top + i*self.square), 1)

        for j in range(self.num_col+1):
            pg.draw.line(self.scr, (0, 0, 0), (self.x_left + j*self.square, self.y_top),
                                              (self.x_left + j*self.square, y_bottom), 1)

    def place_food(self):
        available_coords = self.all_coords.copy()

        # subtract the coordinates of the snake(s) from all the coordinates
        for s in self.snakes_list:
            available_coords = list(set(available_coords) - set(s.snake))

        # randomly choose one coordinate out of the remaining coordinates and add it to the list with food
        random_coord = random.choice(available_coords)
        self.food.add(random_coord)

        # draw the food
        if not self.training:
            self.draw_tile(random_coord, (255, 0, 0))

    def draw_tile(self, coord, color):
        pg.draw.rect(self.scr, color,
                     (self.x_left + coord[0]*self.square + 1, self.y_top + coord[1]*self.square + 1,
                      self.square - 1, self.square - 1))

        pg.display.flip()


if __name__ == '__main__':
    # Run PEP 8 check on this file
    pycodestyle.StyleGuide().check_files([__file__])

    # Test the list_of_board() function
    b = Board(5, 6, 2, 3)
    print(b.snakes_list[0].list_of_board())
