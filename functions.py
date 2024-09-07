import math
import numpy as np
# ALL FUNCTIONS ONLY TAKE  TWO INPUTS

def performance_func(desired, output):
    # MSE LOSS
    # return -0.5 * (desired - output) ** 2
    # LOG LOSS
    return  ( desired * np.log(output) + (1 - desired) * np.log(1 - output) )


def derivative_func(desired, output):
    # MSE LOSS
    # return desired - output
    # LOG LOSS
    return  ( (desired / output) - ((1 - desired) / (1 - output)) )


def AND(coordinate):
    return coordinate[0] & coordinate[1]


def NAND(coordinate):
    return ~(coordinate[0] & coordinate[1]) + 2


def OR(coordinate):
    return coordinate[0] | coordinate[1]


def EQUAL(coordinate):
    return int(coordinate[0] == coordinate[1])

def square(coordinate):
    coordinate = tuple(coordinate)
    if coordinate == (3,3):
        return 1

    return 0


def letter_l_func(coordinate):
    coordinate = tuple(coordinate)

    if coordinate == (1, 0) or coordinate == (2,0)\
        or coordinate == (3,0) or coordinate == (4,0) or coordinate == (0,2)\
        or coordinate == (0,3) or coordinate == (0,4) or coordinate == (0,1):
        return 1

    return 0

def donut(coordinate):
    coordinate = tuple(coordinate)
    if abs(math.sqrt(coordinate[0] ** 2 + coordinate[1] ** 2) - 9) <= 1.3:
        return 0

    return 1

