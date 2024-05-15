import numpy as np
import math
import matplotlib.pyplot as plt
# For AND, OR, NOR, EQUAL, etc. functions
logic_operators_data = [(0,0),(0,1),(1,0),(1,1)]

square_data = (
    (0,0),
    (1,0),
    (2,0),
    (3,0),
    (4,0),
    (1,1),
    (4,1),
    (1,2),
    (3,3),
    (4,2),
    (1,4),
    (4,4),
    (0,4),
    (1,4),
    (2,4),
    (3,4),
    (4,4))

random_coord = list(zip(np.random.randint(low=-10, high=10, size=(150,)), np.random.randint(low=-10, high=10, size=(150,))))
outer_circle = [(x, y) for x, y in random_coord if abs(math.sqrt(x ** 2 + y ** 2) - 9) <= 1.3]
inner_circle = [(x, y) for x, y in random_coord if abs(math.sqrt(x ** 2 + y ** 2) - 3) <= 1.3]

donut_data = outer_circle + inner_circle
np.random.shuffle(donut_data)

