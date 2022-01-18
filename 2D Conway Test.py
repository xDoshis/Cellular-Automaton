# Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animate 
import argparse

# Set up grid values
ON = 255
OFF = 0
vals = [ON, OFF]

# Define function that will return a random inital grid of size NxN containing values
def initGrid(N):
    ''' returns an initial grid of size NxN where the contents of the grid are randomly selected as either ON or OFF '''
    return np.random.choice(vals, N*N, p = [0.2, 0.8]).reshape(N, N)
    # We set the probability p for ON and OFF to be more in favour of an OFF state

def update(frameNum, img, grid, N):
    # Create copy of current grid state, as with a 2D (88 neighbour) model we go row by row
    gridCopy = grid.copy()

    # Iterate through each cell in each row
    for i in range(N):
        for j in range(N):
            '''
            Conway's Rule of Life:
                * Underpopulation = a cell has <2 LIVE neighbours => DIES
                * Overpopulation = a cell has >3 LIVE neighbours => DIES
                * Reproduction = DEAD cell with =3 LIVE neighbours => LIVES
                * any LIVE cell with =2 or =3 LIVE neighbours => LIVES
            '''
            total = int((grid[i, (j-1)%N] + grid[i, (j+1)%N] +
            grid[(i-1)%N, j] + grid[(i+1)%N, j] +
            grid[(i-1)%N, (j-1)%N] + grid[(i-1)%N, (j+1)%N] +
            grid[(i+1)%N, (j-1)%N] + grid[(i+1)%N, (j+1)%N])/255)
 
            # apply Conway's rules
            if grid[i, j]  == ON:
                if (total < 2) or (total > 3):
                    gridCopy[i, j] = OFF
            else:
                if total == 3:
                    gridCopy[i, j] = ON
 
    # update data
    img.set_data(gridCopy)
    grid[:] = gridCopy[:]
    return img,

# set grid size
N = 100
# set animation update interval
updateInterval = 50
# declare grid
grid = initGrid(N)
# set up animation
fig, ax = plt.subplots()
ax.set_title("Conway's Game of Life")
ax.set_xlabel("2D Cellular Automata")

img = ax.imshow(grid, interpolation='nearest')
ani = animate.FuncAnimation(fig, update, fargs=(img, grid, N, ),
                                frames = 10,
                                interval=updateInterval,
                                save_count=50)

plt.show()