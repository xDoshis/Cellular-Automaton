# import libraries
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Define function that creats stacked array for 1D row
def row_stack(current_row):
    # Create two new rows that are the current row shifted left and right
    shift_left = np.roll(current_row, -1) 
    shift_right = np.roll(current_row, 1)

    # Stack all three rows together to obtain new array, 
    # where 1st row is right neighbour and 3rd row is left neighbour
    stacked_current = np.vstack((shift_left, current_row, shift_right)).astype(np.int8)

    return stacked_current

# Define function that sums stacked row
def sum_stack(stacked):
    # Create 3x1 array that stores power of twos value
    powers = np.array([[4], [2], [1]])

    # Sum the columns to obtain the index for binary form of rule #
    rule_index = np.sum(powers * stacked, axis = 0).astype(np.int8)

    return rule_index

# Define function that creates output row based on previous rows rule index
def neighbour_1D_sum(row, rule):

    # Create stacked row
    stacked = row_stack(row)

    # Transform stacked row into index
    index = sum_stack(stacked)

    # Return new row state
    return rule[index]

# Define function that converts rule number into binary array 
def rule_bin(rule):
    # Convert the rule # into binary form, and store each binary element in array
    rule_binary = np.binary_repr(rule, width = 8)
    rule_bin_array = []
    for i in rule_binary:
        rule_bin_array.append(int(i))
    
    # Covert list to numpy array
    rule_bin_array = np.array(rule_bin_array, dtype=np.int8)

    return rule_bin_array

# Define function that will perform automaton loop
def elem_automaton(rule, size, steps):
    # Convert rule into binary format
    rule_bin_array = rule_bin(rule)
    
    # Create empty array that will eventually be filled row by row with automata
    automata = np.zeros((steps, size), dtype=np.int8)

    # Randomly populate 1st row of automata
    automata[0,:] = np.array(np.random.rand(size) < 0.45, dtype=np.int8)

    # Iterate through and update each row based on results of previous row
    for row in range(steps - 1):
        automata[row + 1, :] = neighbour_1D_sum(automata[row,:], rule_bin_array)

    return automata

# Set up grid values for Conway
ON = 255
OFF = 0
vals = [ON, OFF]

def conway_update(grid, N):
    # Create copy of current grid state
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
            # Generate sum for each cell of it's Moore Neighbourhood 
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
 
    # update grid
    grid[:] = gridCopy[:]
    return grid

'''    Define constants for systems    '''
# Set which rule to execute elementary automata with
rule_number = 124 
# Amount of cells in one row
size = 100
# Amount of time to iterate through calculations
steps = 200 

'''  Define constants for animations and initate figures '''
steps_to_show = 100
iterations_frame = 1
frames = int(steps // iterations_frame)
interval = 50 

fig, (ax1, ax2) = plt.subplots(1, 2, sharey = True)

'''  Obtain complete array for elementary automata  '''
x = elem_automaton(rule_number, size, steps)

'''   Initiate Conway's Game of Life with random grid   '''
conway_grid = np.random.choice(vals, size*size, p = [0.2, 0.8]).reshape(size, size)

def animate(i):
    # Set up Life figure
    ax1.clear() 
    ax1.set_title("Conway's Game of Life")
    ax1.set_axis_off() 
    # Set up elementary automata figure
    ax2.clear()
    ax2.set_title('Rule 124')
    ax2.set_axis_off() 
    
    ''' Establish scrolling for elementary automata '''
    # Start with blank figure
    Y = np.zeros((steps_to_show, size), dtype=np.int8)

    # Set upper and lower boundaries, with lower boundary increasing over time to generate
    # scrolling appearance
    low_bound = (i + 1) * iterations_frame
    up_bound = 0 if low_bound <= steps_to_show else low_bound - steps_to_show
    for t in range(up_bound, low_bound):  # assign the values
        Y[t - low_bound, :] = x[t, :]
    
    # Generate frame for elementary automata
    img_elem = ax2.imshow(Y, interpolation='none',cmap='bone')

    # Update Conway's game
    conway = conway_update(conway_grid, size)
    # Generate new frame for Conway's game
    img_conw = ax1.imshow(conway, interpolation='nearest', cmap='spring')
    return [img_elem, img_conw]

# Establish spacing between figures
plt.subplots_adjust(wspace=0.1, hspace=0)

# Call animation function to generate figures
anim = animation.FuncAnimation(fig, animate, frames=frames, interval=50, blit=True)
#anim.save('Rule124.gif', writer='imagemagick', fps=30)
plt.show()