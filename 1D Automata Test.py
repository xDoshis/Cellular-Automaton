# import libraries
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

powers_of_two = np.array([[4], [2], [1]])  # shape (3, 1)

def step(x, rule_binary):
    """Makes one step in the cellular automaton.

    Args:
        x (np.array): current state of the automaton
        rule_binary (np.array): the update rule

    Returns:
        np.array: updated state of the automaton
    """
    # Deal with edge case by looping the edge cases around to each other
    x_shift_right = np.roll(x, 1)  # circular shift to right
    x_shift_left = np.roll(x, -1)  # circular shift to left
    y = np.vstack((x_shift_right, x, x_shift_left)).astype(np.int8)  # stack row-wise, shape (3, cols)
    z = np.sum(powers_of_two * y, axis=0).astype(np.int8)  # LCR pattern as number

    return rule_binary[7 - z]

def cellular_automaton(rule_number, size, steps):
    """Generate the state of an elementary cellular automaton after a pre-determined
    number of steps starting from some random state.

    Args:
        rule_number (int): the number of the update rule to use
        size (int): number of cells in the row
        steps (int): number of steps to evolve the automaton

    Returns:
        np.array: final state of the automaton
    """
    # Check that the function is being called with a valid rule 
    assert 0 <= rule_number <= 255
    
    # Convert the inputed rule number into it's binary counterpart in string format
    rule_binary_str = np.binary_repr(rule_number, width=8)

    # Store each binary value from the string in it's own slot in an array
    rule_binary = np.array([int(ch) for ch in rule_binary_str], dtype=np.int8)

    # Create empty array that will store the entirity of the automata
    x = np.zeros((steps, size), dtype=np.int8)
    # Populate very first row of structure with random numbers
    x[0, :] = np.array(np.random.rand(size) < 0.5, dtype=np.int8)
    
    # Iterate through each row, calling the step function for each row
    for i in range(steps - 1):
        x[i + 1, :] = step(x[i, :], rule_binary)
    
    return x

rule_number = 30  # select the update rule
size = 100  # number of cells in one row
steps = 100  # number of time steps

x = cellular_automaton(rule_number, size, steps)
steps_to_show = 100  # number of steps to show in the animation window
iterations_per_frame = 1  # how many steps to show per frame
frames = int(steps // iterations_per_frame)  # number of frames in the animation
interval=50  # interval in ms between consecutive frames

fig = plt.figure()

ax = plt.axes()
ax.set_axis_off()

def animate(i):
    ax.clear()  # clear the plot
    ax.set_axis_off()  # disable axis
    ax.set_title('Rule 124 - 1D Cellular Automata')
    
    Y = np.zeros((steps_to_show, size), dtype=np.int8)  # initialize with all zeros
    lower_boundary = (i + 1) * iterations_per_frame
    upper_boundary = 0 if lower_boundary <= steps_to_show else lower_boundary - steps_to_show
    for t in range(upper_boundary, lower_boundary):  # assign the values
        Y[t - lower_boundary, :] = x[t, :]
    
    img = ax.imshow(Y, interpolation='none')
    return [img]
    
# call the animator
anim = animation.FuncAnimation(fig, animate, frames=frames, interval=50, blit=True)
anim.save('Rule30.gif', writer='imagemagick', fps=30)
plt.show()