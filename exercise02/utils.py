import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation

def create_animation(MAP_estimates, ML_estimates, stepsize=10, fps=10):
    '''
    Creates an animated view of the MAP estimates (blue) and the ML estimates (orange).

    @Params:
        MAP_estimates...    List of MAP estimates
        ML_estimates...     List of ML estimates
        stepsize...         Number of observations, that were added in each iteration
        fps...              Frames per second

    @Returns:
        FuncAnimation object - Call plt.show() to display it 
        (In Jupyter Notebook, use %matplotlib notebook)
    '''
    
    n_frames = len(MAP_estimates)
    interval = int(1000/fps)

    fig=plt.figure()

    # initial plot
    x = np.arange(1,len(MAP_estimates[0])+1,1)
    bars_MAP = plt.bar(x, MAP_estimates[0], alpha=0.5, label='MAP')
    bars_ML = plt.bar(x, ML_estimates[0], alpha=0.5, label='ML')
    title = plt.title(f'Number of data {stepsize}')
    plt.legend(loc='upper left')
    plt.xlabel('Dice Result')
    plt.ylabel('Probability')
    # update of plot
    def animate(i):
        title.set_text(f'Number of data {stepsize*(i+1)}')
        for j, b in enumerate(bars_MAP):
            b.set_height(MAP_estimates[i][j])
        for j, b in enumerate(bars_ML):
            b.set_height(ML_estimates[i][j])

    return animation.FuncAnimation(fig, animate, repeat=True, blit=False, frames=n_frames, interval=interval)