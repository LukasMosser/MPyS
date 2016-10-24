
def output_to_png(filename, simulation_grid, ax, fig):

    #fig, ax = plt.subplots(1, 1, figsize=(13, 13))
    ax.imshow(simulation_grid, cmap='Greys',  interpolation='nearest')
    for item in ax.get_xticklabels():
        item.set_fontsize(28)
    for item in ax.get_yticklabels():
        item.set_fontsize(28)

    fig.canvas.draw()
    fig.savefig(filename)