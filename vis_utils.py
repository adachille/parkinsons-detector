import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def heatmap(x, y, size, color, title, save_path=None):
  n_labels = len(x.unique())
  f = plt.figure(figsize=(n_labels, n_labels))

  # Allocate space for the color legend
  plot_grid = plt.GridSpec(1, 15, hspace=0.2, wspace=0.1)  # Setup a 1x15 grid
  # Use the leftmost 14 columns of the grid for the main plot
  ax = f.add_subplot(plot_grid[:, :-1])

  """ Get the colors for the figure """
  n_colors = 256  # Use 256 colors for the diverging color palette
  palette = sns.diverging_palette(20, 220, n=n_colors)  # Create the palette
  color_min, color_max = [-1, 1]  # Range of values mapped to the palette, i.e. min and max possible correlation

  def value_to_color(val):
    val_position = float((val - color_min)) / (
        color_max - color_min)  # position of value in the input range, relative to the length of the input range
    ind = int(val_position * (n_colors - 1))  # target index in the color palette
    return palette[ind]

  """ Set up the scatter plot """
  # Mapping from column names to integer coordinates
  x_labels = [v for v in sorted(x.unique())]
  y_labels = [v for v in sorted(y.unique())]
  x_to_num = {p[1]: p[0] for p in enumerate(x_labels)}
  y_to_num = {p[1]: p[0] for p in enumerate(y_labels)}

  size_scale = 500
  ax.scatter(
    x=x.map(x_to_num),  # Use mapping for x
    y=y.map(y_to_num),  # Use mapping for y
    s=size * size_scale,  # Vector of square sizes, proportional to size parameter
    c=color.apply(value_to_color),  # Vector of square color values, mapped to color palette
    marker='s'  # Use square as scatterplot marker
  )

  # Show column labels on the axes
  ax.set_xticks([x_to_num[v] for v in x_labels])
  ax.set_xticklabels(x_labels, rotation=45, horizontalalignment='right')
  ax.set_yticks([y_to_num[v] for v in y_labels])
  ax.set_yticklabels(y_labels)
  ax.set_title(title)

  # Make grid go between boxes
  ax.grid(False, 'major')
  ax.grid(True, 'minor')
  ax.set_xticks([t + 0.5 for t in ax.get_xticks()], minor=True)
  ax.set_yticks([t + 0.5 for t in ax.get_yticks()], minor=True)

  # Center boxes
  ax.set_xlim([-0.5, max([v for v in x_to_num.values()]) + 0.5])
  ax.set_ylim([-0.5, max([v for v in y_to_num.values()]) + 0.5])

  """ Plot color legend to the side """
  # Add color legend on the right side of the plot
  ax = f.add_subplot(plot_grid[:, -1])  # Use the rightmost column of the plot

  col_x = [0] * len(palette)  # Fixed x coordinate for the bars
  bar_y = np.linspace(color_min, color_max, n_colors)  # y coordinates for each of the n_colors bars

  bar_height = bar_y[1] - bar_y[0]
  ax.barh(
    y=bar_y,
    width=[5] * len(palette),  # Make bars 5 units wide
    left=col_x,  # Make bars start at 0
    height=bar_height,
    color=palette,
    linewidth=0
  )
  ax.set_xlim(1, 2)  # Bars are going from 0 to 5, so lets crop the plot somewhere in the middle
  ax.grid(False)  # Hide grid
  ax.set_facecolor('white')  # Make background white
  ax.set_xticks([])  # Remove horizontal ticks
  ax.set_yticks(np.linspace(min(bar_y), max(bar_y), 3))  # Show vertical ticks for min, middle and max
  ax.yaxis.tick_right()  # Show vertical ticks on the right

  # Save the figure to an image if a path is passed in
  if save_path is not None:
    plt.savefig(save_path)