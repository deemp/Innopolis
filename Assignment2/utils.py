import numpy as np
import matplotlib.pyplot as plt

def add_graph(ax=None, x=None, y=None, color=None, label=None):
    """adds a graph to axis"""
    ax.plot(x, y, color=color, label=(f"${label}$" if label else None))

def plot_graphs(
    ax=None,
    x=None,
    xlabel=None,
    ylabel=None,
    title=None,
    graphs=None,
):
    ax.grid()
    # enable latex
    ax.set(
        xlabel=(f"${xlabel}$" if xlabel else None),
        ylabel=(f"${ylabel}$" if ylabel else None),
    )

    # plot graphs
    for graph in graphs:
        add_graph(ax=ax, x=x, **graph)

    # there might be no label
    _, labels = ax.get_legend_handles_labels()
    if labels:
        ax.legend(loc="best")

    if title:
        ax.set_title(f"${title}$")


def plot_sol(args):
    """plot separately list of systems

    -------
    ### Parameters
        `args` : `[dict]`
            `x` : `np.array`
            `xlabel` : `str`
                latex string
            `ylabel` : `str`
                latex string
            `title` : `str`
                latex string
            `graphs` : `[dict]`
                `y` : `np.array`
                `color` : `str`
                    see [options](https://matplotlib.org/stable/gallery/color/named_colors.html)
                `label` : `str`
                    latex string
    -------
    ### Returns
      `None`

    -------
    ### Examples
    """

    # plots will be aligned into a line
    fig, ax = plt.subplots(1, len(args), figsize=(6 * len(args), 6))

    # convert to iterable
    ax = np.array(ax).reshape(-1)

    # set padding between plots
    fig.tight_layout(pad=4.0)

    for i, arg in enumerate(args):
        plot_graphs(ax[i], **arg)

    plt.show()
