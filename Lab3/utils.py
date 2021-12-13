import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

b = 0.4
k = 4
m = 0.1

t0 = 0
tf = 3
increment = 0.01
t = np.arange(t0, tf, increment)
x0 = [2, 0]

def solve_spring_damper(y0=x0, t=(t0,tf), t_eval=t, b=b, k=k, m=m):
    """solver for spring-damper system

    ----------
    ### Parameters
      `y0` : `np.array`
          initial values
      `t` : `np.array`
          time coordinates
      `args` : `[float, float, float]`
          `b` - damping coefficient,
          `k` - spring stiffness,
          `m` - mass

    -------
    ### Returns
      `np.array`
          solution to the system
    """
    A = np.array([[0, 1], [-k / m, -b / m]])

    def state_space(t, y):
        return np.dot(A, y)

    return solve_ivp(fun=state_space, y0=y0, t_span=t, t_eval=t_eval)

def solve_list(args):
    """solver for list of systems

    ----------
    ### Parameters
        `args` : `[[solver, {arguments}]]`

    -------
    ### Returns
        `[solutions]`


    -------
    ### Examples
    >>> sol = solve_list([[solve_spring_damper,{}], [solve_spring_damper,{"b": 4}]])
    """

    # where to save results
    res = [None for i in range(len(args))]

    # apply solvers to arguments
    for i, (f, arg) in enumerate(args):
        res[i] = f(**arg)

    return res

def add_graph(ax=None, x=None, y=None, color=None, label=None):
    """adds a graph to axis"""
    ax.plot(x, y, color=color, label=(f"{label}" if label else None))

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
        xlabel=(f"{xlabel}" if xlabel else None),
        ylabel=(f"{ylabel}" if ylabel else None),
    )

    # plot graphs
    for graph in graphs:
        add_graph(ax=ax, x=x, **graph)

    # there might be no label
    _, labels = ax.get_legend_handles_labels()
    if labels:
        ax.legend(loc="best")

    if title:
        ax.set_title(f"{title}")


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
            `graph` : `[dict]`
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

# sol = solve_list([[solve_spring_damper, {}]])

# solver = solve_spring_damper
# sol = solve_list(
#     [
#         [solver, {}],
#         [solver, {"b": 0}],
#         [solver, {"b": 2 * np.sqrt(k * m)}],
#         [solver, {"b": 2 * np.sqrt(k * m) + 1}],
#     ]
# )