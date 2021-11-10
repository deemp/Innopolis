from Danko_Danila_HA1 import np, get_sols, FK_solve, plt
from matplotlib.widgets import Slider, Button, RadioButtons

input = [1, 1, 1, 1, 1, 1]
input_init = [0., 0., 0., 0., 0., 0]

def plot_manipulator(qs=input_init):
    """
    qs - initial input
    """

    fig = plt.figure()
    ax = fig.add_subplot(1,2,1, projection='3d')

    def plot_q(qs=qs):
        """
        qs - [q_i]
        """
        sols = get_sols(qs=qs)
        for sol in sols:
            t = np.array([np.zeros(3)] + [i[:3,3] for i in FK_solve(qs=sol, flag="full")])
            ax.plot3D(t[:,0], t[:,1], t[:,2])

    def add_sliders(fig=fig):
        def get_ax_sliders():
            ax_sliders = [fig.add_subplot(7,2,2*i) for i in range(1,6+1)]
            return ax_sliders

        ax_sliders = get_ax_sliders()
        delta_q = 0.01
        sliders = [Slider(ax_sliders[i], f"$q_{i}$", 0, 2 * np.pi, valinit=qs[i], valstep=delta_q) for i in range(6)]

        def reset_ax():
            ax.clear()

            ax.set_xlabel('$X$')
            ax.set_ylabel('$Y$')
            ax.set_xlim([-4,4])
            ax.set_ylim([-4,4])
            ax.set_zlim([0,5])

        def update(val=0):
            new_qs = [sliders[i].val for i in range(6)]
            reset_ax()
            plot_q(qs=new_qs)
            fig.canvas.draw_idle()
        
        update()

        for i in sliders:
            i.on_changed(update)
        
        return sliders

    sliders = add_sliders()

    def add_reset_button(fig=fig):
        reset_ax = fig.add_subplot(7,2,14)
        
        reset_button = Button(reset_ax, 'Reset', color='lightgoldenrodyellow')

        def reset(event, sliders=sliders):
            for i in sliders:
                i.reset()

        reset_button.on_clicked(reset)
        return reset_button

    reset_button = add_reset_button()

    plt.show()

plot_manipulator(input_init)