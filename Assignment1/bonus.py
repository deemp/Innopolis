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

plot_manipulator(input)


import numpy as np
import matplotlib.pyplot as plt

def slider_demo():
    fig, ax = plt.subplots()
    plt.subplots_adjust(left=0.25, bottom=0.25)
    t = np.arange(0.0, 1.0, 0.001)
    a0 = 5
    f0 = 3
    delta_f = 5.0
    s = a0 * np.sin(2 * np.pi * f0 * t)
    l, = plt.plot(t, s, lw=2)
    ax.margins(x=0)

    axcolor = 'lightgoldenrodyellow'
    axfreq = plt.axes([0.25, 0.1, 0.65, 0.03], facecolor=axcolor)
    axamp = plt.axes([0.25, 0.15, 0.65, 0.03], facecolor=axcolor)

    sfreq = Slider(axfreq, 'Freq', 0.1, 30.0, valinit=f0, valstep=delta_f)
    samp = Slider(axamp, 'Amp', 0.1, 10.0, valinit=a0)


    def update(val):
        amp = samp.val
        freq = sfreq.val
        l.set_ydata(amp*np.sin(2*np.pi*freq*t))
        fig.canvas.draw_idle()


    sfreq.on_changed(update)
    samp.on_changed(update)

    resetax = plt.axes([0.8, 0.025, 0.1, 0.04])
    button = Button(resetax, 'Reset', color=axcolor, hovercolor='0.975')


    def reset(event):
        sfreq.reset()
        samp.reset()
    button.on_clicked(reset)

    rax = plt.axes([0.025, 0.5, 0.15, 0.15], facecolor=axcolor)
    radio = RadioButtons(rax, ('red', 'blue', 'green'), active=0)


    def colorfunc(label):
        l.set_color(label)
        fig.canvas.draw_idle()
    radio.on_clicked(colorfunc)

    plt.show()

# slider_demo()