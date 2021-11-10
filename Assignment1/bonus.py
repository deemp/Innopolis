from Danko_Danila_HA1 import np, get_sols, FK_solve, plt, eq
from matplotlib.widgets import Slider, Button
import matplotlib.animation as anima

input = [2, 1, 3, 2, 1, 5]
input_init = [0., 0., 0., 0., 0., 0]
import time

class ManipulatorPlotter():
    def __init__(self, qs, qs_0, qs_1, frames) -> None:
        self.qs = qs
        self.qs_0 = qs_0
        self.qs_1 = qs_1
        self.frames = frames
        self.q_space = self.get_q_space()
        self.fig = plt.figure()
        self.plt_ax = self.fig.add_subplot(1,2,1, projection='3d')
        self.plot_qs_list(sols=[self.qs])
        self.sliders_delta_q = 0.01
        self.sliders = self.add_sliders()
        self.update_with_sliders()
        self.reset_button = self.add_reset_button()
        self.animate_button = self.get_animate_button()
        self.animation = None
        
        plt.show()

    def plot_qs_list(self, sols):
        lines = []
        for sol in sols:
            t = np.array([np.zeros(3)] + [i[:3,3] for i in FK_solve(qs=sol, flag="full")])
            lines += [self.plt_ax.plot(t[:,0], t[:,1], t[:,2])]
        return lines

    def reset_ax(self):
        ax = self.plt_ax
        ax.clear()
        ax.set_xlabel('$X$')
        ax.set_ylabel('$Y$')
        ax.set_xlim([-4,4])
        ax.set_ylim([-4,4])
        ax.set_zlim([0,5])

    def update_with_sliders(self, val=None):
        new_qs = [self.sliders[i].val for i in range(6)]
        self.reset_ax()
        self.plot_qs_list(sols=get_sols(qs=new_qs))
        self.fig.canvas.draw_idle()

    def add_sliders(self):
        ax_sliders = [self.fig.add_subplot(8,2,2*i) for i in range(1,6+1)]
        sliders = [Slider(ax_sliders[i], f"$q_{i}$", 0, 2 * np.pi, valinit=self.qs[i], valstep=self.sliders_delta_q) for i in range(6)]
        
        for i in sliders:
            i.on_changed(self.update_with_sliders)
        
        return sliders

    def add_reset_button(self):
        reset_ax = self.fig.add_subplot(8,2,14)
        reset_button = Button(reset_ax, 'Reset', color='lightgoldenrodyellow')
        reset_button.on_clicked(self.reset)
        return reset_button

    def reset(self, event=None):
        self.reset_ax()
        for i in self.sliders:
            i.reset()
        self.plot_qs_list(qs=get_sols(qs=self.qs))

    def get_q_space(self):
        return np.linspace(tuple(self.qs_0), tuple(self.qs_1), self.frames)

    def animate_frame(self, i, many):
        self.reset_ax()
        if many == True:
            return self.plot_qs_list(get_sols(qs=self.q_space[i]))
        else: 
            return self.plot_qs_list([self.q_space[i]])

    def animate(self, many=True, event=None):
        self.animation = anima.FuncAnimation(fig=self.fig, func=lambda i: self.animate_frame(i=i, many=many), frames=self.frames, interval=50, blit=False)
        return anima.FuncAnimation
        
    def get_animate_button(self):
        animate_ax = self.fig.add_subplot(8,2,16)
        animate_button = Button(animate_ax, 'Animate', color='lightgoldenrodyellow')
        animate_button.on_clicked(self.animate)
        return animate_button

mp = ManipulatorPlotter(qs=input_init, qs_0=input_init, qs_1=input, frames=10)