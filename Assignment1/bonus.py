from re import S
from Danko_Danila_HA1 import np, get_sols, FK_solve, plt, eq
from matplotlib.widgets import Slider, Button, CheckButtons
import matplotlib.animation as anima

input = [2, 1, 3, 2, 1, 5]
input_init = [0., 0., 0., 0., 0., 0]
frames = 20

# Use sliders to set the target joint configuration
# Press Animate to see how the manipulator will reach the target joint config from the initial config
# Check Many if you want to see all IK solutions at the same time

class ManipulatorPlotter():
    def __init__(self, qs=input_init, qs_0=input_init, qs_1=input, frames=frames) -> None:
        self.qs = qs
        self.qs_0 = qs_0
        self.qs_1 = qs_1
        self.frames = frames
        self.many = True
        self.q_space = self.get_q_space()
        self.fig = plt.figure()
        self.plt_ax = self.fig.add_subplot(1,2,1, projection='3d')
        self.plot_qs_list(sols=[self.qs])
        self.sliders_delta_q = 0.01
        self.sliders = self.add_sliders()
        self.update_with_sliders()
        self.reset_button = self.add_reset_button()
        self.animate_button = self.add_animate_button()
        self.check_button = self.add_checkbox_many()
                
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
        self.qs_1 = new_qs
        self.q_space = self.get_q_space()
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
        self.animation = None
        for i in self.sliders:
            i.reset()
        sols = get_sols(qs=self.qs) if self.many else [self.qs]
        self.plot_qs_list(sols=sols)

    def get_q_space(self):
        return np.linspace(tuple(self.qs_0), tuple(self.qs_1), self.frames)

    def animate_frame(self, i):
        self.reset_ax()
        if self.many == True:
            return self.plot_qs_list(get_sols(qs=self.q_space[i]))
        else: 
            return self.plot_qs_list([self.q_space[i]])

    def animate(self, event=None):
        self.animation = anima.FuncAnimation(fig=self.fig, func=self.animate_frame, frames=self.frames, interval=50, blit=False, repeat=False)
        
    def add_animate_button(self):
        animate_ax = self.fig.add_subplot(8,4,31)
        animate_button = Button(animate_ax, 'Animate', color='lightgoldenrodyellow')
        animate_button.on_clicked(self.animate)
        return animate_button

    def update_many(self, event=None):
        self.many = not self.many

    def add_checkbox_many(self):
        ax = self.fig.add_subplot(8,4,32)
        label = ["Many"]
        visibility = [True]
        check = CheckButtons(ax, label, visibility)
        check.on_clicked(self.update_many)
        return check

mp = ManipulatorPlotter()
