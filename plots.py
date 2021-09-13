import sys
import numpy as np

from model import y_x, y_t, v_t, a_n_t, a_t_t, k
from vispy import scene, app

canvas = scene.SceneCanvas(keys='interactive', size=(600, 600), show=True)

grid = canvas.central_widget.add_grid(margin=10)

def add_f_h(row, col, data, plot_title, y_label, x_label):
    g = grid.add_grid(row=row,col=col,margin=10)

    title = scene.Label(plot_title, color='white')
    title.height_max = 40

    g.add_widget(title, row=0, col=0, col_span=2)

    y_axis = scene.AxisWidget(orientation='left',
                            axis_label=y_label,
                            axis_font_size=12,
                            axis_label_margin=40,
                            tick_label_margin=5)
    y_axis.width_max = 80
    g.add_widget(y_axis, row=1, col=0)

    x_axis = scene.AxisWidget(orientation='bottom',
                            axis_label=x_label,
                            axis_font_size=12,
                            axis_label_margin=30,
                            tick_label_margin=15)

    x_axis.height_max = 50
    g.add_widget(x_axis, row=2, col=1)

    right_padding = g.add_widget(row=1, col=2, row_span=1)
    right_padding.width_max = 30

    view = g.add_view(row=1, col=1, border_color='white')

    view.camera = 'panzoom'

    plot = scene.Line(data, parent=view.scene, width=2)

    # write after adding lines, so that it can 
    # get their bounds
    view.camera.set_range()

    x_axis.link_view(view)
    y_axis.link_view(view)


add_f_h(0, 0, y_x, 'y(x)', 'y', 'x')
add_f_h(0, 1, y_t, 'y(t)', 'y', 't')
add_f_h(0, 2, v_t, 'v(t)', 'v', 't')
add_f_h(1, 0, a_n_t, 'a_n(t)', 'a_n', 't')
add_f_h(1, 2, a_t_t, 'a_t(t)', 'a_t', 't')
# add_f_h(1, 2, k, 'k(x)', 'k', 'x')

if __name__ == '__main__' and sys.flags.interactive == 0:
    app.run()