from time import time
from PyQt5.QtCore import reset
from numpy.core.fromnumeric import transpose
from vispy import color
from vispy.scene.canvas import SceneCanvas
from vispy.visuals.text.text import TextVisual
from model import \
    points, velocities, free_velocities, angular_velocities, \
    accelerations, free_accelerations, N, links
from vispy.gloo.util import _screenshot
from vispy.scene import AxisWidget
from vispy.scene.visuals import Line, Arrow, Markers
import imageio
import numpy as np
import random
from pathlib import Path
from vispy.color import Colormap
from vispy import app
import sys

"""
Config
"""

random.seed(1235)
scale = 2
frame_time = 0.0166
trace_length = N

# styles
velocity_arrow_style = "angle_60"
acceleration_arrow_style = "stealth"
point_style = 'o'

cm = Colormap(colors=['c','m','y','k'])
point_color, velocity_color, acceleration_color, link_color, trace_color = [
    cm[random.random()] for _ in range(5)
]

# should have non-closed trajectory
# works only for method='gl'
trace_connect = np.empty((trace_length,2),np.int32)
trace_connect[:,0] = np.arange(trace_length)
trace_connect[:,1] = trace_connect[:,0]+1
trace_connect[-1,1] = trace_connect[-1,0]

"""
Visualization
"""

# canvas for everything
canvas = SceneCanvas(keys='interactive', show=True, size=(300,300))

# add grid in center of canvas
grid = canvas.central_widget.add_grid(spacing=0)

# viewbox is a zoomable grid cell 
viewbox = grid.add_view(row=0, col=1, camera='panzoom')

# create a widget for x axis at the bottom
x_axis = AxisWidget(orientation='bottom')
# choose size: w = 1 = 100%, h = 0.08 = 8% of grid cell 
x_axis.stretch = (1, 0.08)
# add widget to grid as the rightmost bottommost cell
grid.add_widget(x_axis, row=1, col=1)
# link to ViewBox
x_axis.link_view(viewbox)

# same for y
y_axis = AxisWidget(orientation='left')
y_axis.stretch = (0.08, 1)
grid.add_widget(y_axis, row=0, col=0)
y_axis.link_view(viewbox)

# default color
transparent = (0.,0.,0.,0.)

class Visuals():
    
    def __init__(self):
        # visual elements
        self.title = [
            TextVisual(
                text='Task 1', 
                bold=True, 
                font_size=24, 
                color='w', 
                pos=(400, 40)
                ),
            ]

        self.points = Markers(
                pos=np.array([[10,10]]),
                face_color = transparent,
                parent=viewbox.scene
                )

        self.point_traces = [
            Line(
                color=trace_color, 
                width=1,
                method='gl',
                parent=viewbox.scene
                )
            for _ in points.items()
            ]

        self.links = [
            Line(
                color=link_color,
                width=3,
                method='agg',
                parent=viewbox.scene
                )
            for _ in links
            ]

        self.velocities = [
            Arrow(
                color=velocity_color,
                arrow_color=velocity_color,
                width=2,
                method='agg',
                arrow_type=velocity_arrow_style,
                parent=viewbox.scene
                )
            for _ in velocities
            ]

        self.accelerations = [
            Arrow(
                color=acceleration_color,
                arrow_color=acceleration_color,
                width=2,
                method='agg',
                arrow_type=acceleration_arrow_style,
                parent=viewbox.scene
                )
            for _ in accelerations
            ]

    def reset_points(self):
        self.points.set_data(pos=np.array([[10,10]]),face_color = transparent)

    def reset_velocities(self):
        for i in range(len(self.velocities)):
            self.velocities[i].set_data(color = transparent)
            # crutch: need to disable arrows
            self.velocities[i]._arrows = None

    def reset_accelerations(self):
        for i in range(len(self.accelerations)):
            self.accelerations[i].set_data(color = transparent)
            # crutch: need to disable arrows
            self.accelerations[i]._arrows = None

    def reset_point_traces(self):
        for i in range(len(self.point_traces)):
            self.point_traces[i].set_data(color = transparent)

    def reset_links(self):
        for i in range(len(self.links)):
            self.links[i].set_data(color = transparent)

visual_elements = Visuals()

class BoundsUpdater():
    # bounds for traces
    # actually, r is the current idx
    def __init__(self) -> None:
        self.r = 0
        self.l = N + self.r - trace_length
        self.ind = self.circular_index(self.l,self.r)

    def circular_index(self, start, stop):
        if stop <= start:
            stop += N
        return np.arange(start, stop) % N

    def update(self, ev):
        self.l = (self.l + 1) % N
        self.r = (self.r + 1) % N
        self.ind = self.circular_index(self.l,self.r)

bounds_updates = BoundsUpdater()

class VisualsUpdater():
    def __init__(self):
        self.enabled = {
            "Animation" : True,
            "Traces" : True, 
            "Velocities": True, 
            "Accelerations": True, 
            "Links": True,
            "Points": True
            }

    def update(self, ev):
        l = bounds_updates.l
        r = bounds_updates.r
        ind = bounds_updates.ind
        
        # choose suitable data for update
        
        if self.enabled["Traces"]:
            for i, (_, data) in enumerate(points.items()):
                # last point in trajectory connects to itself
                last_with_self = trace_connect
                # if there is endpoint of trajectory btw l and r
                if l + trace_length >= N:
                    last_with_self=np.copy(trace_connect)
                    last_with_self[N-l-1,1]=N-l-1
                    
                visual_elements.point_traces[i].set_data(
                    pos=data[ind],
                    color=trace_color,
                    connect=last_with_self
                    )
        else:
            visual_elements.reset_point_traces()
        
        if self.enabled["Links"]:
            for i, (p1, p2) in enumerate(links):
                link = np.array([points[p1][r], points[p2][r]])
                visual_elements.links[i].set_data(
                    pos=link,
                    color=link_color
                    )
        else:
            visual_elements.reset_links()

        if self.enabled["Velocities"]:
            for i, (_, data) in enumerate(velocities.items()):
                segment = data[r].reshape((2,2))
                visual_elements.velocities[i].set_data(
                    pos=segment,
                    arrows=data[[r]],
                    color=velocity_color
                    )
        else:
            visual_elements.reset_velocities()
        
        if self.enabled["Accelerations"]:
            for i, (_, data) in enumerate(accelerations.items()):
                segment = data[r].reshape((2,2))
                visual_elements.accelerations[i].set_data(
                    pos=segment,
                    arrows=data[[r]],
                    color=acceleration_color
                    )
        else:
            visual_elements.reset_accelerations()
        
        # update all points at once
        if self.enabled["Points"]:
            point_pos = np.array([data[r] for _, data in points.items()])
            visual_elements.points.set_data(pos=point_pos,face_color='red')
        else:
            visual_elements.reset_points()

    def toggle_enabled_points(self):
        self.enabled["Points"] ^= True

    def toggle_enabled_velocities(self):
        self.enabled["Velocities"] ^= True

    def toggle_enabled_accelerations(self):
        self.enabled["Accelerations"] ^= True

    def toggle_enabled_traces(self):
        self.enabled["Traces"] ^= True
    
    def toggle_enabled_links(self):
        self.enabled["Links"] ^= True

visualUpdates = VisualsUpdater()

class InformationUpdater():
    idx = None
    velocities = None
    points = None
    accelerations = None
    angular_velocities = None

    def __init__(self):
        self.update(None)

    def update(self, ev):
        self.idx = bounds_updates.r

        self.velocities = {
            k : v[self.idx] for k,v in free_velocities.items() 
        }
        self.points = {
            k : v[self.idx] for k,v in points.items()
        }
        self.accelerations = {
            k : v[self.idx] for k, v in free_accelerations.items()
        }
        self.angular_velocities = {
            k : v[self.idx] for k, v in angular_velocities.items()
        }

informationUpdates = InformationUpdater()

timer = app.Timer()
timer.connect(bounds_updates.update)
timer.connect(visualUpdates.update)
timer.connect(informationUpdates.update)

timer.start(0)

# auto-scale to see the whole line.
viewbox.camera.set_range(x=(-60,130),y=(-80,110))

if __name__ == '__main__' and sys.flags.interactive == 0:
    app.run()