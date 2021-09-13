from __future__ import division
from config import ROOT_DIR
from pathlib import Path


import numpy as np
from numpy.core.fromnumeric import trace

from vispy import app, gloo, visuals
from vispy import color
from vispy.visuals.line import arrow
from vispy.visuals.line.line import LineVisual
from vispy.visuals.transforms import NullTransform
from trajectories import points, velocities, accelerations, N
import random
from vispy.gloo.util import _screenshot
import imageio


from vispy.color import Colormap

random.seed(1235)
x_translation = 400
y_translation = 300
scale = 2
frame_time = 0.01
trace_length = N

# togglers
enabled = {
    "Animation" : True,
    "Traces" : True, 
    "Velocities": True, 
    "Accelerations": True, 
    "Links": True, 
    "Points": True
    }

# styles
velocity_arrow_style = "angle_60"
acceleration_arrow_style = "stealth"
point_style = 'o'


# adapt points to X: >, Y: v coordinate system
def adapt_coordinates(data):
    for i,j in data.items():
        # iflip y
        data[i][:,1::2] = -j[:,1::2]
        # scale
        data[i] *= scale
        # translate
        data[i][:,0::2] += x_translation
        data[i][:,1::2] += y_translation

    return data

points, velocities, accelerations = [
    adapt_coordinates(i)
    for i in [points, velocities, accelerations]     
]


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

links = [["A","B"],["A","C"],["B","C"],["C","D"],["E","F"],["O1","A"],["O2","B"],["O3","F"]]

class Canvas(app.Canvas):

    def on_draw(self, event):
        gloo.clear('black')
        for visual in self.visuals:
            visual.draw()

    def on_resize(self, event):
        # Set canvas viewport and reconfigure visual transforms to match.
        vp = (0, 0, self.physical_size[0], self.physical_size[1])
        self.context.set_viewport(*vp)
        for visual in self.visuals:
            visual.transforms.configure(canvas=self, viewport=vp)

    def __init__(self):
        app.Canvas.__init__(self, keys='interactive',
                            size=(800, 600))
        
        self.points = [
            visuals.MarkersVisual() 
            for _ in points.items()
        ]

        self.point_traces = [
            visuals.LineVisual(
                color=trace_color, 
                width=2,
                method='gl')
            for _ in points.items()
        ]

        self.links = [
            visuals.LineVisual(
                color=link_color, 
                width=3,
                method='agg') 
            for _ in links
        ]

        self.velocities = [
            visuals.ArrowVisual(
                color=velocity_color,
                arrow_color=velocity_color,
                width=2,
                method='agg',
                arrow_type=velocity_arrow_style
                )
            for _ in velocities
        ]

        self.accelerations = [
            visuals.ArrowVisual(
                color=acceleration_color,
                arrow_color=acceleration_color,
                width=2,
                method='agg',
                arrow_type=acceleration_arrow_style
                )
            for _ in accelerations
        ]

        # all visual elements
        self.visuals_list = [
            self.points,
            self.point_traces,
            self.links,
            self.velocities,
            self.accelerations
        ]


        # add visual elements to main visuals list
        self.visuals = []
        
        for i in self.visuals_list:
            self.visuals.extend(i)

        for visual in self.visuals:
            visual.events.update.connect(lambda evt: self.update())

        self.texts = [
            visuals.TextVisual('Task 1', bold=True, font_size=24, color='w', pos=(400, 40))
        ]
                                         
        for text in self.texts:
            text.transform = NullTransform()

        # variables for traces
        self.l = N - trace_length
        self.r = 0

        self._timer = app.Timer('auto',connect=self.on_timer,start=True)

        
        """
        Key lines for animation
        """        
        self._animation_timer = app.Timer(0.0, connect=self.on_animation,start=True)
        Path(f"{ROOT_DIR}/resources/out").mkdir(exist_ok=True, parents=True)
        self._animation_writer = imageio.get_writer(
            uri=f"{ROOT_DIR}/resources/out/task1.gif", 
            fps=60,
            )
        self._animation_counter = 0


        self.show()


    """
    Key lines animation
    """        
    def on_animation(self, event):
        frame = _screenshot()
        self._animation_writer.append_data(frame)
        self._animation_counter += 1
        if self._animation_counter >= 196:
            self._animation_timer.stop()
            self._animation_writer.close()
            self.close()
    

    def on_timer(self, event):
                # drawing traces
        def circular_index(start, stop):
            if stop <= start:
                stop += N
            return np.arange(start, stop) % N
        
        l = self.l
        r = self.r
        l = (l + 1) % N
        r = (r + 1) % N
        self.l = l
        self.r = r

        ind = circular_index(l,r)
        
        # choose suitable data for update
        if enabled["Points"]:
            for i, (_, data) in enumerate(points.items()):
                win.points[i].set_data(
                    pos=data[[r]],
                    edge_color=point_color,
                    face_color=point_color
                    )
        
        if enabled["Traces"]:
            for i, (_, data) in enumerate(points.items()):
                # last point in trajectory connects to itself
                last_with_self = trace_connect
                # if there is endpoint of trajectory btw l and r
                if l + trace_length >= N:
                    last_with_self=np.copy(trace_connect)
                    last_with_self[N-l-1,1]=N-l-1
                    
                win.point_traces[i].set_data(
                    pos=data[ind],
                    color=trace_color,
                    connect=last_with_self
                    )
        
        if enabled["Links"]:
            for i, (p1, p2) in enumerate(links):
                link = np.array([points[p1][r], points[p2][r]])
                win.links[i].set_data(
                    pos=link,
                    color=link_color
                    )

        if enabled["Velocities"]:
            for i, (_, data) in enumerate(velocities.items()):
                segment = data[r].reshape((2,2))
                win.velocities[i].set_data(
                    pos=segment,
                    arrows=data[[r]],
                    color=velocity_color
                    )
        
        if enabled["Accelerations"]:
            for i, (_, data) in enumerate(accelerations.items()):
                segment = data[r].reshape((2,2))
                win.accelerations[i].set_data(
                    pos=segment,
                    arrows=data[[r]],
                    color=acceleration_color
                    )
        
        self.update()


if __name__ == '__main__' and enabled["Animation"]:
    win = Canvas()
    app.run()