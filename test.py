from matplotlib.animation import FuncAnimation
from matplotlib import pyplot as plt
import numpy as np
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
line, = plt.plot([], [], "r-", animated=True)
x = []
y = []

def init():
    ax.set_xlim(-np.pi, np.pi)
    ax.set_ylim(-1, 1)
    return line,

def update(frame):
    x.append(frame)
    y.append(np.sin(frame))
    line.set_data(x, y)
    return line,

ani = FuncAnimation(fig
                   ,update
                   ,frames=np.linspace(-np.pi ,np.pi, 90)
                   ,interval=10
                   ,init_func=init
                   ,blit=True
                   )
