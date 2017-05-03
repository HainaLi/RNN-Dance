
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation
from pyAudioAnalysis import audioAnalysis

stFeatures = audioAnalysis.beatExtractionWrapper("wavs/24k_magic.wav", True)
energy_x = np.zeros(len(stFeatures[:][0]))
print(len(stFeatures[:][0]))
energy_y = np.zeros(len(stFeatures[:][0]))
for i in range(len(stFeatures[:][0])):
    energy_y[i] = stFeatures[1][i]
    energy_x[i] = (i+1)*50


interval = 20 #delay between frames in milliseconds -- each second = 1000/20 frames
#how many seconds is each unit on the x-axis?
ratio = .001
#seems to be called 50 times per second? = len * .05 = seconds * 50 = frames?
frames = energy_x[-1] * ratio * (1000/interval)

step = (interval/1000.)/ratio #20

# First set up the figure, the axis, and the plot element we want to animate
fig = plt.figure()
#ax = plt.axes(xlim=(0, 10), ylim=(0, 5))
#plt.plot(stFeatures[1, :], 'k')
plt.plot(energy_x, energy_y, 'k')
line, = plt.plot([], [], lw=2)

# initialization function: plot the background of each frame
def init():
    line.set_data([], [])
    return line,

# animation function.  This is called sequentially
def animate(i):
    line.set_data(((i-1000)*step, (i-1000)*step), (-50, 50))
    return line,

# call the animator.  blit=True means only re-draw the parts that have changed.
anim = animation.FuncAnimation(fig, animate, init_func=init,
                               frames=int(frames*2), interval=interval, blit=True)


plt.show()