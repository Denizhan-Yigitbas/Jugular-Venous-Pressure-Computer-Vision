import matplotlib.pyplot as plt
from matplotlib import image
import math


class LineBuilder:
    def __init__(self, line):
        self.line = line
        self.xs = list(line.get_xdata())
        self.ys = list(line.get_ydata())
        self.xs.pop()
        self.ys.pop()
        self.cid = line.figure.canvas.mpl_connect('button_press_event', self)
        self.distance = 0

    def __call__(self, event):
        print('click', event)
        if event.inaxes != self.line.axes: return
        self.xs.append(event.xdata)
        self.ys.append(event.ydata)
        if len(self.xs) == 2:
            self.line.set_data(self.xs, self.ys)
            self.line.figure.canvas.draw()
            self.distance = math.sqrt((self.xs[0] - self.xs[1]) ** 2 + (self.ys[0] - self.ys[1]) ** 2)
            ax.annotate(f'Line distance is {self.distance} pixels', xy=(260, 20), xycoords='figure pixels')
            plt.savefig('testimage.png')
            ax.set_title('Click anywhere on the image to exit')
        if len(self.xs) == 3:
            self.xs.pop()
            self.ys.pop()
            plt.close(fig)


im = image.imread('/Users/joshuakowal/Downloads/bball1.jpeg')
fig, ax = plt.subplots()
ax.set_title('click to build line segments')
line, = ax.plot([0], [0])  # empty line

frame1 = plt.gca()
for xlabel_i in frame1.axes.get_xticklabels():
    xlabel_i.set_visible(False)
    xlabel_i.set_fontsize(0.0)
for xlabel_i in frame1.axes.get_yticklabels():
    xlabel_i.set_fontsize(0.0)
    xlabel_i.set_visible(False)
for tick in frame1.axes.get_xticklines():
    tick.set_visible(False)
for tick in frame1.axes.get_yticklines():
    tick.set_visible(False)

linebuilder = LineBuilder(line)

plt.imshow(im)
plt.show()

x, y = linebuilder.xs, linebuilder.ys
print(f"The first point's coordinates are: ({x[0]}, {y[0]})")
print(f"The second point's coordinates are: ({x[1]}, {y[1]})")