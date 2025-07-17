import matplotlib.pyplot as plt
import numpy as np

#The perceptron
class Perceptron:
    def __init__(self):
        self.w = np.zeros(2)  # weights [w1, w2]
        self.b = 0.0          # bias

    def predict(self, x):
        result = np.dot(self.w, x) + self.b
        return 1 if result >= 0 else -1

    def update(self, x, y_true):
        y_pred = self.predict(x)
        if y_pred != y_true:
            self.w += y_true * np.array(x)
            self.b += y_true
            return True
        return False

#global variables
points = []
labels = []
perceptron = Perceptron()
iteration = 0
training_complete = False
iteration_text = None
line_handle = None

#Mouse event handler
def onclick(event):
    if event.xdata is None or event.ydata is None:
        return

    x, y = event.xdata, event.ydata

    if event.button == 1:  # left click = red = +1
        points.append([x, y])
        labels.append(1)
        plt.plot(x, y, 'ro')
    elif event.button == 3:  # right click = blue = -1
        points.append([x, y])
        labels.append(-1)
        plt.plot(x, y, 'bo')

    plt.draw()

# Decision boundary
def draw_decision_boundary():
    global line_handle
    ax = plt.gca()

    if line_handle:
        line_handle.remove()

    if perceptron.w[1] == 0:
        return  # avoid divide-by-zero

    x_vals = np.array(ax.get_xlim())
    y_vals = -(perceptron.w[0] * x_vals + perceptron.b) / perceptron.w[1]
    line_handle, = ax.plot(x_vals, y_vals, 'k--')  

#Keyboard event handler
def on_key(event):
    global iteration, training_complete

    if event.key == ' ' and not training_complete:
        updated = False
        for x, y in zip(points, labels):
            if perceptron.update(x, y):
                updated = True

        if updated:
            iteration += 1
            iteration_text.set_text(f"Iteration: {iteration}")
            draw_decision_boundary()
            plt.draw()
        else:
            training_complete = True
            iteration_text.set_text(f"Iteration: {iteration} (Done)")
            print("🎉 mission accomplished")
            plt.draw()

#setting up the canvas
fig, ax = plt.subplots()
ax.set_title("Left click = red (+1), Right click = blue (-1), Space = train")
ax.set_xlim(-10, 10)
ax.set_ylim(-10, 10)

iteration_text = ax.text(-9, 9.5, "Iteration: 0", fontsize=12, color='black')

#connecting the events
fig.canvas.mpl_connect('button_press_event', onclick)
fig.canvas.mpl_connect('key_press_event', on_key)

#running thr program
plt.show()
