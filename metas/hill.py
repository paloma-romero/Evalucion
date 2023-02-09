import numpy as np
from matplotlib import pyplot

def function(x)->int:
    return x**3
def plot():
    limites = np.asarray([[-10.0,10.0]])
    x_inputs = np.arange(limites[0,0],limites[0,1],0.1)
    y_inputs= [function(x) for x in x_inputs]
    pyplot.plot(x_inputs,y_inputs,'--')
    pyplot.show()