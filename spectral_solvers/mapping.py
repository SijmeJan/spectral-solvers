import numpy as np

class CoordinateMap():
    def __init__(self, L=1):
        self.t = lambda x: x/L
        self.dtdx = lambda t: 1/L + t - t
        self.d2tdx2 = lambda t: 0 + t - t
        self.x = lambda t: L*t

class ChebychevMap():
    def __init__(self, x_interval=[-1,1], L=1):
        if x_interval[0] != -np.inf:
            if x_interval[1] != np.inf:
                # Finite interval
                a = 2/(x_interval[1] - x_interval[0])
                b = 1 - a*x_interval[1]

                self.t = lambda x: a*x + b
                self.dtdx = lambda t: a + t - t
                self.d2tdx2 = lambda t: 0 + t - t
                self.x = lambda t: (t - b)/a
            else:
                # Semi-infinite interval
                self.t = \
                  lambda x: (x - x_interval[0] - L)/(x - x_interval[0] + L)
                self.dtdx = lambda t: 0.5*(1-t)**2/L
                self.d2tdx2 = lambda t: -0.5*(1 - t)**3/L/L
                self.x = lambda t: L*(1 + t)/(1 - t) + x_interval[0]
        else:
            if x_interval[1] == np.inf:
                self.t = lambda x: x/np.sqrt(L*L + x*x)
                self.dtdx = lambda t: np.sqrt(1 - t*t)*(1 - t*t)/L
                self.d2tdx2 = lambda t: -3*(1 - t*t)**2*t/L/L
                self.x = lambda t: L*t/np.sqrt(1 - t*t)
            else:
                print('NOT IMPLEMENTED')
