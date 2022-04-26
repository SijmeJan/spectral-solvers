import numpy as np
import scipy as sp

class LaguerrePoly():
    def __init__(self, n):
        self.n = n

    def __call__(self, x):
        return np.polynomial.laguerre.lagval(x, np.eye(self.n+1)[-1,:])

    def derivative(self, x, k=1):
        if k == 0:
            return self.__call__(x)
        else:
            c = np.polynomial.laguerre.lagder(np.eye(self.n+1)[-1,:], m=k)
            return np.polynomial.laguerre.lagval(x, c)

class LaguerreFunc():
    def __init__(self, n):
        self.n = n

    def __call__(self, x):
        return LaguerrePoly(self.n)(x)*np.exp(-x/2)

    def derivative(self, x, k=1):
        if k == 0:
            return LaguerreFunc(self.n)(x)
        elif k == 1:
            return (LaguerrePoly(self.n).derivative(x) - \
                    0.5*LaguerrePoly(self.n)(x))*np.exp(-x/2)
        elif k == 2:
            return (LaguerrePoly(self.n).derivative(x, k=2) - \
                    LaguerrePoly(self.n).derivative(x) + \
                    0.25*LaguerrePoly(self.n)(x))*np.exp(-x/2)
        else:
            print("Not implemented")
