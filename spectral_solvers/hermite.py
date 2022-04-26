#!/usr/bin/python

import numpy as np

class HermiteFunc():
    def __init__(self, n):
        self.n = n

    def __call__(self, x):
        if self.n==0:
            return np.ones_like(x)*np.pi**(-0.25)*np.exp(-x**2/2)
        if self.n==1:
            return np.sqrt(2.)*x*np.exp(-x**2/2)*np.pi**(-0.25)
        h_i_2=np.ones_like(x)*np.pi**(-0.25)
        h_i_1=np.sqrt(2.)*x*np.pi**(-0.25)
        sum_log_scale=np.zeros_like(x)
        for i in range(2, self.n+1):
            h_i=np.sqrt(2./i)*x*h_i_1-np.sqrt((i-1.)/i)*h_i_2
            h_i_2, h_i_1=h_i_1, h_i
            log_scale=np.log(np.abs(h_i)).round()
            scale=np.exp(-log_scale)
            h_i=h_i*scale
            h_i_1=h_i_1*scale
            h_i_2=h_i_2*scale
            sum_log_scale+=log_scale
        return h_i*np.exp(-x**2/2+sum_log_scale)

    def derivative(self, x, k=1):
        if k == 1:
            return x*HermiteFunc(self.n)(x) - \
              np.sqrt(2*(self.n + 1))*HermiteFunc(self.n + 1)(x)
        elif k == 2:
            return HermiteFunc(self.n)(x) + \
              x*HermiteFunc(self.n).derivative(x) - \
              np.sqrt(2*(self.n + 1))*HermiteFunc(self.n + 1).derivative(x)
        else:
            print("Not implemented!")
