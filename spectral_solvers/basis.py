#!/usr/bin/python

import numpy as np
from enum import Enum
import h5py as h5

from .hermite import HermiteFunc
from .laguerre import LaguerreFunc

class BoundaryCondition(Enum):
    NONE = 0
    DIRICHLET = 1
    NEUMANN = 2

class ChebychevBasis():
    """
    Chebychev basis for boundary-value problems

    """
    def __init__(self, N, boundary_conditions):
        self.N = N
        self.bc = boundary_conditions
        self.experiment = True

        # cos(theta) are the collocation points
        self.theta = np.pi*np.arange(N - 1, -1, -1)/(N - 1)

        # Collocation points (including end points)
        self.x = np.cos(self.theta)

        # Helper array: (-1)^n
        self.pm = np.ones((N + 2))
        self.pm[1::2] = -self.pm[1::2]

        self.start_n = 0
        self.end_n = len(self.x) - 2
        if self.bc == [BoundaryCondition.DIRICHLET, BoundaryCondition.DIRICHLET]:
            self.start_n = 2
            self.end_n = len(self.x)
        if self.bc == [BoundaryCondition.NEUMANN, BoundaryCondition.NEUMANN]:
            self.start_n = 0
            self.end_n = len(self.x) - 2
        if self.bc == [BoundaryCondition.DIRICHLET, BoundaryCondition.NEUMANN]:
            self.start_n = 1
            self.end_n = len(self.x) - 1
        if self.bc == [BoundaryCondition.NEUMANN, BoundaryCondition.DIRICHLET]:
            self.start_n = 1
            self.end_n = len(self.x) - 1


    def T(self, n, k=0):
        '''kth derivative of T_n(x) for all collocation points'''
        if k == 0:
            return np.cos(n*self.theta)
        elif k == 1:
            ret = n*np.sin(n*self.theta)

            # Need to take special care of +-1 because of 0/0
            # x = -1
            ret[0] = -self.pm[n]*n*n
            # x = 1
            ret[-1] = n*n
            # |x| not equal to 1
            ret[1:-1] = ret[1:-1]/np.sqrt(1 - self.x[1:-1]*self.x[1:-1])

            return ret
        elif k == 2:
            ret = n*self.x*np.sin(n*self.theta)
            # Need to take special care of +-1 because of 0/0
            # x = -1
            ret[0] = self.pm[n]*n*n*(n*n - 1)/3
            # x = 1
            ret[-1] = n*n*(n*n - 1)/3
            # |x| not equal to 1
            ret[1:-1] = \
              (ret[1:-1]/np.sqrt(1 - self.x[1:-1]*self.x[1:-1]) - \
               n*n*np.cos(n*self.theta[1:-1]))/(1 - self.x[1:-1]*self.x[1:-1])

            return ret
        else:
            print('Higher derivatives of T not implemented!')

    def collocation_points(self):
        if self.experiment is True:
            return self.x[1:-1]

        if self.bc[0] == BoundaryCondition.DIRICHLET:
            if self.bc[1] != BoundaryCondition.DIRICHLET:
                # Homogeneous Dirichlet boundary left, other right
                return self.x[1:]
            else:
                # Dirichlet left and right
                return self.x[1:-1]
        elif self.bc[1] == BoundaryCondition.DIRICHLET:
            # Homogeneous Dirichlet boundary right:
            return self.x[:-1]
        else:
            return self.x

    def evaluate(self, n, k=0):
        '''Evaluate kth derivative of basis functions at collocation points'''
        if self.experiment is True:
            if self.bc[0] == BoundaryCondition.DIRICHLET:
                if self.bc[1] == BoundaryCondition.NEUMANN:
                    return self.T(n,k)[1:-1] - \
                      n*n*self.T(n+1,k)[1:-1]/(n+1)/(n+1) -\
                      self.pm[n]*self.T(0, k)[1:-1]*(n*n/(n+1)/(n+1) + 1)
                elif self.bc[1] == BoundaryCondition.DIRICHLET:
                    r = n % 2
                    return self.T(n,k)[1:-1] - r*self.T(1,k)[1:-1] + \
                      (r - 1)*self.T(0, k)[1:-1]
                else:
                    # No BC on right, homogeneous Dirichlet boundary left:
                    # phi_n = T_n - (-1)^n
                    # n = 1, 2, 3, ...
                    return self.T(n,k)[1:-1] - self.pm[n]*self.T(0, k)[1:-1]
            elif self.bc[0] == BoundaryCondition.NEUMANN:
                if self.bc[1] == BoundaryCondition.NEUMANN:
                    if n == 0:
                        return self.T(0,k)[1:-1]
                    elif n % 2 == 0:
                        return self.T(n,k)[1:-1] - \
                          (n*n/(n+2)/(n+2))*self.T(n+2,k)[1:-1]
                    else:
                        return self.T(n,k)[1:-1] - \
                          (n*n/(n+2)/(n+2))*self.T(n+2,k)[1:-1]
                elif self.bc[1] == BoundaryCondition.DIRICHLET:
                    return self.T(n,k)[1:-1] - self.T(0,k)[1:-1] + \
                      n*n*(self.T(n+1,k)[1:-1] - self.T(0,k)[1:-1])/(n+1)/(n+1)
                else:
                    # No BC on right, Neumann on left
                    return self.T(n,k)[1:-1] + \
                      n*n*self.T(n+1,k)[1:-1]/(n+1)/(n+1)
            else:
                # No BC specified on left
                if self.bc[1] == BoundaryCondition.NEUMANN:
                    # No BC on left, Neumann on right
                    return self.T(n,k)[1:-1] - \
                      n*n*self.T(n+1,k)[1:-1]/(n+1)/(n+1)
                elif self.bc[1] == BoundaryCondition.DIRICHLET:
                    # Homogeneous Dirichlet boundary right:
                    # phi_n = T_n - 1
                    # n = 1, 2, 3, ...
                    return self.T(n,k)[1:-1] - self.T(0,k)[1:-1]
                else:
                    # NO BC SPECIFIED AT ALL
                    return self.T(n,k)[1:-1]

        if self.bc[0] == BoundaryCondition.DIRICHLET:
            if self.bc[1] == BoundaryCondition.NEUMANN:
                return self.T(n,k)[1:] - n*n*self.T(n+1,k)[1:]/(n+1)/(n+1) +\
                  self.pm[n]*(n*n/(n+1)/(n+1) - 1)
            elif self.bc[1] == BoundaryCondition.DIRICHLET:
                r = n % 2
                return self.T(n,k)[1:-1] - r*self.T(1,k)[1:-1] + \
                  (r - 1)*self.T(0, k)[1:-1]
            else:
                # No BC on right, homogeneous Dirichlet boundary left:
                # phi_n = T_n - (-1)^n
                # n = 1, 2, 3, ...
                return self.T(n,k)[1:] - self.pm[n]*self.T(0, k)[1:]
        elif self.bc[0] == BoundaryCondition.NEUMANN:
            if self.bc[1] == BoundaryCondition.NEUMANN:
                if n == 0:
                    return self.T(0,k)
                elif n % 2 == 0:
                    return self.T(n,k) - (n*n/(n+2)/(n+2))*self.T(n+2,k)
                else:
                    return self.T(n,k) - (n*n/(n+2)/(n+2))*self.T(n+2,k)
            elif self.bc[1] == BoundaryCondition.DIRICHLET:
                return self.T(n,k)[:-1] - self.T(0,k)[:-1] + \
                  n*n*(self.T(n+1,k)[:-1] - self.T(0,k)[:-1])/(n+1)/(n+1)
            else:
                # No BC on right, Neumann on left
                return self.T(n,k) + n*n*self.T(n+1,k)/(n+1)/(n+1)
        else:
            # No BC specified on left
            if self.bc[1] == BoundaryCondition.NEUMANN:
                # No BC on left, Neumann on right
                return self.T(n,k) - n*n*self.T(n+1,k)/(n+1)/(n+1)
            elif self.bc[1] == BoundaryCondition.DIRICHLET:
                # Homogeneous Dirichlet boundary right:
                # phi_n = T_n - 1
                # n = 1, 2, 3, ...
                return self.T(n,k)[:-1] - self.T(0,k)[:-1]
            else:
                # NO BC SPECIFIED AT ALL
                return self.T(n,k)

    def interpolate(self, n, x, k=0):
        '''Evaluate basis functions at general x (derivatives up to 2)'''

        # Make sure |x| < 1 for derivatives
        s = 1.0e-5
        #x = x - s*s*x/(np.abs(x) - 1 + s)/(np.abs(x) + s)
        x = x - np.sign(x)*s*s/(np.abs(x-1) + s)

        # Zeroth, first or second derivative of T's
        if k == 0:
            T = lambda m: np.cos(m*np.arccos(x))
        if k == 1:
            T = lambda m: m*np.sin(m*np.arccos(x))/np.sqrt(1 - x*x)
        if k == 2:
            T = lambda m: (m*x*np.sin(m*np.arccos(x))/np.sqrt(1 - x*x) - \
                               m*m*np.cos(m*np.arccos(x)))/(1 - x*x)

        if self.bc[0] == BoundaryCondition.DIRICHLET:
            if self.bc[1] == BoundaryCondition.NEUMANN:
                return T(n) - n*n*T(n+1)/(n+1)/(n+1) -\
                  self.pm[n]*T(0)*(n*n/(n+1)/(n+1) + 1)
            elif self.bc[1] == BoundaryCondition.DIRICHLET:
                r = n % 2
                return T(n) - r*T(1) + (r - 1)*T(0)
            else:
                # No BC on right, homogeneous Dirichlet boundary left:
                # phi_n = T_n - (-1)^n
                # n = 1, 2, 3, ...
                return T(n) - self.pm[n]*T(0)
        elif self.bc[0] == BoundaryCondition.NEUMANN:
            if self.bc[1] == BoundaryCondition.NEUMANN:
                if n == 0:
                    return T(0)
                elif n % 2 == 0:
                    return T(n) - (n*n/(n+2)/(n+2))*T(n+2)
                else:
                    return T(n) - (n*n/(n+2)/(n+2))*T(n+2)
            elif self.bc[1] == BoundaryCondition.DIRICHLET:
                return T(n) - T(0) + \
                  n*n*(T(n+1) - T(0))/(n+1)/(n+1)
            else:
                # No BC on right, Neumann on left
                return T(n) + n*n*T(n+1)/(n+1)/(n+1)
        else:
            # No BC specified on left
            if self.bc[1] == BoundaryCondition.NEUMANN:
                # No BC on left, Neumann on right
                return T(n) - n*n*T(n+1)/(n+1)/(n+1)
            elif self.bc[1] == BoundaryCondition.DIRICHLET:
                # Homogeneous Dirichlet boundary right:
                # phi_n = T_n - 1
                # n = 1, 2, 3, ...
                return T(n) - T(0)
            else:
                # NO BC SPECIFIED AT ALL
                return T(n)

    def summation(self, coef, t, k=0):
        # Use same data type as coef
        u = np.zeros((len(t)), dtype=coef.dtype)
        for n in range(self.start_n, self.end_n):
            u += coef[n - self.start_n]*self.interpolate(n, t, k=k)

        return u

    def derivative_matrices(self):
        N = len(self.collocation_points())

        # Matrix with psi_n(t_i) in the nth column
        A = np.zeros((N, N))
        # Matrix with d_z psi_n(t_i) in the nth column
        dA = np.zeros((N, N))
        # Matrix with d_z^2 psi_n)(t_i) in the nth column
        ddA = np.zeros((N, N))

        for n in range(self.start_n, self.end_n):
            i = n - self.start_n
            A[:, i] = self.evaluate(n)
            dA[:, i] = self.evaluate(n, 1)
            ddA[:, i] = self.evaluate(n, 2)

        return A, dA, ddA

class HermiteBasis():
    """
    Hermite basis for boundary-value problems

    """
    def __init__(self, N, L=1, symmetry=None, filename=None):
        self.N = N
        self.start_n = 0
        self.end_n = N

        self.L = L

        self.symmetry = symmetry

        # Collocation points
        self.x = np.polynomial.hermite.hermroots(np.eye(N+1)[-1,:])

        # Only use
        if symmetry == 'even' or symmetry == 'odd':
            self.x = np.polynomial.hermite.hermroots(np.eye(2*N+1)[-1,:])
            self.x = self.x[-N:]

        self.filename = filename

    def collocation_points(self):
        return self.x

    def evaluate(self, n, k=0):
        '''Evaluate kth derivative of basis functions at collocation points'''

        return self.interpolate(n, self.x, k=k)

    def interpolate(self, n, x, k=0):
        '''Evaluate basis functions at general x'''

        if self.symmetry == 'even':
            n = 2*n
        if self.symmetry == 'odd':
            n = 2*n + 1

        if k == 0:
            return HermiteFunc(n)(x)

        return HermiteFunc(n).derivative(x, k=k)

    def summation(self, coef, t, k=0):
        u = np.zeros((len(t)), dtype=np.cdouble)
        for n in range(self.start_n, self.end_n):
            u += coef[n]*self.interpolate(n, t, k=k)

        return u

    def derivative_matrices(self):
        N = len(self.collocation_points())
        N_string = str(N)

        found_in_file = False
        if self.filename is not None:
            # Try and read from HDF file
            with h5.File(self.filename, 'a') as hf:
                if N_string in hf:
                    found_in_file = True
                    #print('Found matrices in file for N =', N)
                    g = hf[N_string]
                    A = g.get('A')[()]
                    dA = g.get('dA')[()]
                    ddA = g.get('ddA')[()]

        if found_in_file is False:
            # Matrix with psi_n(t_i) in the nth column
            A = np.zeros((N, N))
            # Matrix with d_z psi_n(t_i) in the nth column
            dA = np.zeros((N, N))
            # Matrix with d_z^2 psi_n)(t_i) in the nth column
            ddA = np.zeros((N, N))

            for n in range(self.start_n, self.end_n):
                i = n - self.start_n
                A[:, i] = self.evaluate(n)
                dA[:, i] = self.evaluate(n, 1)
                ddA[:, i] = self.evaluate(n, 2)

            # Write to HDF file
            if self.filename is not None:
                #print('Saving matrices in file for N =', N)
                with h5.File(self.filename, 'a') as hf:
                    g = hf.create_group(N_string)
                    g.create_dataset('A', data=A)
                    g.create_dataset('dA', data=dA)
                    g.create_dataset('ddA', data=ddA)

        return A, dA, ddA

class LaguerreBasis():
    """
    Laguerre basis for boundary-value problems

    """
    def __init__(self, N, boundary_condition, L=1):
        self.N = N
        self.start_n = 1
        self.end_n = N
        self.bc = boundary_condition

        self.L = L

        # Collocation points
        self.x = np.polynomial.laguerre.lagroots(np.eye(N+1)[-1,:])[1:]

    def collocation_points(self):
        return self.x

    def evaluate(self, n, k=0):
        '''Evaluate kth derivative of basis functions at collocation points'''
        return self.interpolate(n, self.x, k=k)

    def interpolate(self, n, x, k=0):
        '''Evaluate basis functions at general x'''

        if self.bc == BoundaryCondition.DIRICHLET:
            return LaguerreFunc(n).derivative(x, k=k) - \
              LaguerreFunc(0).derivative(x, k=k)
        elif self.bc == BoundaryCondition.NEUMANN:
            return LaguerreFunc(n).derivative(x, k=k) - \
              (2*n + 1)*LaguerreFunc(0).derivative(x, k=k)
        else:
            return LaguerreFunc(n).derivative(x, k=k)

    def summation(self, coef, t, k=0):
        u = np.zeros((len(t)), dtype=np.cdouble)
        for n in range(self.start_n, self.end_n):
            u += coef[n - self.start_n]*self.interpolate(n, t, k=k)

        return u

    def derivative_matrices(self):
        N = len(self.collocation_points())

        # Matrix with psi_n(t_i) in the nth column
        A = np.zeros((N, N))
        # Matrix with d_z psi_n(t_i) in the nth column
        dA = np.zeros((N, N))
        # Matrix with d_z^2 psi_n)(t_i) in the nth column
        ddA = np.zeros((N, N))

        for n in range(self.start_n, self.end_n):
            i = n - self.start_n
            A[:, i] = self.evaluate(n)
            dA[:, i] = self.evaluate(n, 1)
            ddA[:, i] = self.evaluate(n, 2)

        return A, dA, ddA
