import numpy as np
from scipy.linalg import eigvals, eig
import scipy.sparse as sparse

from .basis import ChebychevBasis, BoundaryCondition, HermiteBasis, LaguerreBasis
from .mapping import CoordinateMap, ChebychevMap

class SpectralSolver():
    # Turn a set of equations A*u'' + B*u' + C*u = w*u + E
    # into a matrix equation M*U = N*U + S by decomposing u into
    # spectral basis functions. Here u is the vector of unknowns,
    # A, B and C are matrices, w is

    def __init__(self,
                 interval=[0, 1],
                 symmetry=None,
                 basis='Chebychev'):
        self.basis_kind = basis
        self.symmetry = symmetry
        self.interval = interval

    def set_resolution(self, N, L=1, n_eq=1):
        # Coordinate mapping and basis functions
        if self.basis_kind == 'Hermite':
            self.mapping = CoordinateMap(L=L)
            self.basis = HermiteBasis(N, symmetry=self.symmetry)
        elif self.basis_kind == 'Laguerre':
            self.mapping = CoordinateMap(L=L)
            if self.symmetry == 'even':
                self.basis = LaguerreBasis(N, BoundaryCondition.NEUMANN)
            else:
                self.basis = LaguerreBasis(N, BoundaryCondition.DIRICHLET)
        else:
            # Mapping to [-1, 1]
            self.mapping = ChebychevMap(x_interval=self.interval, L=L)

            # Basis functions with appropriate BCs
            if self.symmetry == 'even':
                self.basis = ChebychevBasis(N, [BoundaryCondition.NEUMANN,
                                                BoundaryCondition.DIRICHLET])
            else:
                self.basis = ChebychevBasis(N, [BoundaryCondition.DIRICHLET,
                                                BoundaryCondition.DIRICHLET])

        # Collocation points in t and z
        t = self.basis.collocation_points()
        self.z = self.mapping.x(t)

        # Construct A, A' and A''
        self.construct_a()

        # Block-diagonal matrix with A's on the diagonal
        self.B = self.A
        for i in range(1, n_eq):
            # Zeros in upper right and lower left
            UR = np.zeros((np.shape(self.B)[0], np.shape(self.A)[0]))
            LL = np.zeros((np.shape(self.A)[0], np.shape(self.B)[1]))
            # Add block
            self.B = np.block([[self.B, UR], [LL, self.A]])

    def construct_a(self):
        t = self.basis.collocation_points()

        dtdz = self.mapping.dtdx(t)
        d2tdz2 = self.mapping.d2tdx2(t)

        N = len(t)

        # Matrix with psi_n(t_i) in the nth column
        self.A = np.zeros((N, N))
        # Matrix with dtdz(t_i)*d_z psi_n in the nth column
        self.dA = np.zeros((N, N))
        # Matrix with (d2tdz2*d_z psi_n + (dtdz)**2*d_z^2 psi_n)(t_i)
        self.ddA = np.zeros((N, N))

        for n in range(self.basis.start_n, self.basis.end_n):
            i = n - self.basis.start_n
            self.A[:, i] = self.basis.evaluate(n)
            self.dA[:, i] = dtdz*self.basis.evaluate(n, 1)
            self.ddA[:, i] = \
              d2tdz2*self.basis.evaluate(n, 1) + \
              dtdz*dtdz*self.basis.evaluate(n, 2)

    def construct_m(self, f, k=0):
        # f should be vector of N-1 elements at z collocation points

        if k > 2:
            print("ERROR: too large k in construct_matrix")

        if k == 2:
            return np.matmul(np.diag(f), self.ddA)
        if k == 1:
            return np.matmul(np.diag(f), self.dA)
        return np.matmul(np.diag(f), self.A)

    def evaluate(self, z, sol, n_eq=1, k=0):
        t = self.mapping.t(z)

        L = len(self.basis.collocation_points())
        sol = sol.reshape((n_eq, L))

        # Use same data type as sol
        u = np.zeros((n_eq, len(t)), dtype=sol[0].dtype)
        for m in range(0, n_eq):
            u[m, :] = self.basis.summation(sol[m, :], t, k=k)

        z = self.mapping.x(t)

        return self.transform(z, u, n_eq=n_eq)

    def transform(self, z, u, n_eq=1):
        return u

class BoundaryValueProblemSolver(SpectralSolver):
    # P(t)u'' + Q(t)u' + R(t)u = S(t)
    def matrixP(self):
        return self.ddA

    def matrixQ(self):
        return self.dA

    def matrixR(self):
        return self.A

    def vectorS(self):
        return self.z

    def matrixM(self):
        return self.matrixP + self.matrixQ + self.matrixR

    def solve(self, N, L=1, n_eq=1, **kwargs):
        self.set_resolution(N, L=L, n_eq=n_eq)

        M = self.matrixM(**kwargs)
        S = self.vectorS(**kwargs)

        return np.linalg.solve(M, S)

class EigenValueSolver(SpectralSolver):
    def matrixM(self):
        return self.ddA

    def solve(self, N, L=1, n_eq=1, **kwargs):
        self.set_resolution(N, L=L, n_eq=n_eq)

        # Construct left-hand side matrix
        M = self.matrixM(**kwargs)

        use_sparse = False

        if use_sparse is True:
            M_csr = sparse.csr_matrix(M)
            B_csr = sparse.csr_matrix(self.B)

            return sparse.linalg.eigs(M_csr, M=B_csr, which='SM')
        else:
            return eig(M, self.B)

    def safe_eval_evec(self, eval_low, evec_low, eval_hi, evec_hi,
                       drift_threshold=1e6, use_ordinal=False,
                       degeneracy=1):
        # Returns 'safe' eigenvalues and eigenvectors using two resolutions

        # Reverse engineer correct indices to make unsorted list from sorted
        reverse_eval_low_indx = np.arange(len(eval_low))
        reverse_eval_hi_indx = np.arange(len(eval_hi))

        eval_low_and_indx = \
          np.asarray(list(zip(eval_low, reverse_eval_low_indx)))
        eval_hi_and_indx = np.asarray(list(zip(eval_hi, reverse_eval_hi_indx)))

        # remove nans
        eval_low_and_indx = eval_low_and_indx[np.isfinite(eval_low)]
        eval_hi_and_indx = eval_hi_and_indx[np.isfinite(eval_hi)]

        # Sort eval_low and eval_hi by real parts
        eval_low_and_indx = \
          eval_low_and_indx[np.argsort(eval_low_and_indx[:, 0].real)]
        eval_hi_and_indx = \
          eval_hi_and_indx[np.argsort(eval_hi_and_indx[:, 0].real)]

        eval_low_sorted = eval_low_and_indx[:, 0]
        eval_hi_sorted = eval_hi_and_indx[:, 0]

        # Compute sigmas from lower resolution run (gridnum = N1)
        degen = degeneracy
        sigmas = np.zeros(len(eval_low_sorted))
        sigmas[0:degen] = np.abs(eval_low_sorted[0:degen] - \
                                 eval_low_sorted[degen:2*degen])
        sigmas[degen:-degen] = \
          [0.5*(np.abs(eval_low_sorted[j] - \
                       eval_low_sorted[j - degen]) + \
                np.abs(eval_low_sorted[j + degen] - eval_low_sorted[j])) \
                for j in range(degen, len(eval_low_sorted) - degen)]
        sigmas[-degen:] = np.abs(eval_low_sorted[-2*degen:-degen] - \
                                 eval_low_sorted[-degen:])

        if not (np.isfinite(sigmas)).all():
            logger.warning("At least one eigenvalue spacings (sigmas) is non-finite (np.inf or np.nan)!")

        # Ordinal delta
        self.delta_ordinal = np.array([np.abs(eval_low_sorted[j] - \
                                       eval_hi_sorted[j])/sigmas[j] \
                                        for j in range(len(eval_low_sorted))])

        # Nearest delta
        small = 1.0e-16
        self.delta_near = \
          np.array([np.nanmin(np.abs(eval_low_sorted[j] -
                              eval_hi_sorted)/sigmas[j]) + small \
                        for j in range(len(eval_low_sorted))])


        # Discard eigenvalues with 1/delta_near < drift_threshold
        if use_ordinal:
            inverse_drift = 1/self.delta_ordinal
        else:
            inverse_drift = 1/(self.delta_near + small)
        eval_low_and_indx = \
          eval_low_and_indx[np.where(inverse_drift > drift_threshold)]

        eval_low = eval_low_and_indx[:, 0]
        indx = eval_low_and_indx[:, 1].real.astype(np.int)

        evec = []
        for i in indx:
            evec.append(evec_low[:, i])

        return eval_low, evec

    def safe_solve(self, N, L=1, n_eq=1, factor=2,
                   drift_threshold=1e6, use_ordinal=False,
                   degeneracy=1, **kwargs):
        eval_hi, evec_hi = self.solve(int(factor*N), L=L, n_eq=n_eq, **kwargs)

        eval_low, evec_low = self.solve(N, L=L, n_eq=n_eq, **kwargs)

        return self.safe_eval_evec(eval_low, evec_low, eval_hi, evec_hi,
                                   drift_threshold=drift_threshold,
                                   use_ordinal=use_ordinal,
                                   degeneracy=degeneracy)
