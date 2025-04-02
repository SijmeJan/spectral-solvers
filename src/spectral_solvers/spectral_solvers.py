import numpy as np
from scipy.linalg import eigvals, eig
import scipy.sparse as sparse
from scipy.sparse.linalg import LinearOperator, gmres, cgs

#import matplotlib.pyplot as plt

try:
    from petsc4py import PETSc
    from slepc4py import SLEPc
    can_use_slepc = True
except ImportError:
    print('Can not import slepc/petc not installed, will use scipy', flush=True)
    can_use_slepc = False

from .basis import ChebychevBasis, BoundaryCondition, HermiteBasis, LaguerreBasis
from .mapping import CoordinateMap, ChebychevMap

def safe_eval_evec(eval_low, evec_low, eval_hi, evec_hi,
                   drift_threshold=1e6, use_ordinal=False,
                   degeneracy=1):
    # Returns 'safe' eigenvalues and eigenvectors using two resolutions
    # Based on the eigentools package
    # (Oishi et al 2021, doi:10.21105/joss.03079)

    #print('safe_eval_evec: ', len(eval_low), len(eval_hi), flush=True)
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

    # Should be the same length, unless sparse solver did not converge
    n_eig = np.min([len(eval_low_sorted), len(eval_hi_sorted)])

    sigmas = np.zeros(n_eig)
    if len(sigmas) > degen:
        sigmas[0:degen] = np.abs(eval_low_sorted[0:degen] - \
                                 eval_low_sorted[degen:2*degen])
        sigmas[degen:-degen] = \
          [0.5*(np.abs(eval_low_sorted[j] - \
                eval_low_sorted[j - degen]) + \
                np.abs(eval_low_sorted[j + degen] - eval_low_sorted[j])) \
                for j in range(degen, n_eig - degen)]
        sigmas[-degen:] = np.abs(eval_low_sorted[-2*degen:-degen] - \
                                 eval_low_sorted[-degen:])
    else:
        sigmas += 1.0

    # Ordinal delta
    delta_ordinal = np.array([np.abs(eval_low_sorted[j] - \
                              eval_hi_sorted[j])/sigmas[j] \
                              for j in range(n_eig)])

    # Nearest delta
    small = 1.0e-16
    delta_near = \
      np.array([np.nanmin(np.abs(eval_low_sorted[j] -
                          eval_hi_sorted)/sigmas[j]) + small \
                for j in range(n_eig)])


    # Discard eigenvalues with 1/delta_near < drift_threshold
    if use_ordinal:
        inverse_drift = 1/delta_ordinal
    else:
        inverse_drift = 1/(delta_near + small)
    eval_low_and_indx = \
      eval_low_and_indx[np.where(inverse_drift > drift_threshold)]

    eval_low = eval_low_and_indx[:, 0]
    indx = eval_low_and_indx[:, 1].real.astype(np.int64)

    evec = []
    for i in indx:
        evec.append(evec_low[:, i])

    return eval_low, np.asarray(evec)

class SpectralSolver():
    # Turn a set of equations A*u'' + B*u' + C*u = w*u + E
    # into a matrix equation M*U = N*U + S by decomposing u into
    # spectral basis functions. Here u is the vector of unknowns,
    # A, B and C are matrices, w is

    def __init__(self,
                 interval=[0, 1],
                 symmetry=None,
                 basis='Chebychev',
                 sparse_flag=True,
                 use_PETSc=True):
        self.basis_kind = basis
        self.symmetry = symmetry
        self.interval = interval

        self.sparse_flag = sparse_flag
        self.use_PETSc = use_PETSc
        if can_use_slepc == False:
            self.use_PETSc = False

        self.N = -1

    def set_resolution(self, N, L=1, n_eq=1):
        #print('Number of equations:', n_eq)
        # Coordinate mapping and basis functions
        if self.basis_kind == 'Hermite':
            self.mapping = CoordinateMap(L=L)
            self.basis = HermiteBasis(N, symmetry=self.symmetry,
                                      filename='hermite.h5')
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
        N = len(t)
        self.z = self.mapping.x(t)

        # Construct A, A' and A''
        self.construct_a()

        # Block-diagonal matrix with A's on the diagonal

        # he standard BSR representation where the block column indices for
        # row i are stored in indices[indptr[i]:indptr[i+1]] and their
        # corresponding block values are stored in data[ indptr[i]: indptr[i+1] ].

        # indices = [0, 1, 2, 3, 4]
        # indptr = [0, 1, 2, 3, 4, 5]
        # block column index for row i: indices[i] = i

        indptr = np.asarray(range(0, n_eq + 1))
        indices = np.asarray(range(0, n_eq))
        data = []
        for i in range(0, n_eq):
            data.append(self.A)
        data = np.asarray(data)

        self.B = \
          sparse.bsr_matrix((data, indices, indptr), shape=(n_eq*N, n_eq*N))

        if self.sparse_flag == False:
            self.B = self.B.todense()

    def construct_a(self, filename=None):
        t = self.basis.collocation_points()
        #print('Start constructing A for N = ', len(t))

        dtdz = self.mapping.dtdx(t)
        d2tdz2 = self.mapping.d2tdx2(t)

        N = len(t)

        if N != self.N:
            self.A, self.dA, self.ddA = \
              self.basis.derivative_matrices()

            self.N = N

            self.dA = self.dA*dtdz[:,np.newaxis]
            self.ddA = self.ddA*(dtdz*dtdz)[:,np.newaxis] + \
              self.dA*(d2tdz2/dtdz)[:,np.newaxis]
        #print('Done constructing A for N = ', N)

    def construct_m(self, f, k=0):
        # f should be vector of N-1 elements at z collocation points

        if k > 2:
            print("ERROR: too large k in construct_matrix")

        if k == 2:
            return np.matmul(np.diag(f), self.ddA)
        if k == 1:
            return np.matmul(np.diag(f), self.dA)
        return np.matmul(np.diag(f), self.A)

    def fast_evaluate(self, sol, n_eq=1):
        t = self.mapping.t(self.z)

        L = len(self.basis.collocation_points())
        sol = sol.reshape((n_eq, L))

        for i in range(0, n_eq):
            sol[i,:] = np.matmul(self.A, sol[i,:])

        return t, sol

    def evaluate(self, z, sol, n_eq=1, k=0):
        #print('Number of equations:', n_eq)

        t = self.mapping.t(z)

        L = len(self.basis.collocation_points())
        sol = sol.reshape((n_eq, L))

        # Use same data type as sol
        u = np.zeros((n_eq, len(t)), dtype=sol[0].dtype)
        for m in range(0, n_eq):
            u[m, :] = self.basis.summation(sol[m, :], t, k=k)

        if k == 1:
            dtdx = self.mapping.dtdx(t)
            u = u*dtdx

        if k == 2:
            du = np.zeros((n_eq, len(t)), dtype=sol[0].dtype)
            for m in range(0, n_eq):
                du[m, :] = self.basis.summation(sol[m, :], t, k=1)

            dtdx = self.mapping.dtdx(t)
            d2tdx2 = self.mapping.d2tdx2(t)

            u = dtdx*dtdx*u + d2tdx2*du

        z = self.mapping.x(t)

        return self.transform(z, u, n_eq=n_eq)

    def transform(self, z, u, n_eq=1):
        #print('Number of equations:', n_eq)
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
        return self.matrixP() + self.matrixQ() + self.matrixR()

    def solve(self, N, L=1, n_eq=1, **kwargs):
        #print('Number of equations:', n_eq)

        #print('setting resolution')
        self.set_resolution(N, L=L, n_eq=n_eq)

        #print('calculating matrices')
        M = self.matrixM(**kwargs)
        S = self.vectorS(**kwargs)

        #print('solving')
        return np.linalg.solve(M, S)

class EigenValueSolver(SpectralSolver):
    def matrixM(self):
        return self.ddA

    def solve(self, N, L=1, n_eq=1,
              sigma=None, n_eig=6,
              **kwargs):
        #print('Number of equations:', n_eq)

        self.set_resolution(N, L=L, n_eq=n_eq)

        # Try and find *all* eigenvalues
        if n_eig < 0:
            n_eig = n_eq*N

        # Construct left-hand side matrix
        M = self.matrixM(**kwargs)

        if self.sparse_flag == True:
            if self.use_PETSc == True:
                petsc_M = PETSc.Mat().createBAIJ(size=M.shape,
                                                 bsize=M.blocksize,
                                                 csr=(M.indptr,
                                                      M.indices,
                                                      M.data))
                #petsc_B = PETSc.Mat().createBAIJ(size=self.B.shape,
                #                                 bsize=self.B.blocksize,
                #                                 csr=(self.B.indptr,
                #                                      self.B.indices,
                #                                      self.B.data))

                SLEPcSolver = SLEPc.EPS()
                SLEPcSolver.create()

                #SLEPcSolver.setType(SLEPc.EPS.Type.CISS)
                #R = SLEPc.RG()
                #R.create()
                #R.setType('ellipse')
                #R.setEllipseParameters(0.0+0.001j, 2.0)
                #SLEPcSolver.setRG(R)

                SLEPcSolver.setDimensions(nev=n_eig)
                shift = SLEPc.ST().create()
                shift.setType(SLEPc.ST.Type.SINVERT)
                SLEPcSolver.setST(shift)
                SLEPcSolver.setTarget(sigma)

                SLEPcSolver.setOperators(petsc_M)#, petsc_B)
                SLEPcSolver.setProblemType(SLEPc.EPS.ProblemType.NHEP)

                SLEPcSolver.solve()

                nconv = SLEPcSolver.getConverged()

                # Create the results vectors
                vr, wr = petsc_M.getVecs()
                vi, wi = petsc_M.getVecs()

                n_found = np.min([nconv, n_eig])
                eigenvalues = np.zeros((n_found), dtype=np.cdouble)
                eigenvectors = np.zeros((N*n_eq, n_found), dtype=np.cdouble)

                for i in range(0, n_found):
                    k = SLEPcSolver.getEigenpair(i, vr, vi)
                    eigenvalues[i] = k
                    eigenvectors[:, i] = vr.getArray() + 1j*vi.getArray()

                ret = eigenvalues, eigenvectors
            else:
                ret = sparse.linalg.eigs(M, k=n_eig, M=None,
                                         sigma=sigma, which='LM')

            return ret
        else:
            return eig(M, self.B)

    def safe_solve(self, N, L=1, n_eq=1,
                   sigma=None, n_eig=6,
                   factor=2, drift_threshold=1e6, use_ordinal=False,
                   degeneracy=1, n_safe_levels=1,
                   **kwargs):
        #print('Number of equations:', n_eq)

        N_high = int(factor**(n_safe_levels)*N)
        eval_hi, evec_hi = self.solve(N_high, L=L, n_eq=n_eq,
                                      sigma=sigma,
                                      n_eig=n_eig, **kwargs)

        self.eval_hires = eval_hi
        self.evec_hires = evec_hi

        for n_safe in range(0, n_safe_levels):
            N_low = int(factor**(n_safe_levels - n_safe - 1)*N)
            eval_low, evec_low = self.solve(N_low, L=L, n_eq=n_eq,
                                            sigma=sigma,
                                            n_eig=n_eig,
                                            **kwargs)

            if len(eval_low) == 0 or len(eval_hi) == 0:
                # Something has gone wrong: no converged eigenvalues
                rad = 0
                if len(eval_low) > 0:
                    rad = np.max(np.abs(sigma - eval_low))
                if len(eval_hi) > 0:
                    rad = np.max(np.abs(sigma - eval_hi))
                return [], [], rad

            eval_hi, evec_hi = \
              safe_eval_evec(eval_low, evec_low, eval_hi, evec_hi,
                             drift_threshold=drift_threshold,
                             use_ordinal=use_ordinal,
                             degeneracy=degeneracy)

        rad = np.max(np.abs(sigma - eval_low))

        return eval_hi, evec_hi, rad

class SpectralSolver2D():
    '''Spectral solvers for 2 independent variables

    Matrix composition: A*B = [[a11*B, a12*B, ...], [a21*B, a22*B, ...], ...]

    '''
    def __init__(self,
                 interval=[[0, 1], [0, 1]],
                 symmetry=[None, None],
                 basis=['Chebychev', 'Chebychev'],
                 sparse_flag=True,
                 use_PETSc=True):

        # Create solver for x direction
        self.spec_sol_x = \
            SpectralSolver(interval[0],
                           symmetry[0],
                           basis[0],
                           sparse_flag,
                           use_PETSc)

        self.spec_sol_z = \
            SpectralSolver(interval[1],
                           symmetry[1],
                           basis[1],
                           sparse_flag,
                           use_PETSc)

        self.sparse_flag = sparse_flag
        self.use_PETSc = use_PETSc
        if can_use_slepc == False:
            self.use_PETSc = False

        self.Nx = -1
        self.Nz = -1

    def set_resolution(self, Nx, Nz, Lx=1, Lz=1, n_eq=1):
        #print('Number of equations:', n_eq)

        self.spec_sol_x.set_resolution(Nx, L=Lx, n_eq=n_eq)
        self.spec_sol_z.set_resolution(Nz, L=Lz, n_eq=n_eq)

        #print('Constructing A')
        #self.construct_a(n_eq=n_eq)

    def mat_vect_combined(self, vec, A, B, f=None):
        '''Matrix vector product linear operator'''
        Nx = len(self.spec_sol_x.basis.collocation_points())
        Nz = len(self.spec_sol_z.basis.collocation_points())

        res = np.zeros(np.shape(vec), dtype=np.complex128)

        for i in range(0, Nx):             # Block row number
            if f is not None:
                _B = np.matmul(np.diag(np.asarray(f(self.spec_sol_x.z[i], self.spec_sol_z.z))), B)
            else:
                _B = B

            for j in range(0, Nx):             # Block column number
                #print('Shapes: ', np.shape(A[i,j]*_B), np.shape(vec[j*Nz:(j+1)*Nz]), np.shape(vec), j*Nz, (j+1)*Nz)
                res[j*Nz:(j+1)*Nz] += np.matmul(A[i,j]*_B, vec[j*Nz:(j+1)*Nz])

        return res

    def mat_vect_part_m(self, vec, kx=0, kz=0, f=None):
        if kx + kz > 2:
            print("ERROR: too large k in construct_matrix")

        if kx == 0:
            if kz == 0:
                return self.mat_vect_combined(vec, self.spec_sol_x.A, self.spec_sol_z.A, f=f)
            if kz == 1:
                return self.mat_vect_combined(vec, self.spec_sol_x.A, self.spec_sol_z.dA, f=f)
            if kz == 2:
                return self.mat_vect_combined(vec, self.spec_sol_x.A, self.spec_sol_z.ddA, f=f)
        if kx == 1:
            if kz == 0:
                return self.mat_vect_combined(vec, self.spec_sol_x.dA, self.spec_sol_z.A, f=f)
            if kz == 1:
                return self.mat_vect_combined(vec, self.spec_sol_x.dA, self.spec_sol_z.dA, f=f)
            if kz == 2:
                return self.mat_vect_combined(vec, self.spec_sol_x.dA, self.spec_sol_z.ddA, f=f)
        if kx == 2:
            if kz == 0:
                return self.mat_vect_combined(vec, self.spec_sol_x.ddA, self.spec_sol_z.A, f=f)
            if kz == 1:
                return self.mat_vect_combined(vec, self.spec_sol_x.ddA, self.spec_sol_z.dA, f=f)
            if kz == 2:
                return self.mat_vect_combined(vec, self.spec_sol_x.ddA, self.spec_sol_z.ddA, f=f)

    def construct_combined(self, A, B, f=None):
        # Discretize a term f(x,z)*d^n W/dx^n
        # Function f must have two parameters, x and z

        Nx = len(self.spec_sol_x.basis.collocation_points())
        Nz = len(self.spec_sol_z.basis.collocation_points())

        indptr = np.asarray(range(0, Nx+1))*Nx

        indices = []
        data = []
        for i in range(0, Nx):             # Block row number
            indices.append(range(0, Nx))
            if f is not None:
                _B = np.matmul(np.diag(np.asarray(f(self.spec_sol_x.z[i], self.spec_sol_z.z))), B)
            else:
                _B = B

            for j in range(0, Nx):         # Block column number
                data.append(A[i,j]*_B)
        data = np.asarray(data)

        indices = np.asarray(indices).flatten()

        ret = \
            sparse.bsr_matrix((data, indices, indptr), shape=(Nx*Nz, Nx*Nz))

        if self.sparse_flag == False:
            ret = ret.todense()

        return ret

    def construct_a(self, n_eq=1, filename=None):
        #print('Number of equations:', n_eq)

        Nx = len(self.spec_sol_x.basis.collocation_points())
        Nz = len(self.spec_sol_z.basis.collocation_points())

        self.A = \
            self.construct_combined(self.spec_sol_x.A, self.spec_sol_z.A)

        # Make sure A is dense....
        if self.sparse_flag == True:
            A = self.A.todense()
        else:
            A = self.A

        # Block-diagonal matrix with A's on the diagonal
        indptr = np.asarray(range(0, n_eq + 1))
        indices = np.asarray(range(0, n_eq))
        data = []
        for i in range(0, n_eq):
            data.append(A)
        data = np.asarray(data)

        self.B = \
          sparse.bsr_matrix((data, indices, indptr), shape=(n_eq*Nx*Nz, n_eq*Nx*Nz))

        if self.sparse_flag == False:
            self.B = self.B.todense()

    def construct_rhs(self, f=None):
        Nx = len(self.spec_sol_x.basis.collocation_points())
        Nz = len(self.spec_sol_z.basis.collocation_points())

        ret = np.zeros((Nx*Nz), dtype=np.complex64)

        if f is not None:
            for i in range(0, Nx):
                x = self.spec_sol_x.z[i]
                z = self.spec_sol_z.z

                ret[Nz*i:Nz*i+Nz] = f(x, z)

        return ret

    def construct_m(self, f=None, kx=0, kz=0):
        '''Distretize a term f(x, z)*d^n W/dx^n'''

        if kx + kz > 2:
            print("ERROR: too large k in construct_matrix")

        if kx == 0:
            if kz == 0:
                return self.construct_combined(self.spec_sol_x.A, self.spec_sol_z.A, f=f)
            if kz == 1:
                return self.construct_combined(self.spec_sol_x.A, self.spec_sol_z.dA, f=f)
            if kz == 2:
                return self.construct_combined(self.spec_sol_x.A, self.spec_sol_z.ddA, f=f)
        if kx == 1:
            if kz == 0:
                return self.construct_combined(self.spec_sol_x.dA, self.spec_sol_z.A, f=f)
            if kz == 1:
                return self.construct_combined(self.spec_sol_x.dA, self.spec_sol_z.dA, f=f)
            if kz == 2:
                return self.construct_combined(self.spec_sol_x.dA, self.spec_sol_z.ddA, f=f)
        if kx == 2:
            if kz == 0:
                return self.construct_combined(self.spec_sol_x.ddA, self.spec_sol_z.A, f=f)
            if kz == 1:
                return self.construct_combined(self.spec_sol_x.ddA, self.spec_sol_z.dA, f=f)
            if kz == 2:
                return self.construct_combined(self.spec_sol_x.ddA, self.spec_sol_z.ddA, f=f)

    def evaluate(self, x, z, sol, n_eq=1, kx=0, kz=0):
        #print('Number of equations:', n_eq)

        Nx = len(self.spec_sol_x.basis.collocation_points())
        Nz = len(self.spec_sol_z.basis.collocation_points())

        u = np.zeros((n_eq, Nx, len(z)), dtype=sol[0].dtype)

        # First do inner sum for z
        for n in range(0, n_eq):
            for i in range(0, Nx):
                w = self.spec_sol_z.evaluate(z, sol[n*Nx*Nz + i*Nz:n*Nx*Nz + i*Nz + Nz], n_eq=1, k=kz)
                u[n, i, :] = w

        ret = np.zeros((n_eq, len(x), len(z)), dtype=sol[0].dtype)

        # Do outer sum for x
        for n in range(0, n_eq):
            for j in range(0, len(z)):
                w = self.spec_sol_x.evaluate(x, u[n,:,j], n_eq=1, k=kx)
                ret[n,:,j] = w

        return ret

    def transform(self, z, u, n_eq=1):
        #print('Number of equations:', n_eq)

        return u

class BoundaryValueProblemSolver2D(SpectralSolver2D):
    # uxx + uyy = S(x,y)
    def vectorS(self, f=None):
        return self.construct_rhs(f)

    def mat_vect_m(self, vec):
        # 2D Laplace operator
        return self.mat_vect_part_m(vec, f=None, kx=2, kz=0) + \
            self.mat_vect_part_m(vec, f=None, kx=0, kz=2)

    def matrixM(self):
        # 2D Laplace operator
        return self.construct_m(f=None, kx=2, kz=0) + \
            self.construct_m(f=None, kx=0, kz=2)

    def solve(self, Nx, Nz, Lx=1, Lz=1, n_eq=1):
        print('Setting resolution...')
        #print('Number of equations:', n_eq)

        self.set_resolution(Nx, Nz, Lx=Lx, Lz=Lz, n_eq=n_eq)

        print('Filling matrices...')
        M = self.matrixM()
        S = self.vectorS()

        print('Solving linear system...')
        return np.linalg.solve(M, S)

    def solve_operator(self, Nx, Nz, Lx=1, Lz=1, n_eq=1):
        #print('Number of equations:', n_eq)

        print('Setting resolution...')
        self.set_resolution(Nx, Nz, Lx=Lx, Lz=Lz, n_eq=n_eq)

        print('Calculating S')
        S = self.vectorS()

        print('Defining new operator')
        Nx = len(self.spec_sol_x.basis.collocation_points())
        Nz = len(self.spec_sol_z.basis.collocation_points())
        op = LinearOperator((n_eq*Nx*Nz, n_eq*Nx*Nz), matvec=self.mat_vect_m)

        ret, info = cgs(op, S)

        return ret

