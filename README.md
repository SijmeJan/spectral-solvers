# Install

Clone the repository:

    $ git clone https://github.com/SijmeJan/spectral-solvers.git

Install with `pip`:

    $ pip install -e ~/path/to/repo

## If you want to use PETSc/SLEPc (recommended)

Clone PETSc repository:

    $ git clone -b release https://gitlab.com/petsc/petsc.git petsc

Follow the instructions
[here](https://petsc.org/release/install/install_tutorial/#qqtw-quickest-quick-start-in-the-west)
to install PETSc. As a minimal installation, go into the `petsc`
directory and do

    ./configure --with-fc=0 --with-scalar-type=complex
    make all check

Make sure the `PETSC_ARCH` and `PETSC_DIR` environment variables are
set.

Download SLEPc from [here](https://slepc.upv.es/download/) and follow
the installation instructions
[here](https://slepc.upv.es/documentation/instal.htm). Make sure the
environment variable `SLEPC_DIR` is set correctly.

Finally, install the Python bindings:

    $ pip install petsc4py slepc4py
