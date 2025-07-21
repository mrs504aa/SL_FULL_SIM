# Programs used to simulate absorption spectra of superradiance lattices (SLs)
* Files with name ```Type_*_Main.jl``` are programs used to simulate the velocity resolved absorption spectra for different SL configurations.
* Files with name ```Floquet_2D_Solver_*.jl``` are solvers modified from https://github.com/qutip/QuantumToolbox.jl/blob/main/src/steadystate.jl.
* Before using the code, make sure to activate the Julia project environment and install all required packages by running the following steps in REPL:
  ```julia
  using Pkg
  Pkg.activate(".")      # Activate the current project environment
  Pkg.instantiate()      # Install all dependencies specified in the project
  ```
  These steps are essential to ensure that all necessary packages are installed for the code to run properly.
