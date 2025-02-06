# Vertical Mixing Coefficients

## 1 Overview

Representation of unresolved vertical fluxes of momentum, heat, salt, and biogeochemical tracers in ocean models is essential to simulations fidelity. Models of turbulent fluxes spans a wide range of complexity, but the models generally fall into a few categories: simply polynomial relationships, equilibrium turbulence models, and prognostic turbulence models.  

## 2 Requirements

### 2.1 Requirement: Vertical mixing interface should be modular

To prepare for additional mixing closures and later improvements, the interface to vertical mixing must allow for easy connection of closures. It is assumed that the strength of vertical mixing from each closure is additive, such that the final vertical diffusion coefficient is the sum of the coefficients and potential non local terms from each closure.

### 2.2 Requirement: Vertical mixing models should be have local and/or gradient free terms

For optimal performance, initial models of vertical turbulent fluxes for Omega should be cast as an implicit, down gradient mixing and an explicit, gradient free, component, e.g.,

$$
\overline{w' \phi'}\approx \kappa(\overline{\rho},\overline{u},\overline{v}) \left(\frac{\partial \overline{\phi}}{\partial z} + \gamma(\overline{\rho},\overline{u},\overline{v}) \right)
$$

Here, $\phi$ is a generic tracer, $\overline{w'\phi'}$ is the vertical turbulent flux of that tracer, and $\gamma$ is the gradient free portion of the flux (often referred to as 'non-local'). The vertical diffusivity can be as simple as a constant value, to complex functions of quantities like shear and stratification. A similar equation can be written for turbulent momentum fluxes.

The use of a flux term proportional to the local gradient allows this problem to be cast implicitly, allowing for larger time steps, reducing the cost of the mixing model.  When more complex schemes are considered, other techniques such as subcycling can be considered to improve performance.

### 2.3 Requirement: Vertical mixing models cannot ingest single columns at a time

Many standard vertical mixing libraries (e.g. CVMix) receive and operate on one column at a time. This results in poor performance, especially on accelerators. Any utilized model of vertical mixing must ingest and operate on multiple columns concurrently.

## 3 Algorithmic Formulation

For Omega-0 a few simple vertical mixing algorithms will be used.  

### 3.1 Richardson number dependent mixing

One of the simplest class of models of vertical turbulence are polynomial functions of the gradient Richardson number. As in MPAS-Ocean, we choose to define the Richardson number as

$$
Ri = \frac{N^2}{\left|\frac{\partial \mathbf{U}}{\partial z}\right|^2}
$$

where $N^2$ is the Brunt Vaisala Frequency, which for non Boussinesq flows is defined as

$$
N^2 = \frac{g}{\rho}\frac{\partial \rho}{\partial z}
$$

This term is discretized as

$$
N^2(k) = g \frac{\ln \rho_{DD}(k) - \ln \rho(k)}{z_m(k-1) - z_m(k)}
$$

Here $\rho_{DD}$ is the density of the fluid of layer k-1 displaced to layer k adiabatically.

The shear in the denominator is discretized as

$$
\left| \frac{\partial \mathbf{U}}{\partial z}\right|^2 = \left(\frac{U_n(k-1)-U_n(k)}{z_m(k-1) - z_m(k)}\right)^2 + \left(\frac{U_t(k-1)-U_t(k)}{z_m(k-1) - z_m(k)}\right)^2
$$

Where the subscripts *n* and *t* are the normal and tangential velocities respectively.

With the definition of the gradient richardson number, the viscosity and diffusivity are defined according to [Pacanowski and Philander (1981)](https://journals.ametsoc.org/view/journals/phoc/11/11/1520-0485_1981_011_1443_povmin_2_0_co_2.xml?tab_body=pdf) as

$$
\nu = \frac{\nu_o}{(1+\alpha Ri)^n} + \nu_b
$$

$$
\kappa = \frac{\nu}{(1+\alpha Ri)} + \kappa_b
$$

in these formulae, $\alpha$ and *n* are tunable parameters, most often chosen as 5 and 2 respectively. $\nu_b$ and $\kappa_b$ are the background viscosity and diffusivity respectively. MPAS-Ocean has a $\kappa_{b,passive}$ for applying to just the passive tracers, leaving the active tracers with $\kappa_b$.

### 3.2 Convective Instability Mixing

Commonly, mixing due to convective instability is treated as a step function for the diffusivity and viscosity. This is often represented as

$$
\kappa =
\begin{cases}
\kappa_{conv} \quad \text{ if } N^2 \leq N^2_{crit}\\
0 \quad \text{ if } N^2 > N^2_{crit}
\end{cases}
$$

A similar expression is utilized for viscosity. The effect of this formula is to homogenize T, S, passive tracers, and normal velocity, for unstable stratification, but also in neutral stratification. The behavior in unstable stratification is likely correct, but for neutral stratification, homogenization of momentum is questionable. To allow for a different behavior, we slightly modifty the algorithm above to,

$$
\kappa =
\begin{cases}
\kappa_{conv} \quad \text{ if } N^2 < N^2_{crit}\\
0 \quad \text{ if } N^2 \geq N^2_{crit}
\end{cases}
$$

$N^2_{crit}$ is typically chosen to be 0.

## 4 Design

The current vertical chunking does not work well for vertical derivatives due to the potential for single layers in each chunk, so a first design of the vertical mixing coefficients computation does not do vertical chunking and just computes with the whole column. 

To start, only the down gradient contribution to vertical mixing will be added, with contributions to the vertical viscosity and diffusivity coming from the shear (Richardson number mixing), convective, and background mixing models detailed in the prior section. In future developments, the K Profile Parameterization [(KPP; Large et al., 1994)](https://agupubs.onlinelibrary.wiley.com/doi/abs/10.1029/94rg01872) and other non-local and/or higher-order mixing models will be added. Note that these parameterizations require vertical derivatives and integrals, similarly to those detailed here, thus a solution that allows chunking and vertical operations such as derivatives and integrals would be advantageous.

### 4.1 Data types and parameters

#### 4.1.1 Parameters

The following config options should be included for the Richardson number dependent -- Pacanowski and Philander (1981) based -- vertical mixing:

1. `config_use_vmix_shear_mixing`. If true, shear-based mixing is computed based upon Pacanowski and Philander (1981) and applied to velocity components and tracer quantities. \[.true./.false.\]
2. `config_vmix_shear_nu_zero`. Numerator of Pacanowski and Philander (1981) Eq (1). \[0.005 $m^2 s^{-1}$\]
3. `config_vmix_shear_alpha`. Alpha value used in Pacanowski and Philander (1981) Eqs (1) and (2). \[5\]
4. `config_vmix_shear_exp`. Exponent used in denominator of Pacanowski and Philander (1981) Eqs (1). \[2\]

The following config options should be included for  convective vertical mixing:

1. `config_use_vmix_conv_mixing`. If true, convective mixing is computed and applied to velocity components and tracer quantities. \[.true./.false.\]
2. `config_vmix_convective_diffusion`. Convective vertical diffusion applied to tracer quantities. \[1.0 $m^2 s^{-1}$\]
3. `config_vmix_convective_viscosity`. Convective vertical viscosity applied to velocity components. \[1.0 $m^2 s^{-1}$\]
4. `config_vmix_convective_triggerBVF`. Value of Brunt Viasala frequency squared below which convective mixing is triggered. \[0.0\]

The following config options should be included for background vertical mixing:

1. `config_vmix_background_viscosity`. Background vertical viscosity applied to velocity components. \[1.0e-4 $m^2 s^{-1}$\]
2. `config_vmix_background_diffusion`. Background vertical diffusion applied to all tracer quantities. \[1.0e-5 $m^2 s^{-1}$\]

### 4.2 Methods

<!--- List and describe all public methods and their interfaces (actual code for
interface that would be in header file). Describe typical use cases. -->

It is assumed that viscosity and diffusivity will be stored at cell centers and that the displaced density is computed in the density routine prior.

```c++
parallelFor(
    {NCellsAll, NVertLevels}, KOKKOS_LAMBDA(int ICell, int K) {

        vertVisc(ICell, K) = 0.0
        vertDiff(ICell, K) = 0.0

        // If shear mixing true, add shear contribution to viscosity and diffusivity 
        if (config_use_vmix_shear_mixing) {
            invAreaCell = 1.0 / areaCell(ICell)

            // Calculate the square of the shear
            for (i = 0; i < NEdgesOnCell(ICell), ++i) {
                IEdge = edgesOnCell(ICell, i)
                factor = 0.5 * dcEdge(IEdge) * dvEdge(IEdge) * invAreaCell  
                delU2 = pow(normalVelocity(IEdge,K-1) - normalVelocity(IEdge,K), 2) + pow(tangentialVelocity(IEdge,K-1) - tangentialVelocity(IEdge,K), 2)
                shearSquaredTop(ICell, K) = shearSquaredTop(ICell, K) + factor * delU2
            }
            shearSquaredTop(ICell, K) = shearSquaredTop(ICell, K) / pow(zCoor(ICell, K-1) - zCoor(ICell, K), 2);

            // Calculate Brunt Vaisala Frequency
            BruntVaisalaFreqTop(ICell, K) = (-gravity / rho_sw) * (displacedDensity(K) - density(K)) / (zCoor(ICell, K-1) - zCoor(ICell, K))

            // Calculate Richardson number
            RiTopOfCell(ICell, K) = BruntVaisalaFreqTop(ICell, K) / (shearSquaredTop(ICell, K) + 1.0e-12)

            // Add in shear contribution to vertical mixing
            vertViscTopOfCell(ICell, K) = vertViscTopOfCell(ICell, K) + nu_zero / pow(1 + alpha * RiTopOfCell(ICell, K), n) + nu_b
            vertDiffTopOfCell(ICell, K) = vertDiffTopOfCell(ICell, K) + vertViscTopOfCell(ICell, K) / (1 + alpha * RiTopOfCell(ICell, K)) + kappa_b
        }

        // If conv mixing true, add conv contribution to viscosity and diffusivity 
        if (config_use_vmix_conv_mixing) {
            if (BruntVaisalaFreqTop(ICell, K) < config_vmix_convective_triggerBVF) {
                vertViscTopOfCell(ICell, K) = vertViscTopOfCell(ICell, K) + convVisc
                vertDiffTopOfCell(ICell, K) = vertDiffTopOfCell(ICell, K) + convDiff
            }
        }

        // Add background contribution to viscosity and diffusivity 
        vertViscTopOfCell(ICell, K) = vertViscTopOfCell(ICell, K) + config_vmix_background_viscosity
        vertDiffTopOfCell(ICell, K) = vertDiffTopOfCell(ICell, K) + config_vmix_background_diffusion
    });
```

From here, the vertical viscosity and vertical diffusivity will enter into the tridiagonal solver to compute the vertical flux of momentum and tracers.

## 5 Verification and Testing

Unit tests can be initialized with linear-with-depth initial conditions (velocity and density) and use a linear equation of state. Expected values of the Richardson number, vertical viscosity, and vertical diffusion coefficients can be computed and compared to. Tests for both the shear, convective, and combined mixing contributions should be made to test requirement 2.1.

This assumes that the displaced density has already been tested. Vertical fluxes of momentum and tracers will be tested separately with the tridiagonal solver.

