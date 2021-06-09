import proposal as pp
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc

rc('font', **{'family': 'serif',
   'serif': ['Computer Modern']})
rc('text', usetex=True)
rc('text.latex', preamble=r'\usepackage{calrsfs}')

FIG_SIZE = (5.78, 3.57)

def plot_dsigma_tm(lpm_args):

    param = pp.parametrization.bremsstrahlung.SandrockSoedingreksoRhode(**lpm_args)
    param_wo_tm = pp.parametrization.bremsstrahlung.SandrockSoedingreksoRhode()
    comp = pp.component.StandardRock()

    fig, ax = plt.subplots(figsize=FIG_SIZE)
    energies = [1e9, 1e7, 1e5, 1e3]
    labels = ['1 PeV', '10 TeV', '100 GeV', '1 GeV']
    for idx in range(4):
        energy = energies[idx]
        varr = np.logspace(-np.log10(energy),0,300)
        ax.plot(varr, 
                [param.differential_crosssection(p_def, comp, energy, vidx) for vidx in varr],
                c='C'+str(idx), label=labels[idx])
        ax.plot(varr,
                [param_wo_tm.differential_crosssection(p_def, comp, energy, vidx) for vidx in varr],
                c='C'+str(idx), ls='--')

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel(r'Relative Energy Loss $v$')
    ax.set_ylabel(r'd$\sigma$/d$v$ / MeV$^2$')
    ax.set_xlim(1/max(energies), 1)
    ax.legend()
    ax.grid()
    fig.savefig('brems_dielectric.pdf', bbox_inches='tight', pad_inches=0.02)
    plt.close()

def plot_dedx_lpm(args, lpm_args):

    cross_epair = pp.crosssection.make_crosssection(
        parametrization=pp.parametrization.pairproduction.SandrockSoedingreksoRhode(**lpm_args), **args)

    cross_epair_wo_lpm = pp.crosssection.make_crosssection(
        parametrization=pp.parametrization.pairproduction.KelnerKokoulinPetrukhin(), **args)

    cross_brems = pp.crosssection.make_crosssection(
        parametrization=pp.parametrization.bremsstrahlung.SandrockSoedingreksoRhode(**lpm_args), **args)

    cross_brems_wo_lpm = pp.crosssection.make_crosssection(
        parametrization=pp.parametrization.bremsstrahlung.SandrockSoedingreksoRhode(), **args)

    energy_min=1e3
    energy_max=1e24
    n_energies=100

    energies = np.logspace(
        np.log10(energy_min),
        np.log10(energy_max),
        n_energies)

    fig, ax = plt.subplots(figsize=FIG_SIZE)
    ax.plot(energies, cross_brems.calculate_dEdx(energies)/energies, label='Bremsstrahlung', c='C0')
    ax.plot(energies, cross_brems_wo_lpm.calculate_dEdx(energies)/energies, c='C0', ls='--')
    ax.plot(energies, cross_epair.calculate_dEdx(energies)/energies, label='Pair Production', c='C1')
    ax.plot(energies, cross_epair_wo_lpm.calculate_dEdx(energies)/energies, c='C1', ls='--')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim(energy_min, energy_max)
    ax.set_ylim(bottom=1e-8)
    ax.set_xlabel('Muon Energy $E_\mu$ / MeV')
    ax.set_ylabel(r'$\frac{1}{E} \left<\frac{\mathrm{d}E}{\mathrm{d}X}\right> / ( \rm{cm}^2 \rm{g}^{-1} )$')
    ax.legend()
    fig.savefig('dedx_lpm.pdf', bbox_inches='tight', pad_inches=0.02)
    plt.close()

if __name__ == '__main__':

    p_def = pp.particle.MuMinusDef()
    medium = pp.medium.Ice()
    ecuts = pp.EnergyCutSettings(np.inf, 1, False)

    args = {
        "particle_def": p_def,
        "target": medium,
        "interpolate": False,
        "cuts": ecuts,
    }
    lpm_args = {
        "particle_def": p_def,
        "lpm": True,
        "medium": medium,
    }

    plot_dedx_lpm(args, lpm_args)
    plot_dsigma_tm(lpm_args)
