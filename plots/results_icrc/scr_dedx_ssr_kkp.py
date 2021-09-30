import proposal as pp
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc

rc('font', **{'family': 'serif',
   'serif': ['Computer Modern']})
rc('text', usetex=True)
rc('text.latex', preamble=r'\usepackage{calrsfs}')

FIG_SIZE = (5.78, 3.57)

def plot_helper(energies, dedx_arr, labels, plot_file):

    fig = plt.figure(figsize=FIG_SIZE)
    ax1 = fig.add_subplot(211)
    ax1.plot(energies, dedx_arr[0]/energies/1e-6, label=labels[0], ls='--')
    ax1.plot(energies, dedx_arr[1]/energies/1e-6, label=labels[1], ls=':')

    ax1.set_xscale('log')
    # ax1.set_yscale('log')
    ax1.set_ylabel(r'$\frac{1}{E} \left<\frac{\mathrm{d}E}{\mathrm{d}X}\right> / ( \rm{cm}^2 \rm{g}^{-1} \times 10^{-6})$')
    ax1.legend(loc='lower right')
    ax1.grid()
    ax1.set_xlim(min(energies), max(energies))

    ax2 = fig.add_subplot(212, sharex=ax1)
    ax2.plot(energies, dedx_arr[1]/dedx_arr[0], label=r"{}/{}".format(labels[1], labels[0]))
    ax2.set_xlabel(r'Muon Energy / MeV')
    ax2.set_ylabel(r'Ratio')
    # ax2.set_yscale('log')
    ax2.legend(loc='lower right')
    ax2.grid()

    plt.subplots_adjust(hspace=.1)
    plt.setp(ax1.get_xticklabels(), visible=False)
    fig.savefig(plot_file, bbox_inches='tight', pad_inches=0.02)
    plt.close()

def plot_dedx_lpm(energies, args):

    epair_ssr = pp.crosssection.make_crosssection(
        parametrization=pp.parametrization.pairproduction.SandrockSoedingreksoRhode(), **args)

    epair_kkp = pp.crosssection.make_crosssection(
        parametrization=pp.parametrization.pairproduction.KelnerKokoulinPetrukhin(), **args)

    brems_ssr = pp.crosssection.make_crosssection(
        parametrization=pp.parametrization.bremsstrahlung.SandrockSoedingreksoRhode(), **args)

    brems_kkp = pp.crosssection.make_crosssection(
        parametrization=pp.parametrization.bremsstrahlung.KelnerKokoulinPetrukhin(), **args)

    labels = ['KKP95', 'SSR19']

    plot_helper(energies,
        [brems_kkp.calculate_dEdx(energies), brems_ssr.calculate_dEdx(energies)],
        labels, 'dedx_compare_brems.pdf')
    plot_helper(energies,
        [epair_kkp.calculate_dEdx(energies), epair_ssr.calculate_dEdx(energies)],
        labels, 'dedx_compare_epair.pdf')

if __name__ == '__main__':

    args = {
        "particle_def": pp.particle.MuMinusDef(),
        "target": pp.medium.Ice(),
        "interpolate": False,
        "cuts": pp.EnergyCutSettings(np.inf, 1, False),
    }

    energies = np.geomspace(1e3, 1e12, 100)

    plot_dedx_lpm(energies, args)
