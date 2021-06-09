import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from matplotlib import rc
import proposal as pp

rc('font', **{'family': 'serif',
   'serif': ['Computer Modern']})
rc('text', usetex=True)
rc('text.latex', preamble=r'\usepackage{calrsfs}')

FIG_SIZE = (5.78, 3.57)


def sigma_decay(energy, density, life_time, mass):
    r""" Calculates the decay cross section
    Converts the decay parameters (life-time, mass, energy)
    to a cross section.
    $1/(\beta \gamma \tau c_0 \rho)$
    with $\beta \gamma = \sqrt{(E^2 - m^2) / m}$

    Parameters
    ----------
    energy : float or array
        current energy of the muon
    density : foat
        density of the medium (in g/cm^3 ?)
    life_time : float
        life time of the partice
    mass : float
        mass of the particle
    """
    c0 = 2.99792458e10 #speed of light in cm / s
    betagamma = np.sqrt((energy + mass)*(energy - mass))/mass
    return 1 / (betagamma * life_time * c0 * density)

def calculate_losses(energies, ecut=np.inf, modus='dEdx'):

    p_def = pp.particle.MuMinusDef()
    medium = pp.medium.Ice()
    ecuts = pp.EnergyCutSettings(ecut, 1, False)

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

    # =========================================================
    #   Create x sections out of their parametrizations
    # =========================================================

    cross_epair = pp.crosssection.make_crosssection(
        parametrization=pp.parametrization.pairproduction.SandrockSoedingreksoRhode(**lpm_args), **args)

    cross_ioniz = pp.crosssection.make_crosssection(
        parametrization=pp.parametrization.ionization.BetheBlochRossi(ecuts), **args)

    cross_brems = pp.crosssection.make_crosssection(
        parametrization=pp.parametrization.bremsstrahlung.SandrockSoedingreksoRhode(**lpm_args), **args)

    shadow = pp.parametrization.photonuclear.ShadowButkevichMikheyev()
    cross_photo = pp.crosssection.make_crosssection(
        parametrization=pp.parametrization.photonuclear.AbramowiczLevinLevyMaor97(shadow), **args)

    cross_mupair = pp.crosssection.make_crosssection(
        parametrization=pp.parametrization.mupairproduction.KelnerKokoulinPetrukhin(), **args)

    cross_weak = pp.crosssection.make_crosssection(
        parametrization=pp.parametrization.weakinteraction.CooperSarkarMertsch(),
        particle_def=p_def,
        target=medium,
        interpolate=False,
        cuts=None,)

    # =========================================================
    #   Calculate dE/dx or dN/dx at the given energies
    # =========================================================

    xsection = np.empty((7, len(energies)))

    cross_list = [cross_ioniz, cross_epair, cross_brems, cross_photo, cross_mupair]
    for idx, cross in enumerate(cross_list):
        if modus == 'dEdx':
            xsection[idx] = cross.calculate_dEdx(energies)
        elif modus == 'dNdx':
            xsection[idx] = cross.calculate_dNdx(energies)
        else:
            raise AttributeError('modus must be dEdx or dNdx')


    if modus == 'dEdx':
        # tweak the purely stochastic processes to produce continuous losses
        xsection[-2] = energies * cross_weak.calculate_dNdx(energies)
        xsection[-1] = energies * sigma_decay(energies, medium.mass_density, p_def.lifetime, p_def.mass)
    elif modus == 'dNdx':
        xsection[-2] = cross_weak.calculate_dNdx(energies)
        xsection[-1] = sigma_decay(energies, medium.mass_density, p_def.lifetime, p_def.mass)

    return xsection

def calc_dEdx_params(energies, dedx_sum):
    def func(x, a, b):
        return a + b*x
    return curve_fit(func, energies, dedx_sum, sigma=dedx_sum)[0]


def plot_ranges(energy_min=1e3, energy_max=1e12, n_energies=200,
    output='dedx_range.pdf'):
    fig = plt.figure()
    ax = fig.add_subplot(111)

    energies = np.logspace(
        np.log10(energy_min),
        np.log10(energy_max),
        n_energies)
    dedx_all = calculate_losses(energies)
    # get the sum without the pure stochastic losses
    # weak interaction and decay
    dedx_sum = np.zeros(n_energies)
    for dedx in dedx_all[:-2]:
        dedx_sum += dedx

    fit_params = calc_dEdx_params(energies, dedx_sum)

    def average_range_calc(energies, a, b):
        return np.log(1 + b*energies/a) / b

    # def average_energy_range_calc(range, a, b):
    #     return a/b*(np.exp(b*raange)-1)
    # raange = 1e5
    # print('energy to travel 1km')
    # print(average_energy_range_calc(raange, *fit_params)/1e3)

    dedx_ranges = average_range_calc(energies, *fit_params)
    fig = plt.figure(figsize=FIG_SIZE)
    ax = fig.add_subplot(111)
    ax.plot(energies/1e3, dedx_ranges/100, label='fit')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Muon Energy / GeV')
    ax.set_xlim(energy_min/1e3, energy_max/1e3)
    ax.set_ylabel('Range / m')
    ax.legend()
    ax.grid()

    fig.savefig(output, bbox_inches='tight', pad_inches=0.02)
    plt.close()


def plot_dEdx(energy_min=1e3, energy_max=1e12, n_energies=200, ecut=np.inf,
    output='dEdx_ecut_{ecut:.4g}.pdf', do_fit=False):
    output = output.format(**{'ecut':ecut})
    fig = plt.figure()
    ax = fig.add_subplot(111)

    energies = np.logspace(
        np.log10(energy_min),
        np.log10(energy_max),
        n_energies)
    dedx_all = calculate_losses(energies, ecut, modus='dEdx')

    labels = ['Ionization',
        r'$e$ Pair Production',
        'Bremsstrahlung',
        'Nuclear Interaction',
        r'$\mu$ Pair Production',
        'Weak Interaction',
        'Decay',
    ]

    if do_fit:
        # get the sum without the pure stochastic losses
        # weak interaction and decay
        dedx_sum = np.zeros(n_energies)
        for dedx in dedx_all[:-2]:
            dedx_sum += dedx
        fit_params = calc_dEdx_params(energies, dedx_sum)
        ax.plot(energies, dedx_sum, linestyle='-', label='Sum')
        ax.plot(energies, fit_params[0] + fit_params[1]*energies,
                label='Fit: a= {:.4g}, b={:.4g}'.format(fit_params[0], fit_params[1]))
        ax.set_ylim(1e-4, 1e7)

        for dedx, _label in zip(dedx_all, labels):
            ax.plot(energies, dedx, linestyle='-', label=_label)

    else:
        for dedx, _label in zip(dedx_all[:-2], labels[:-2]):
            ax.plot(energies, dedx, linestyle='-', label=_label)
        ax.set_ylabel(r'$\frac{\mathrm{d}E}{\mathrm{d}X} \,\left/\, \left( \rm{MeV} / \rm{cm} \right) \right. $')


    ax.set_xlabel(r'Muon Energy $E \,/\, \mathrm{MeV} $')
    ax.set_ylabel(r'$\left\langle\frac{\mathrm{d}E}{\mathrm{d}X}\right\rangle \,\left/\, \left( \rm{MeV} / \rm{cm} \right) \right. $')
    ax.legend(loc='best')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim(energy_min, energy_max)

    plt.savefig(output, bbox_inches='tight', pad_inches=0.02)
    plt.close()

def plot_dNdx(energy_min=1e3, energy_max=1e12, n_energies=200, ecut=1, #MeV
    output='dNdx_ecut_{ecut:.4g}.pdf', with_sum=False):
    output = output.format(**{'ecut':ecut})
    fig = plt.figure()
    ax = fig.add_subplot(111)

    energies = np.logspace(
        np.log10(energy_min),
        np.log10(energy_max),
        n_energies)
    dndx_all = calculate_losses(energies, ecut, modus='dNdx')

    labels = ['Ionization',
        r'$e$ Pair Production',
        'Bremsstrahlung',
        'Nuclear Interaction',
        r'$\mu$ Pair Production',
        'Weak Interaction',
        'Decay',
    ]

    if with_sum:
        dndx_sum = np.zeros(n_energies)
        for dndx in dedx_all:
            dndx_sum += dndx
        ax.plot(energies, dndx_sum, linestyle='-', label='Sum')

    for dndx, _label in zip(dndx_all, labels):
        ax.plot(energies, dndx, linestyle='-', label=_label)

    ax.set_xlabel(r'Muon Energy $E \,/\, \mathrm{MeV} $')
    ax.set_ylabel(r'$\left\langle\frac{\mathrm{d}N}{\mathrm{d}X}\right\rangle \,\left/\, \left( 1 / \rm{cm} \right) \right. $')
    ax.legend(loc='lower right')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim(energy_min, energy_max)
    # ax.set_ylim(1e-4, 1e7)

    plt.savefig(output, bbox_inches='tight', pad_inches=0.02)
    plt.close()

if __name__ == "__main__":
    # plot_dEdx(output='dedx_all.pdf', ecut=np.inf, do_fit=True)
    plot_dEdx(output='dedx_ecut_{ecut:.4g}.pdf', ecut=1)
    # plot_dEdx(output='dedx_ecut_{ecut:.4g}.pdf', ecut=500)
    # plot_dNdx(output='dndx_ecut_{ecut:.4g}.pdf', ecut=1)
    # plot_dNdx(output='dndx_ecut_{ecut:.4g}.pdf', ecut=500)
    # plot_ranges(output='dedx_range.pdf')