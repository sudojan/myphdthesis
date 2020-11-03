import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

import proposal as pp


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

def calculate_losses(energies, ecut, modus='dEdx'):
    mu_def = pp.particle.MuMinusDef()
    medium = pp.medium.Ice(1.0)  # With densitiy correction
    cuts = pp.EnergyCutSettings(ecut, -1)  # ecut, vcut
    multiplier_all = 1.0
    lpm_effect = True

    # =========================================================
    #   Create x sections out of their parametrizations
    # =========================================================

    crosssections = []

    crosssections.append(pp.crosssection.IonizIntegral(
        pp.parametrization.ionization.BetheBlochRossi(
            mu_def, medium, cuts, multiplier_all)
    ))  

    crosssections.append(pp.crosssection.EpairIntegral(
        pp.parametrization.pairproduction.KelnerKokoulinPetrukhin(
            mu_def, medium, cuts, multiplier_all, lpm_effect)
        ))

    crosssections.append(pp.crosssection.BremsIntegral(
        pp.parametrization.bremsstrahlung.KelnerKokoulinPetrukhin(
            mu_def, medium, cuts, multiplier_all, lpm_effect)
    ))

    crosssections.append(pp.crosssection.PhotoIntegral(
        pp.parametrization.photonuclear.AbramowiczLevinLevyMaor97(
            mu_def, medium, cuts, multiplier_all,
            pp.parametrization.photonuclear.ShadowButkevichMikhailov())
    ))    

    crosssections.append(pp.crosssection.MupairIntegral(
        pp.parametrization.mupairproduction.KelnerKokoulinPetrukhin(
            mu_def, medium, cuts, multiplier_all, False)
    ))

    cross_weak = pp.crosssection.WeakIntegral(
        pp.parametrization.weakinteraction.CooperSarkarMertsch(
            mu_def, medium, multiplier_all))

    # =========================================================
    #   Calculate dE/dx or dN/dx at the given energies
    # =========================================================

    xsection = np.empty((7, len(energies)))

    for idx, cross in enumerate(crosssections):
        if modus == 'dEdx':
            xsection[idx] = np.vectorize(cross.calculate_dEdx)(energies)
        elif modus == 'dNdx':
            xsection[idx] = np.vectorize(cross.calculate_dNdx)(energies)
        else:
            raise AttributeError('modus must be dEdx or dNdx')


    if modus == 'dEdx':
        # tweak the purely stochastic processes to produce continuous losses
        xsection[-2] = energies * np.vectorize(cross_weak.calculate_dNdx)(energies)
        xsection[-1] = energies * sigma_decay(energies, medium.mass_density, mu_def.lifetime, mu_def.mass)
    elif modus == 'dNdx':
        xsection[-2] = np.vectorize(cross_weak.calculate_dNdx)(energies)
        xsection[-1] = sigma_decay(energies, medium.mass_density, mu_def.lifetime, mu_def.mass)

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
    dedx_all = calculate_losses(energies, ecut=-1)
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
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(energies/1e3, dedx_ranges/100, label='fit')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Muon Energy / GeV')
    ax.set_ylabel('Range / m')
    ax.legend()
    ax.grid()

    fig.savefig(output, bbox_inches='tight', pad_inches=0.02)
    ax.cla()
    plt.close()


def plot_dEdx(energy_min=1e3, energy_max=1e12, n_energies=200, ecut=-1, #MeV
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

    if output.endswith('.pdf'):
        plt.savefig(output, bbox_inches='tight', pad_inches=0.02)
    else:
        plt.savefig(output, bbox_inches='tight', pad_inches=0.02, dpi=300)
    ax.cla()
    plt.close()

def plot_dNdx(energy_min=1e3, energy_max=1e12, n_energies=200, ecut=1, #MeV
    output='dNdx_ecut_{ecut:.4g}.pdf'):
    output = output.format(**{'ecut':ecut})
    fig = plt.figure()
    ax = fig.add_subplot(111)

    energies = np.logspace(
        np.log10(energy_min),
        np.log10(energy_max),
        n_energies)
    dedx_all = calculate_losses(energies, ecut, modus='dNdx')

    labels = ['Ionization',
        r'$e$ Pair Production',
        'Bremsstrahlung',
        'Nuclear Interaction',
        r'$\mu$ Pair Production',
        'Weak Interaction',
        'Decay',
    ]

    dedx_sum = np.zeros(n_energies)
    for dedx in dedx_all:
        dedx_sum += dedx
    ax.plot(energies, dedx_sum, linestyle='-', label='Sum')

    for dedx, _label in zip(dedx_all, labels):
        ax.plot(energies, dedx, linestyle='-', label=_label)

    ax.set_xlabel(r'Muon Energy $E \,/\, \mathrm{MeV} $')
    ax.set_ylabel(r'$\left\langle\frac{\mathrm{d}N}{\mathrm{d}X}\right\rangle \,\left/\, \left( 1 / \rm{cm} \right) \right. $')
    ax.legend(loc='lower right')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim(energy_min, energy_max)
    # ax.set_ylim(1e-4, 1e7)

    if output.endswith('.pdf'):
        plt.savefig(output, bbox_inches='tight', pad_inches=0.02)
    else:
        plt.savefig(output, bbox_inches='tight', pad_inches=0.02, dpi=300)
    ax.cla()
    plt.close()

if __name__ == "__main__":
    plot_dEdx(output='dedx_all.pdf', ecut=-1, do_fit=True)
    plot_dEdx(output='dedx_ecut_{ecut:.4g}.pdf', ecut=500)
    plot_dNdx(output='dndx_ecut_{ecut:.4g}.pdf', ecut=1)
    plot_dNdx(output='dndx_ecut_{ecut:.4g}.pdf', ecut=500)
    plot_ranges(output='dedx_range.pdf')