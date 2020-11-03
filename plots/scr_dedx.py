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

def calculate_dEdx(energies):
    mu_def = pp.particle.MuMinusDef()
    medium = pp.medium.Ice(1.0)  # With densitiy correction
    cuts = pp.EnergyCutSettings(-1, -1)  # ecut, vcut
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
    #   Calculate DE/dx at the given energies
    # =========================================================

    dedx_all = []

    for cross in crosssections:
        dedx_all.append(np.vectorize(cross.calculate_dEdx)(energies))

    dedx_all.append(energies * np.vectorize(cross_weak.calculate_dNdx)(energies))
    dedx_all.append(energies * sigma_decay(energies, medium.mass_density, mu_def.lifetime, mu_def.mass))

    return dedx_all

def calc_dEdx_params(energies, dedx_sum):
    def func(x, a, b):
        return a + b*x
    return curve_fit(func, energies, dedx_sum, sigma=dedx_sum)[0]


def plot_ranges():
    fig = plt.figure()
    ax = fig.add_subplot(111)

    n_energies = 200
    energies = np.logspace(3, 12, n_energies)
    dedx_all = calculate_dEdx(energies)
    dedx_sum = np.zeros(n_energies)
    for dedx in dedx_all[:-2]:
        dedx_sum += dedx

    fit_params = calc_dEdx_params(energies, dedx_sum)

    def average_range_calc(energies, a, b):
        return np.log(1 + b*energies/a) / b

    dedx_ranges = average_range_calc(energies, *fit_params)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(energies/1e3, dedx_ranges/100, label='fit')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Muon Energy / GeV')
    ax.set_ylabel('Range / m')
    ax.legend()

    fig.savefig('dedx_range.pdf',bbox_inches='tight')
    ax.cla()
    plt.close()


def plot_dEdx():
    fig = plt.figure()
    ax = fig.add_subplot(111)

    n_energies = 200
    energies = np.logspace(3, 12, n_energies)
    dedx_all = calculate_dEdx(energies)

    labels = ['Ionization',
        r'$e$ Pair Production',
        'Bremsstrahlung',
        'Nuclear Interaction',
        r'$\mu$ Pair Production',
        'Weak Interaction',
        'Decay',
    ]

    for dedx, _label in zip(dedx_all, labels):
        ax.plot(energies, dedx, linestyle='-', label=_label)

    dedx_sum = np.zeros(n_energies)
    for dedx in dedx_all[:-2]:
        dedx_sum += dedx
    fit_params = calc_dEdx_params(energies, dedx_sum)
    ax.plot(energies, dedx_sum, linestyle='-', label='Sum')
    ax.plot(energies, fit_params[0] + fit_params[1]*energies,
            label='Fit: a= {:.4g}, b={:.4g}'.format(fit_params[0], fit_params[1]))

    ax.set_xlabel(r'Muon Energy $E \,/\, \mathrm{MeV} $')
    ax.set_ylabel(r'$\left\langle\frac{\mathrm{d}E}{\mathrm{d}X}\right\rangle \,\left/\, \left( \rm{MeV} / \rm{cm} \right) \right. $')
    ax.legend(loc='best')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim(1e3, 1e12)
    ax.set_ylim(1e-5, 1e7)

    fig.savefig('dEdx_all.pdf', bbox_inches='tight')
    ax.cla()
    plt.close()

if __name__ == "__main__":
    plot_dEdx()
    # plot_ranges()