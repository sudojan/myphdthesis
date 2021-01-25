
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import rc

rc('font', **{'family': 'serif',
   'serif': ['Computer Modern']})
rc('text', usetex=True)
rc('text.latex', preamble=r'\usepackage{calrsfs}')

FIG_SIZE = (5.78, 3.57)


def cnub(E):
    h = 4.135e-15  # eV * s
    c = 3e10  # cm / s

    k_B = 8.617e-5  # eV / K
    T = 1.9  # K
    return 2 * E**2 / (h**3 * c**2 * (np.exp(E / (k_B * T)) - 1))


solar_ks_E, solar_ks_P = np.loadtxt('resources_nu_flux/solar_ks_2.txt', unpack=True)
solar_ks_P /= 1e6

sn87a_ks_E, sn87a_ks_P = np.loadtxt('resources_nu_flux/sn87a_ks.txt', unpack=True)
sn87a_ks_P /= 1e6

terr_ks_E, terr_ks_P = np.loadtxt('resources_nu_flux/terr_ks.txt', unpack=True)
terr_ks_P /= 1e6

diffuse_sn_ks_E, diffuse_sn_ks_P = np.loadtxt('resources_nu_flux/diffuse_sn_ks.txt',
                                              unpack=True)
diffuse_sn_ks_P /= 1e6

atmo_ks_E, atmo_ks_P = np.loadtxt('resources_nu_flux/atmo_ks.txt', unpack=True)
atmo_ks_P /= 1e6

astro_ks_E, astro_ks_P = np.loadtxt('resources_nu_flux/astro_ks.txt', unpack=True)
astro_ks_P /= 1e6

cosmogenic_ks_E, cosmogenic_ks_P = np.loadtxt('resources_nu_flux/cosmogenic_ks.txt',
                                              unpack=True)
cosmogenic_ks_P /= 1e6


def plot_spectrum():
    E = np.logspace(-6, -2, 401)
    cnub_vals = cnub(E)

    fig, ax = plt.subplots(figsize=FIG_SIZE)
    ax.plot(E, cnub_vals, lw=2)
    ax.annotate(r'Cosmological $\nu$', xy=(3e-3, 1e11), color='C0')
    ax.plot(solar_ks_E, solar_ks_P, lw=2)
    ax.annotate(r'Solar $\nu$', xy=(1e3, 1e6), color='C1')
    ax.plot(sn87a_ks_E, sn87a_ks_P, lw=2)
    ax.annotate(r'SN 1987a', xy=(2e6, 1e4), color='C2')
    ax.plot(terr_ks_E, terr_ks_P, lw=2)
    ax.annotate(r'Terrestrial $\bar{\nu}$', xy=(1e0, 1e-18), color='C3')
    ax.annotate('', xy=(1e6, 1e-1), xytext=(1e2, 1e-15),
                arrowprops=dict(arrowstyle="->", color='C3'))
    ax.plot(diffuse_sn_ks_E, diffuse_sn_ks_P, lw=2)
    ax.annotate(r'Diffuse SN $\nu$', xy=(3.5e7, 1e-6), color='C4')
    ax.plot(atmo_ks_E, atmo_ks_P, lw=2)
    ax.annotate(r'Atmospheric $\nu$', xy=(5e10, 1e-16), color='C5')
    ax.plot(astro_ks_E, astro_ks_P, lw=2)
    ax.annotate(r'Astrophysical $\nu$', xy=(1e7, 1e-28), color='C6')
    ax.plot(cosmogenic_ks_E, cosmogenic_ks_P, lw=2)
    ax.annotate(r'Cosmogenic $\nu$', xy=(2e15, 1e-31), color='C7')
    ax.set_xlim(1e-6, 2e20)
    ax.set_ylim(3e-36, 1e16)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Neutrino energy / eV')
    ax.set_ylabel('Flux $\mathrm{d}\Phi / \mathrm{d}E \, / \, (\mathrm{cm}^{-2} \, \mathrm{s}^{-1} \, \mathrm{sr}^{-1} \, \mathrm{eV}^{-1} )$')
    plt.savefig('nu_spectrum.png', bbox_inches='tight', pad_inches=0.02, dpi=300)


if __name__ == '__main__':
    plot_spectrum()
