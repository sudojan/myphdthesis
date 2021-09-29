
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy import constants
from scipy.integrate import quad, dblquad
from scipy.special import spence
from tqdm import tqdm
from matplotlib import rc

rc('font', **{'family': 'serif',
   'serif': ['Computer Modern']})
rc('text', usetex=True)
rc('text.latex', preamble=r'\usepackage{calrsfs}')

FIG_SIZE = (5.78, 3.57)

cm = 1.0
meter = 1.e2 * cm
MeV = 1.0
m_e = constants.value(u'electron mass energy equivalent in MeV') * MeV
m_mu = constants.value(u'muon mass energy equivalent in MeV') * MeV
m_tau = constants.value(u'tau mass energy equivalent in MeV') * MeV
alpha = constants.value(u'fine-structure constant')
avogadro = constants.value(u'Avogadro constant')
pi = constants.pi
r_e = constants.value(u'classical electron radius') * meter
r_mu = r_e * m_e / m_mu
convert_MeV2_to_cm2 = 0.389379338e-21 * cm * cm
sqrte = np.exp(0.5)

def dilog_scipy(x):
    r'''
    wrap the scipy function for the dilogarithm
    to the definition, like everyone is doing it
    assumes -1 <= x <= 1
    returns nan if outside
    '''
    return spence(1 - x)

def well_known_function(ny):
    summa = 0.
    old_summa = -1.
    n = 1
    while (np.abs(summa - old_summa) > 1e-6): # .and. n < 13)
        old_summa = summa
        summa = summa + 1./(n*(n**2 + ny**2))
        n = n + 1
        # print( n, summa)
    return (ny**2 * summa)

@np.vectorize
def rad_corr_fit(v):
    def __fit_func(a0, a1, a2, a3, a4, a5):
        return a0 + a1 * v + a2 * v * v + a3 * v * np.log(v) + \
                a4 * np.log(1 - v) + a5 * np.log(1 - v)**2

    if (v < 0.) | (v > 1.):
        return 0.0
    elif (v < 0.02):
        return -0.00349 + 148.84 * v - 987.531 * v * v
    elif (v < 0.1):
        return 0.1642 + 132.573 * v - 585.361 * v * v + 1407.77 * v * v * v
    elif(v < 0.9):
        return __fit_func(-2.892, -19.02, 57.70, -63.42, 14.12, 1.842)
    else:
        return __fit_func(2134.19, 581.823, -2708.85, 4767.05, 1.529, 0.36193)


# def plot_two_line_comparison(arr_list,
#         legend_list,
#         output=None,
#         ax_label_list=None,
#         logscale=False,
#         title=None):

#     if len(arr_list) < 3:
#         raise ValueError("array list should at least have len 3")
#     if len(arr_list) != len(legend_list) + 1:
#         raise ValueError("legend_list should have len(arr_list)-1")

#     fig = plt.figure()
#     if title is not None:
#         fig.suptitle(title)
#     gs = GridSpec(3, 1)
#     ax1 = fig.add_subplot(gs[:-1])
#     ax2 = fig.add_subplot(gs[-1], sharex=ax1)

#     for idx in range(len(arr_list)-2):
#         ax1.plot(arr_list[0], arr_list[idx+2], label=legend_list[idx+1])
#         ax2.plot(arr_list[0], arr_list[idx+2]/arr_list[1],label=legend_list[idx+1])
#     ax1.plot(arr_list[0], arr_list[1], label=legend_list[0])

#     if logscale:
#         ax1.set_xscale('log')
#         ax1.set_yscale('log')

#     if ax_label_list is not None:
#         ax2.set_xlabel(ax_label_list[0])
#         ax1.set_ylabel(ax_label_list[1])
#         # ax2.set_ylabel(ax_label_list[2])
#         ax2.set_ylabel(
#             r'$\frac{{\mathrm{{{}}}}}{{\mathrm{{{}}}}}$'.format(
#                 legend_list[1], legend_list[0]))

#     ax1.legend()
#     ax2.grid()
#     plt.subplots_adjust(hspace=.1)
#     plt.setp(ax1.get_xticklabels(), visible=False)

#     if output.endswith('.pdf'):
#         plt.savefig(output, bbox_inches='tight', pad_inches=0.02)
#     else:
#         plt.savefig(output, bbox_inches='tight', pad_inches=0.02, dpi=300)
#     ax1.cla()
#     ax2.cla()
#     plt.close()


class Medium(object):
    def __init__(self, atomic_number_Z, atomic_mass_A):
        self.Z = atomic_number_Z
        self.A = atomic_mass_A
        self.B_el = self.radiation_log_elastic(atomic_number_Z)
        self.B_inel = self.radiation_log_inelastic(atomic_number_Z)


    def radiation_log_elastic(self, atomic_number_Z):
        rad_log = {
            1 : 202.4,
            2 : 151.9,
            3 : 159.9,
            4 : 172.3,
            5 : 177.9,
            6 : 178.3,
            7 : 176.6,
            8 : 173.4,
            9 : 170.0,
            10: 165.8,
            11: 165.8,
            12: 167.1,
            13: 169.1,
            14: 170.8,
            15: 172.2,
            16: 173.4,
            17: 174.8,
            18: 174.8,
            19: 175.1,
            20: 175.6,
            21: 176.2,
            22: 176.8,
            26: 175.8,
            29: 173.1,
            32: 173.0,
            35: 173.5,
            42: 175.9,
            50: 177.4,
            53: 178.6,
            74: 177.6,
            82: 178.0,
            92: 179.8
        }.get(atomic_number_Z, 182.7) # often set to 183

        return rad_log

    def radiation_log_inelastic(self, atomic_number_Z):
        rad_log_inel = {
            1 : 446.0
        }.get(atomic_number_Z, 1429.0) # often set to 1194

        return rad_log_inel

class MupairKelner(object):
    def __init__(self,
                medium,
                m_in=m_mu,
                epsabs = 1.0e-60,
                epsrel = 1.0e-3):
        self.m_in = m_in
        self.Z = medium.Z
        self.A = medium.A
        self.B_el = medium.B_el
        self.B_inel = medium.B_inel
        self.Z13 = 1./np.cbrt(self.Z)
        self.const_prefactor = 2. / (3 * pi) * (self.Z * alpha * r_mu)**2 * avogadro / self.A
        self.rho_min = 0.0
        self.U_nom = 0.65 * self.A**(-0.27) * self.B_el * self.Z13 * self.m_in / m_e
        self.epsabs = epsabs
        self.epsrel = epsrel

        self.rho_max_log = 0.0

    def rho_max(self, E_in, v):
        return 1 - 2 * self.m_in / (v * E_in)

    def v_min(self, E_in):
        return 2 * self.m_in / E_in

    def v_max(self, E_in):
        return 1 - self.m_in / E_in

    def dsigma_dv_drho(self, E_in, v, rho):
        """
        Kelner 2000
        """
        rho2 = rho * rho
        beta = v*v/(2*(1. - v))
        xi = (0.5*v)**2*(1. - rho2)/(1. - v)

        C = ((2. + rho2)*(1. + beta) + xi*(3. + rho2))* np.log(1. + 1./xi) \
            + ((1. + rho2)*(1. + 1.5*beta) - 1./xi*(1. + 2*beta)*(1. - rho2)) * np.log(1. + xi) \
            - 1. - 3*rho2 + beta*(1 - 2*rho2)
        Y = 12 * np.sqrt(self.m_in / E_in)
        def U_func(rho2_):
            nomin = 2*sqrte*self.m_in**2 * self.B_el * self.Z13 * (1 + xi) * (1 + Y)
            denom = m_e * E_in * v * (1 - rho2_)
            return self.U_nom / (1 + nomin / denom )
        X = 1 + U_func(rho2) - U_func(self.rho_max(E_in, v)**2)
        return self.const_prefactor*(1 - v)/v * C * np.log(X)

    def dsigma_dv(self, E_in, v):
        # integrating over 1 - rho
        # to integrate with log substitution
        def func_integrand(rho):
            rho_exp = np.exp(rho)
            return 2 * rho_exp * self.dsigma_dv_drho(E_in, v, 1. - rho_exp)

        return quad(func_integrand,
            np.log(1. - self.rho_max(E_in, v)),
            self.rho_max_log,
            epsabs=self.epsabs, epsrel=self.epsrel)[0]


class BremsstrahlungParam(object):
    def __init__(self,
                medium,
                m_in=m_mu,):
        self.m_in = m_in
        self.Z = medium.Z
        self.A = medium.A
        self.B_el = medium.B_el
        self.B_inel = medium.B_inel
        self.Z13 = 1./np.cbrt(self.Z)
        self.d_n = 1.54 * self.A**0.27
        self.const_prefactor = alpha * (2 * self.Z * r_e * m_e / self.m_in)**2 * avogadro / self.A
        self.coulomb_factor = well_known_function(self.Z * alpha)
        
        q_c = m_mu * sqrte*sqrte/ self.d_n
        rho = np.sqrt(1 + (2*self.m_in/q_c)**2)
        tmp1 = np.log(self.m_in / q_c)
        tmp2 = 0.5*rho * np.log((rho + 1)/(rho - 1))
        self.Delta1 = tmp1 + tmp2
        self.Delta2 = tmp1 + 0.5*(3 - rho**2) * tmp2 + 2 * (self.m_in/q_c)**2

    def v_min(self, E_in=1e3):
        return 0

    def v_max(self, E_in):
        return 1 - 0.75 * sqrte * self.m_in / (E_in * self.Z13)

    def atom_inel_mu(self, E_in, v):
        r""" interaction on atomic electrons
        bremsstrahlung emitted at the muon line
        calculation of Kelner Kokoulin Petrukhin
        """
        delta = self.m_in * self.m_in * v / (2 * E_in * (1 - v))
        Phi_mu = np.log(self.m_in / delta / (self.m_in * delta / m_e**2 + sqrte)) - \
            np.log(1 + m_e / (delta * self.B_inel * self.Z13**2 * sqrte))
        tmp = 4./3. * (1 - v) + v*v
        return self.const_prefactor * tmp / v * Phi_mu / self.Z

    def coulomb_corr(self, E_in, v):
        r""" COulomb Correction
        Andreev 97 eq. 3.25
        """
        tmp = 1 - 2./3*v + v*v
        return self.const_prefactor * tmp / v * self.coulomb_factor

    def nlo_corr(self, E_in, v):
        r""" NLO Corrections
        Sandrock 2018
        """
        delta = self.m_in * self.m_in * v / (2 * E_in * (1 - v))
        phi1_0 = np.log( self.B_el * self.Z13 * self.m_in / m_e / \
            (1 + self.B_el * self.Z13 * sqrte * delta / m_e) )

        phi1 = phi1_0 - self.Delta1 * (1 - 1./self.Z)

        return self.const_prefactor * 0.25 * alpha * phi1 * rad_corr_fit(v) / v

    def vacuum_polarization(self, E_in, v):
        r""" Vacuum Polarization
        Sandrock 2018
        """
        delta = self.m_in * self.m_in * v / (2 * E_in * (1 - v))
        phi1_0 = np.log( self.B_el * self.Z13 * self.m_in / m_e / \
            (1 + self.B_el * self.Z13 * sqrte * delta / m_e) )

        phi1 = phi1_0 - self.Delta1 * (1 - 1./self.Z)

        def func(f1, f2):
            return f1 + f2 * self.Z13 * 1e-3
        a = func(2.603, -64.68)
        b = func(0.2672, 9.791)
        c = func(2.055, -86.08)
        tmp = b / np.pi * np.log(a**(1./b) + np.exp(c/b)*delta)

        return self.const_prefactor * alpha * phi1 * tmp / v

class BremsKelner(BremsstrahlungParam):

    def atom_el(self, delta):
        r""" Screening of the nucleus due to atomic electrons
        using Thomas Fermi model
        """
        fs = np.log(self.m_in / delta) - 0.5
        Delta_a = np.log(1 + 1/(delta * sqrte * self.B_el * self.Z13 / m_e))
        return fs - Delta_a

    def nucl_el(self, delta):
        r""" Correction of finite size of nucleus
        nuclear form factor
        """
        return np.log(self.d_n / (1 + delta * (self.d_n * sqrte - 2) / self.m_in))

    def dsigma_dv(self, energy, v):
        delta = self.m_in * self.m_in * v / (2 * energy * (1 - v))
        v_dependency = (4./3. * (1 - v) + v*v) / v
        contributions = self.atom_el(delta) - self.nucl_el(delta) * (1 - 1/self.Z)# + self.atom_inel(delta)

        return self.const_prefactor * v_dependency * contributions

class BremsSandrock(BremsstrahlungParam):

    def dsigma_dv_full_screen(self, E_in, v):
        phi_1 = np.log(self.m_in / m_e * self.B_el * self.Z13)
        phi_2 = phi_1 - 1./6
        tmp = (2 - 2*v + v*v) * phi_1 - 2./3 * (1 - v) * phi_2
        return self.const_prefactor * tmp / v

    def dsigma_dv_no_screen(self, E_in, v):
        delta = self.m_in * self.m_in * v / (2 * E_in * (1 - v))
        phi_1 = np.log(self.m_in/delta) - 0.5
        phi_2 = phi_1
        tmp = (2 - 2*v + v*v) * phi_1 - 2./3 * (1 - v) * phi_2
        return self.const_prefactor * tmp / v

    def dsigma_dv_screen_interpol(self, E_in, v):
        delta = self.m_in * self.m_in * v / (2 * E_in * (1 - v))
        phi_1 = np.log((self.m_in / m_e * self.B_el * self.Z13) / \
            (1 + delta / m_e * sqrte * self.B_el * self.Z13))
        phi_2 = np.log((self.m_in / m_e * self.B_el * self.Z13 * np.exp(-1./6)) / \
            (1 + delta / m_e * np.exp(1./3) * self.B_el * self.Z13))
        tmp = (2 - 2*v + v*v) * phi_1 - 2./3 * (1 - v) * phi_2
        return self.const_prefactor * tmp / v

    def dsigma_dv_nucl_el(self, E_in, v):
        tmp = (2 - 2*v + v*v) * self.Delta1 - 2./3 * (1 - v) * self.Delta2
        return self.const_prefactor * tmp / v

    def dsigma_dv_nucl_inel(self, E_in, v):
        return self.dsigma_dv_nucl_el(E_in, v) / self.Z

    def dsigma_dv_sum(self, E_in, v):
        return self.dsigma_dv_screen_interpol(E_in, v) - \
            self.dsigma_dv_nucl_el(E_in, v) + \
            self.dsigma_dv_nucl_inel(E_in, v) + \
            self.atom_inel_mu(E_in, v) + \
            self.nlo_corr(E_in, v)

def plot_brems_dsigma_contributions(energy, medium, nsteps,
    output='brems_contributions.pdf'):
    brems = BremsSandrock(medium)
    brems_kelner = BremsKelner(medium)
    v_arr = np.geomspace(1e-5, brems.v_max(energy),
            nsteps+1, endpoint=False)[1:]

    screen_arr = brems.dsigma_dv_screen_interpol(energy, v_arr)
    fs_arr = brems.dsigma_dv_full_screen(energy, v_arr)
    ns_arr = brems.dsigma_dv_no_screen(energy, v_arr)
    nucl_arr = brems.dsigma_dv_nucl_el(energy, v_arr)
    atom_inel_arr = brems.atom_inel_mu(energy, v_arr)
    nucl_inel_arr = brems.dsigma_dv_nucl_inel(energy, v_arr)
    nlo_arr = brems.nlo_corr(energy, v_arr)
    vac_pol_arr = brems.vacuum_polarization(energy, v_arr)
    coulomb_arr = brems.coulomb_corr(energy, v_arr)
    screen_approx_arr = brems_kelner.dsigma_dv(energy, v_arr)

    fig = plt.figure(figsize=(7,7))

    gs = GridSpec(4, 1)
    ax1 = fig.add_subplot(gs[:-1])
    ax2 = fig.add_subplot(gs[-1], sharex=ax1)

    ax1.plot(v_arr, screen_arr, label=r'Screening $\Phi_0$')
    ax1.plot(v_arr, fs_arr, label=r'Full Screening', ls='--')
    ax1.plot(v_arr, ns_arr, label=r'No Screening', ls='--')
    ax1.plot(v_arr, nucl_arr, label=r'- Nuclear el.')
    ax1.plot(v_arr, atom_inel_arr, label=r'Atom. inel. $\mu$')
    ax1.plot(v_arr, nucl_inel_arr, label=r'Nuclear inel.')
    ax1.plot(v_arr, nlo_arr, label=r'Radiative Corr.')
    ax1.plot(v_arr, vac_pol_arr, label=r'Vacuum Pol.')
    ax1.plot(v_arr, coulomb_arr, label=r'- Coulomb Corr.')
    # ax1.plot(v_arr, screen_approx_arr, label=r'Screen approx')

    ax1.set_xscale('log')
    ax1.set_yscale('log')

    ax1.set_xlim(v_arr[0], 1)

    ax1.set_ylabel(r'Differential Cross Section $\frac{\mathrm{d}\sigma}{\mathrm{d}v} \,/\, \mathrm{cm}^2$')

    ax1.legend(ncol=2)
    ax1.grid()

    ax2.plot(v_arr, screen_approx_arr / (screen_arr - nucl_arr + nucl_inel_arr),
            label=r'$\frac{\Phi_1 \approx \Phi_2}{\Phi_1 \neq \Phi_2}$')

    ax2.set_xlabel(r'Relative Energy Loss $v$')
    ax2.set_ylabel(r'Ratio of $\frac{\mathrm{d}\sigma}{\mathrm{d}v}$')
    ax2.legend()

    plt.subplots_adjust(hspace=.1)
    plt.setp(ax1.get_xticklabels(), visible=False)

    plt.savefig(output, bbox_inches='tight', pad_inches=0.02)
    plt.close()

def plot_zeta_effect(output='epair_zeta_effect.pdf'):

    epair_list = [
        EPairSandrock(Medium(1, 1)),
        EPairSandrock(Medium(11, 22)),
        EPairSandrock(Medium(92, 238))
    ]

    energies = np.logspace(3, 9, 100)

    fig = plt.figure(figsize=FIG_SIZE)
    ax = fig.add_subplot()

    for idx, epair in enumerate(epair_list):
        ax.axhline((epair.Z + 1)/epair.Z - 1, c='C{}'.format(idx), ls='--')
        tmp_arr = np.array([epair.zeta(eidx) for eidx in energies])
        ax.plot(energies, (epair.Z + tmp_arr)/epair.Z - 1, c='C{}'.format(idx), label='Z={}'.format(epair.Z))
    # ax.plot(energies, tmp_arr)

    ax.set_xlabel(r'Muon Energy $E_\mu$ / MeV')
    ax.set_ylabel(r'$\frac{Z + \zeta}{Z} - 1$')

    ax.set_xscale('log')
    ax.set_yscale('log')

    ax.legend()

    plt.savefig(output, bbox_inches='tight', pad_inches=0.02)
    plt.close()



class PairProductionParam(object):
    def __init__(self,
                medium,
                m_in=m_mu,
                m_pair=m_e,):
        self.m_in = m_in
        self.m_pair = m_pair
        self.Z = medium.Z
        self.A = medium.A
        self.B_el = medium.B_el
        self.B_inel = medium.B_inel
        self.Z13 = 1./np.cbrt(self.Z)
        self.d_n = 1.54 * self.A**0.27
        self.const_prefactor = 2. / (3 * pi) * (self.Z * alpha * r_e)**2 * avogadro / self.A
        self.rho_min = 0.0
        self.coulomb_factor = well_known_function(self.Z * alpha)

    def Phi_e(self):
        pass

    def Phi_mu(self):
        pass

    def zeta(self, E_in):
        if (self.Z == 1):
            g1 = 4.4e-5
            g2 = 4.8e-5
        else:
            g1 = 1.95e-5
            g2 = 5.3e-5

        tmp = E_in/self.m_in
        nomin = (0.073*np.log(tmp / (1 + g1 / self.Z13**2 * tmp)) - 0.26)
        denom = (0.058*np.log(tmp / (1 + g2 / self.Z13 * tmp)) - 0.14)
        if (nomin > 0.0) & (denom > 0.0):
            return nomin / denom
        else:
            return 0.0

    def rho_max(self, E_in, v):
        return np.sqrt(1 - 4 * self.m_pair / (E_in * v)) * \
            (1 - 6 * self.m_in * self.m_in / (E_in * E_in * (1 - v)))

    def v_min(self, E_in):
        return 4 * self.m_pair / E_in

    def v_max(self, E_in):
        return 1 - 0.75 * sqrte * self.m_in / (E_in * self.Z13)

    def dsigma_dv_drho_e(self, E_in, v, rho):
        return self.const_prefactor*(1 - v)/v * self.Phi_e(E_in, v, rho)

    def dsigma_dv_drho_mu(self, E_in, v, rho):
        return self.const_prefactor*(1 - v)/v * \
            (self.m_pair/self.m_in)**2 * self.Phi_mu(E_in, v, rho)

    def dsigma_dv_drho(self, E_in, v, rho):
        return self.const_prefactor*(1 - v)/v * \
            (self.Phi_e(E_in, v, rho) + (self.m_pair/self.m_in)**2 * self.Phi_mu(E_in, v, rho))

    def dsigma_dv_drho_eatom(self, E_in, v, rho):
        return self.dsigma_dv_drho(E_in, v, rho) * self.zeta(E_in) / self.Z

    def dsigma_dv_drho_coulomb(self, E_in, v, rho):
        r""" Coulomb Corrections
        Calculated by Ivanov et al
        """
        rho2 = rho * rho
        beta = v*v/(2*(1. - v))
        xi = ((self.m_in*v)/(2*self.m_pair))**2*(1. - rho2)/(1. - v)

        C_e = ((2. + rho2)*(1. + beta) + xi*(3. + rho2))* np.log(1. + 1./xi) \
            + (1. - rho2 - beta)/(1. + xi) - (3. + rho2)
        return self.const_prefactor * (1 - v)/v * C_e * self.coulomb_factor

class EPairSandrock(PairProductionParam):

    def Phi_e(self, E_in, v, rho):
        rho2 = rho * rho
        beta = v*v/(2*(1 - v))
        xi = (self.m_in*v/(2*m_e))**2*(1 - rho2)/(1 - v)
        Ce = ((2 + rho2)*(1 + beta) + xi*(3 + rho2))*np.log(1 + 1/xi) + \
            (1 - rho2 - beta)/(1 + xi) - (3 + rho2)

        Ce2 = ((1 - rho2)*(1 + beta) + xi*(3 - rho2))*np.log(1 + 1/xi) + \
            2*(1 - beta - rho2)/(1 + xi) - (3 - rho2)
        Ce1 = Ce - Ce2
        De = ((2 + rho2)*(1 + beta) + xi*(3 + rho2))*dilog_scipy(1/(1 + xi)) - \
            (2 + rho2)*xi*np.log(1 + 1/xi) - (xi + rho2 + beta)/(1 + xi)
        Xe = np.exp(-De/Ce)
        Le1 = np.log(self.B_el*self.Z13*np.sqrt(1 + xi) / \
            (Xe + 2*m_e*sqrte*self.B_el*self.Z13*(1 + xi)/(E_in*v*(1 - rho2)))) - \
            De/Ce - \
            0.5*np.log(Xe + (m_e/self.m_in*self.d_n)**2*(1 + xi))
        Le2 = np.log(self.B_el*self.Z13*np.exp(-1/6.0)*np.sqrt(1 + xi) / \
            (Xe + 2*m_e*np.exp(1/3.0)*self.B_el*self.Z13*(1 + xi)/(E_in*v*(1 - rho2)))) - \
            De/Ce - \
            0.5*np.log(Xe + (m_e/self.m_in*self.d_n)**2*np.exp(-1/3.0)*(1 + xi))

        return Ce1*Le1 + Ce2*Le2

    def Phi_mu(self, E_in, v, rho):
        rho2 = rho * rho
        beta = v*v/(2*(1 - v))
        xi = (self.m_in*v/(2*m_e))**2*(1 - rho2)/(1 - v)
        Cm = ((1 + rho2)*(1 + 1.5*beta) - 1/xi*(1 + 2*beta)*(1 - rho2)) \
          *np.log(1 + xi) + xi*(1 - rho2 - beta)/(1 + xi) + (1 + 2*beta)*(1 - rho2)

        Cm2 = ((1 - beta)*(1 - rho2) - xi*(1 + rho2))*np.log(1 + xi)/xi \
            - 2*(1 - beta - rho2)/(1 + xi) + 1 - beta - (1 + beta)*rho2
        Cm1 = Cm - Cm2
        Dm = ((1 + rho2)*(1 + 1.5*beta) - 1/xi*(1 + 2*beta)*(1 - rho2)) \
            *dilog_scipy(xi/(1 + xi)) + (1 + 1.5*beta)*(1 - rho2)/xi*np.log(1 + xi) \
            + (1 - rho2 - 0.5*beta*(1 + rho2) + (1 - rho2)/(2*xi)*beta) \
            *xi/(1 + xi)
        if (Dm/Cm > 0.0):
            Xm = np.exp(-Dm/Cm)
            Lm1 = np.log(Xm*self.m_in/m_e*self.B_el*self.Z13/self.d_n \
                /(Xm + 2*m_e*sqrte*self.B_el*self.Z13*(1 + xi)/(E_in*v*(1 - rho2))))
            Lm2 = np.log(Xm*self.m_in/m_e*self.B_el*self.Z13/self.d_n \
                /(Xm + 2*m_e*np.exp(1/3.0)*self.B_el*self.Z13*(1 + xi) \
                /(E_in*v*(1 - rho2))))
        else:
            Xm_inv = np.exp(Dm/Cm)
            Lm1 = np.log(self.m_in/m_e*self.B_el*self.Z13/self.d_n \
                /(1 + 2*m_e*sqrte*self.B_el*self.Z13*(1 + xi)/(E_in*v*(1 - rho2))*Xm_inv))
            Lm2 = np.log(self.m_in/m_e*self.B_el*self.Z13/self.d_n \
                /(1 + 2*m_e*np.exp(1/3.0)*self.B_el*self.Z13*(1 + xi)/(E_in*v*(1 - rho2)) \
                *Xm_inv))

        return Cm1*Lm1 + Cm2*Lm2

class EPairKelner(PairProductionParam):

    def Phi_e(self, E_in, v, rho):
        r"""
        calculates the cross section of e-diagrams
        derived by Kelner, Kokoulin, Petrukhin
        """
        rho2 = rho * rho
        beta = v*v/(2*(1. - v))
        xi = ((self.m_in*v)/(2*self.m_pair))**2*(1. - rho2)/(1. - v)

        Y_e = (5. - rho2 + 4*beta*(1. + rho2)) \
            / (2*(1. + 3*beta) * np.log(3 + 1./xi) - rho2 - 2*beta*(2. - rho2))
        L_e = np.log( self.B_el*self.Z13*np.sqrt((1. + xi)*(1. + Y_e)) \
            / (1. + 2*self.m_pair*sqrte*self.B_el*self.Z13*(1. + xi)*(1. + Y_e) \
            / (E_in*v*(1. - rho2))) ) \
            - 0.5*np.log(1. + (3*self.m_pair/(2*self.m_in)/self.Z13)**2*(1. + xi)*(1. + Y_e))
        C_e = ((2. + rho2)*(1. + beta) + xi*(3. + rho2))* np.log(1. + 1./xi) \
            + (1. - rho2 - beta)/(1. + xi) - (3. + rho2)

        return C_e * L_e

    def Phi_mu(self, E_in, v, rho):
        r"""
        calculates the cross section of mu-diagrams
        derived by Kelner, Kokoulin, Petrukhin
        """
        rho2 = rho * rho
        beta = v*v/(2*(1. - v))
        xi = ((self.m_in*v)/(2*self.m_pair))**2*(1. - rho2)/(1. - v)
        
        Y_m = (4. + rho2 + 3*beta*(1. + rho2)) \
            / ((1. + rho2)*(1.5 + 2*beta)*np.log(3 + xi) + 1. - 1.5*rho2)
        L_m = (2*self.m_in)/(3*self.m_pair)*self.B_el*self.Z13*self.Z13 \
            / (1. + 2*self.m_pair*sqrte*self.B_el*self.Z13*(1. + xi)*(1. + Y_m) / (E_in*v*(1. - rho2)))
        C_m = ((1. + rho2)*(1. + 1.5*beta) - 1./xi*(1. + 2*beta)*(1. - rho2)) * np.log(1. + xi) \
            + xi*(1. - rho2 - beta)/(1. + xi) + (1. + 2*beta)*(1. - rho2)
        
        return C_m * np.log(L_m)


class EPairXsection(object):
    def __init__(self,
                medium,
                param_name,
                m_in=m_mu,
                m_pair=m_e,
                epsabs = 1.0e-60,
                epsrel = 1.0e-3):
        self.epsabs = epsabs
        self.epsrel = epsrel

        self.rho_max_log = 0.0
        # self.energy_min = 4 * self.m_pair + 0.75 * sqrte * self.m_in / self.Z13

        if param_name == 'sandrock':
            self.param = EPairSandrock(medium)
        elif param_name == 'kelner':
            self.param = EPairKelner(medium)
        else:
            raise NameError('param_name is not sandrock or kelner')


    def integration_rho(self, E_in, v, func):
        # integrating over 1 - rho
        # to integrate with log substitution
        def func_integrand(rho):
            rho_exp = np.exp(rho)
            return 2 * rho_exp * func(E_in, v, 1. - rho_exp)

        return quad(func_integrand,
            np.log(1. - self.param.rho_max(E_in, v)),
            self.rho_max_log,
            epsabs=self.epsabs, epsrel=self.epsrel)[0]

    def dsigma_dv_e(self, E_in, v):
        return self.integration_rho(E_in, v, self.param.dsigma_dv_drho_e)

    def dsigma_dv_mu(self, E_in, v):
        return self.integration_rho(E_in, v, self.param.dsigma_dv_drho_mu)

    def dsigma_dv(self, E_in, v):
        return self.integration_rho(E_in, v, self.param.dsigma_dv_drho)


    # def integration_rho_v(self, E_in, func):
    #     # integrating over 1 - rho
    #     # and log substitution for v and rho
    #     v_min_log = np.log(self.param.v_min(E_in))
    #     v_max_log = np.log(self.param.v_max(E_in))

    #     def func_integrand(rho, v):
    #         rho_exp = np.exp(rho)
    #         v_exp = np.exp(v)
    #         return rho_exp * v_exp * func(E_in, v_exp, 1. - rho_exp)

    #     def _rho_min(v):
    #         return np.log(1. - self.param.rho_max(E_in, np.exp(v)))
    #     def _rho_max(v):
    #         return self.rho_max_log

    #     return dblquad(func_integrand,
    #         v_min_log, v_max_log,
    #         _rho_min, _rho_max,
    #         epsabs=self.epsabs, epsrel=self.epsrel)

    # def dedx_e(self, E_in):
    #     return self.integration_rho_v(E_in, self.param.dsigma_dv_drho_e)

    # def dedx_mu(self, E_in):
    #     return self.integration_rho_v(E_in, self.param.dsigma_dv_drho_mu)

    # def dedx(self, E_in):
    #     return self.integration_rho_v(E_in, self.param.dsigma_dv_drho)


    def create_energy_loss_array(self, E_in, nsteps):
        return np.logspace(np.log10(self.param.v_min(E_in)),
                            np.log10(self.param.v_max(E_in)),
                            num=nsteps,
                            endpoint=False)


def plot_epair_dsigma_contributions(energy, medium, nsteps=10,
    output='epair_dsigma_contributions.pdf'):
    epair_sandrock = EPairXsection(medium, 'sandrock')
    epair_kelner = EPairXsection(medium, 'kelner')
    xsection_brems = BremsSandrock(medium)

    v_arr = epair_sandrock.create_energy_loss_array(energy, nsteps+1)[1:]
    e_arr = np.empty(nsteps)
    mu_arr = np.empty(nsteps)
    brems_arr = np.empty(nsteps)
    coulomb_arr = np.empty(nsteps)
    eatom_arr = np.empty(nsteps)
    kelner_e = np.empty(nsteps)
    kelner_mu = np.empty(nsteps)

    for idx in tqdm(range(nsteps)):
        e_arr[idx] = epair_sandrock.dsigma_dv_e(energy, v_arr[idx])
        mu_arr[idx] = epair_sandrock.dsigma_dv_mu(energy, v_arr[idx])
        brems_arr[idx] = xsection_brems.dsigma_dv_sum(energy, v_arr[idx])
        eatom_arr[idx] = epair_sandrock.integration_rho(
            energy, v_arr[idx], epair_sandrock.param.dsigma_dv_drho_eatom)
        coulomb_arr[idx] = epair_sandrock.integration_rho(
            energy, v_arr[idx], epair_sandrock.param.dsigma_dv_drho_coulomb)
        kelner_e[idx] = epair_kelner.dsigma_dv_e(energy, v_arr[idx])
        kelner_mu[idx] = epair_kelner.dsigma_dv_mu(energy, v_arr[idx])

    fig = plt.figure()

    gs = GridSpec(4, 1)
    ax1 = fig.add_subplot(gs[:-1])
    ax2 = fig.add_subplot(gs[-1], sharex=ax1)

    ax1.plot(v_arr, e_arr, label=r'$e$-Diagram')
    ax1.plot(v_arr, mu_arr, label=r'$\mu$-Diagram')
    ax1.plot(v_arr, eatom_arr, label=r'Atom. inel.')
    ax1.plot(v_arr, coulomb_arr, label=r'- Coulomb Corr.')
    ax1.plot(v_arr, brems_arr, ls='--', label=r'Bremsstrahlung')

    ax1.set_xscale('log')
    ax1.set_yscale('log')

    # ax1.set_xlabel(r'Relative Energy Loss $v$')
    ax1.set_ylabel(r'Differential Cross Section $\frac{\mathrm{d}\sigma}{\mathrm{d}v} \,/\, \mathrm{cm}^2$')

    ax1.set_xlim(right=1)
    ax1.set_ylim(bottom=1e-11)

    ax1.legend()
    ax1.grid()

    ax2.plot(v_arr, kelner_e/e_arr, label=r'$e$-Diagram')
    ax2.plot(v_arr, kelner_mu/mu_arr, label=r'$\mu$-Diagram')

    ax2.set_xlabel(r'Relative Energy Loss $v$')
    ax2.set_ylabel(r'$\frac{\Phi_1 \approx \Phi_2}{\Phi_1 \neq \Phi_2}$')

    ax2.legend()
    ax2.grid()

    plt.subplots_adjust(hspace=.1)
    plt.setp(ax1.get_xticklabels(), visible=False)

    plt.savefig(output, bbox_inches='tight', pad_inches=0.02)
    plt.close()



def plot_mupair_dsigma(energy, medium, nsteps=10,
    output='mupair_dsigma.pdf'):
    epair_kelner = EPairXsection(medium, 'kelner')
    mupair_kelner = MupairKelner(medium)

    v_arr_e = epair_kelner.create_energy_loss_array(energy, nsteps+1)[1:]
    v_arr_mu = np.logspace(np.log10(mupair_kelner.v_min(energy)),
                        np.log10(mupair_kelner.v_max(energy)),
                        num=nsteps+1,
                        endpoint=False)[1:]
    epair_arr = np.empty(nsteps)
    mupair_arr = np.empty(nsteps)

    for idx in range(nsteps):
        epair_arr[idx] = epair_kelner.dsigma_dv(energy, v_arr_e[idx])
        mupair_arr[idx] = mupair_kelner.dsigma_dv(energy, v_arr_mu[idx])

    fig = plt.figure(figsize=FIG_SIZE)
    ax = fig.add_subplot()

    ax.plot(v_arr_mu, mupair_arr, label=r'$\mu^+ \mu^-$ Pair Production')
    ax.plot(v_arr_e, epair_arr, label=r'$e^+ e^-$ Pair Production')

    ax.set_xscale('log')
    ax.set_yscale('log')

    ax.set_xlabel(r'Relative Energy Loss $v$')
    ax.set_ylabel(r'Differential Cross Section $\frac{\mathrm{d}\sigma}{\mathrm{d}v} \,/\, \mathrm{cm}^2$')

    ax.set_xlim(right=1)
    ax.set_ylim(bottom=1e-11)

    ax.legend()
    ax.grid()

    plt.savefig(output, bbox_inches='tight', pad_inches=0.02)
    plt.close()


if __name__ == "__main__":
    medium = Medium(11, 22)
    energy = 1e6 * MeV
    plot_epair_dsigma_contributions(energy, medium, 100)
    plot_brems_dsigma_contributions(energy, medium, 200)
    plot_mupair_dsigma(energy, medium, 100)
    plot_zeta_effect()

