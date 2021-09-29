
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
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
convert_MeV2_to_cm2 = 0.389379338e-21 * cm * cm
sqrte = np.exp(0.5)
ioniz_const = 0.307075

@np.vectorize
def density_correction(x):
    delta0 = 0.
    x0 = 0.0492
    x1 = 3.0549
    a = 0.08301
    b = 3.4120
    c = 3.7738
    tmp = 2*np.log(10)*x + c
    if x < x0:
        return delta0 * 10**(2*(x - x0))
    elif x < x1:
        return tmp + a*(x1 - x)**b
    else:
        return tmp

class Medium(object):
    def __init__(self, atomic_number_Z, atomic_mass_A):
        self.Z = atomic_number_Z
        self.A = atomic_mass_A
        self.ZA = atomic_number_Z / atomic_mass_A

class ionization(object):

    def __init__(self, medium,
                m_in=m_mu,
                epsabs = 1.0e-60,
                epsrel = 1.0e-3):
        self.m = m_in
        self.ZA = medium.ZA
        self.excitation_energy = 136.4 * 1e-6
        self.const_prefactor = 0.5 * self.ZA * ioniz_const

        self.epsabs = epsabs
        self.epsrel = epsrel

    def v_min(self, E_in):
        betagamma = np.sqrt((E_in + self.m) * (E_in - self.m)) / self.m
        return 1./(2*m_e*E_in) * (self.excitation_energy/betagamma)**2

    def v_max(self, E_in):
        betagamma = np.sqrt((E_in + self.m) * (E_in - self.m)) / self.m
        gamma = E_in / self.m
        return 2 * m_e * betagamma**2 / \
            (1 + 2*gamma * m_e/self.m + (m_e/self.m)**2) / E_in


    def dsigma_dv(self, E_in, v):
        p = np.sqrt((E_in + self.m) * (E_in - self.m))
        beta = p / E_in
        gamma = E_in / self.m
        betagamma = p / self.m
        tmp = 1 - beta**2 * v / self.v_max(E_in) + 0.5 * (v/(1 + 1./gamma))**2
        return self.const_prefactor / (beta*E_in * v)**2 * tmp

    def dsigma_dv_brems_e(self, E_in, v):
        gamma = E_in / self.m
        a = np.log(1 + 2*E_in*v/m_e)
        b = np.log((1 - v/self.v_max(E_in))/(1 - v))
        c = np.log(2*m_e*gamma*(1 - v)/(self.m*v))
        tmp = alpha/(np.exp(1)*np.pi) * (a*(2*b + c) - b*b)
        return self.dsigma_dv(E_in, v) * tmp

    def dedx_integrate(self, E_in, xsection_type):
        if xsection_type == 'Bethe-Bloch':
            def func_integrand(v):
                v_exp = np.exp(v)
                return v_exp**2 * self.dsigma_dv(E_in, v_exp)
        elif xsection_type == 'Brems-e':
            def func_integrand(v):
                v_exp = np.exp(v)
                return v_exp**2 * self.dsigma_dv_brems_e(E_in, v_exp)
        else:
            raise NameError('wrong type {}'.format(xsection_type))

        return E_in**2 * quad(func_integrand,
            np.log(self.v_min(E_in)),
            np.log(self.v_max(E_in)),
            epsabs=self.epsabs, epsrel=self.epsrel)[0]


    def dedx(self, E_in, v_up):
        p = np.sqrt((E_in + self.m) * (E_in - self.m))
        beta = p / E_in
        gamma = E_in / self.m
        betagamma = p / self.m
        tmp1 = np.log(2 * m_e * betagamma**2 * E_in * v_up / \
            self.excitation_energy**2)
        tmp2 = beta**2 * (1 + v_up / self.v_max(E_in))
        tmp3 = (v_up / (2*(1 + 1./gamma)))**2
        return self.const_prefactor / beta**2 * (tmp1 - tmp2 + tmp3)

    def dedx_density(self, E_in):
        p = np.sqrt((E_in + self.m) * (E_in - self.m))
        beta = p / E_in
        betagamma = p / self.m
        x = np.log10(betagamma)
        return self.const_prefactor / beta**2 * density_correction(x)


def plot_dsigma(energy, medium, output='ioniz_dsigma.pdf'):
    ioniz = ionization(medium)
    v_arr = np.geomspace(ioniz.v_min(energy), ioniz.v_max(energy), 200)

    ioniz_arr = ioniz.dsigma_dv(energy, v_arr)
    brems_e_arr = ioniz.dsigma_dv_brems_e(energy, v_arr)

    fig = plt.figure(figsize=FIG_SIZE)
    ax = fig.add_subplot()

    ax.plot(v_arr, v_arr*ioniz_arr, label=r'Bethe-Bloch')
    ax.plot(v_arr, v_arr*brems_e_arr, label=r'$e$-diagram Bremsstrahlung')

    ax.set_xscale('log')
    ax.set_yscale('log')

    ax.set_xlim(right = 1)
    ax.set_xlabel(r'Relative Energy Loss $v$')
    ax.set_ylabel(r'Differential Cross Section $\frac{\mathrm{d}\sigma}{\mathrm{d}v}  \,/\, \mathrm{cm}^2$')

    ax.legend()
    ax.grid()

    if output.endswith('.pdf'):
        plt.savefig(output, bbox_inches='tight', pad_inches=0.02)
    else:
        plt.savefig(output, bbox_inches='tight', pad_inches=0.02, dpi=300)
    ax.cla()
    plt.close()

def plot_dedx(medium, output='ioniz_dedx.pdf'):
    ioniz = ionization(medium)
    e_arr = np.geomspace(m_mu + 1*MeV, 1e9, 200)

    ioniz_arr = ioniz.dedx(e_arr, ioniz.v_max(e_arr))
    ioniz_int_arr = [ioniz.dedx_integrate(eidx, 'Bethe-Bloch') for eidx in e_arr]
    delta_arr = ioniz.dedx_density(e_arr)
    brems_e_arr = [ioniz.dedx_integrate(eidx, 'Brems-e') for eidx in e_arr]

    fig = plt.figure(figsize=FIG_SIZE)
    ax = fig.add_subplot()

    ax.plot(e_arr, ioniz_arr, label=r'Bethe-Bloch $\mathrm{d}E / \mathrm{d}X$')
    ax.plot(e_arr, ioniz_int_arr, label=r'Bethe-Bloch Integration', ls='--')
    ax.plot(e_arr, delta_arr, label=r'- Density correction')
    ax.plot(e_arr, brems_e_arr, label=r'$e$-diagram Bremsstrahlung')

    ax.set_xscale('log')
    ax.set_yscale('log')

    ax.set_xlim(m_mu, max(e_arr))
    ax.set_ylim(bottom=1e-2)

    ax.set_xlabel(r'Muon Energy $E$ / MeV')
    ax.set_ylabel(r'Average Energy Loss $\left\langle -\frac{\mathrm{d}E}{\mathrm{d}X}\right\rangle \,\left/\, \left( \rm{MeV} \rm{g}^{-1} \rm{cm}^2 \right) \right. $')

    ax.legend()
    ax.grid()

    if output.endswith('.pdf'):
        plt.savefig(output, bbox_inches='tight', pad_inches=0.02)
    else:
        plt.savefig(output, bbox_inches='tight', pad_inches=0.02, dpi=300)
    ax.cla()
    plt.close()


if __name__ == '__main__':
    medium = Medium(11, 22)
    plot_dsigma(1e6 * MeV, medium)
    plot_dedx(medium)
