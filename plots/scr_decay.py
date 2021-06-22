import argparse
from tqdm import tqdm
import numpy as np
from scipy import constants
from scipy.optimize import brentq
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib import rc

rc('font', **{'family': 'serif',
   'serif': ['Computer Modern']})
rc('text', usetex=True)
rc('text.latex', preamble=r'\usepackage{calrsfs}')

FIG_SIZE = (5.78, 3.57)

GeV = 1.0e3 # MeV
m_e = constants.value('electron mass energy equivalent in MeV')
m_mu = constants.value('muon mass energy equivalent in MeV')
m_tau = constants.value('tau mass energy equivalent in MeV')
G_fermi = constants.value('Fermi coupling constant') / (GeV*GeV)
pi = constants.pi

def particle_name_to_mass_converter(particle_str):
    r""" get mass from name string

    converts the name sting of a lepton
    (electron, muon, tau)
    to its mass

    Parameters
    ----------
    particle_str : str
        string of the lepton name

    Returns
    -------
    mass : float
        mass of the desired lepton
    """
    return {
        "electron"  : m_e,
        "muon"      : m_mu,
        "tau"       : m_tau
        }[particle_str]

def particle_name_to_symbol_converter(particle_str):
    r""" get symbol from name string

    converts the name sting of a lepton
    (electron, muon, tau)
    to its symbol (e, $\mu$, $\tau$)

    Parameters
    ----------
    particle_str : str
        string of the lepton name

    Returns
    -------
    symbol : str
        symbol of the desired lepton
    """
    return {
        "electron"  : r"$e$",
        "muon"      : r"$\mu$",
        "tau"       : r"$\tau$"
        }[particle_str]

class cs_calc(object):
    def __init__(self,
                decay_leptopn_str="muon",
                produced_lepton_str="electron",
                nsteps=10000):
        self.dec_str = decay_leptopn_str
        self.dec_str_short = particle_name_to_symbol_converter(decay_leptopn_str)
        self.prod_str = produced_lepton_str
        self.prod_str_short = particle_name_to_symbol_converter(produced_lepton_str)

        self.m_d = particle_name_to_mass_converter(decay_leptopn_str)
        self.m_p = particle_name_to_mass_converter(produced_lepton_str)

        self.Energy_min = self.m_p
        self.Energy_max = (self.m_d**2 + self.m_p**2)/(2*self.m_d)

        self.x_min = self.Energy_min/self.Energy_max
        self.x_max = 1.0

        self.nsteps = nsteps

        self._init_f()

    def _init_f(self):
        pass

    def get_right_side(self, rnd):
        return self.f_min + (self.f_max - self.f_min) * rnd

    def sample_spectrum(self):
        r""" Sample energy spectrum

        Sample the energy spectrum for the produced lepton
        in the desired process (approx or not)

        Parameters
        ----------
        formula : str
            with approximation (PDG) or without (LP)

        Returns
        -------
        sampled_energy : numpy float array
            array of the sampled energies for the desierd process
        """

        right_side_arr = self.get_right_side(np.random.rand(self.nsteps))

        def func(x, right_side):
            return self.integrated_spectrum(x) - right_side

        sampled_x = np.empty(self.nsteps)
        for idx in tqdm(range(self.nsteps)):
            sampled_x[idx] = brentq(func,
                                    self.x_min,
                                    self.x_max,
                                    xtol=1e-5,
                                    rtol=1e-5,
                                    maxiter=1000,
                                    args=(right_side_arr[idx]))
        return sampled_x

    def norm_hist(self, sampled_arr, with_errors=False):
        heights, bins = np.histogram(sampled_arr,
                                bins=30,
                                density=False,
                                range=(self.x_min, self.x_max))
        norm = np.diff(bins) * np.sum(heights)
        heights = heights / norm

        if with_errors:
            err = np.sqrt(heights)
            err = err / norm
            bincenters = 0.5 * (bins[1:] + bins[:-1])
            return (heights, bins, err, bincenters)
        else:
            return (heights, bins)


class cs_calc_neutrino(object):
    def __init__(self, m_primary=m_mu):
        self.m_primary = m_primary
        self.y_min = 0
        self.y_max = 1
        self.prefac = G_fermi**2 * m_primary**5 / (32 * pi**3)

    def integrated_spectrum(self, y):
        return (1/3. - 0.25 * y)*y**3

    def dNdx_normed(self, y):
        return (1 - y) * y*y / self.integrated_spectrum(self.y_max)

    def dNdx(self, y):
        return self.prefac * (1 - y) * y*y

class cs_calc_PDG(cs_calc):

    def _init_f(self):
        self.f_min = self.integrated_spectrum(self.x_min)
        self.f_max = 0.5
        self.prefac = G_fermi**2 * self.m_d**5 / (192 * pi**3)

    def integrated_spectrum(self, x):
        r""" integrated approx. decay width function

        indefinite integral of the approximated decay width (PDG)
        .. math:: x^3(1 - 0.5x)
        The approximation is $m_l^2 / M^2 \approx 0$

        Parameters 
        ----------
        x : float
            relative energy of the produced lepton compared to its maximum energy

        Returns
        -------
        integrated_approx_decay_width : float
            indefinite integral of the approximated decay width
        """
        return x*x*x*(1 - 0.5*x)

    def dNdx_normed(self, x):
        r""" approx differential decay width
        .. math:: x^2(3 - 2x)

        Parameters 
        ----------
        x : float
            relative energy of the produced lepton compared to its maximum energy

        Returns
        -------
        approx_decay_width : float
            approximated decay width
        """
        return (3 - 2*x)*x*x / (self.f_max - self.f_min)

    def dNdx(self, x):
        return self.prefac * (3 - 2*x)*x*x


class cs_calc_LP(cs_calc):

    def _init_f(self):
        self.f_min = 3 * self.m_p**4 * self.m_d * np.log(self.m_p)
        self.f_max = self.integrated_spectrum(self.x_max)
        self.prefac = G_fermi**2 * self.m_d**4 / (24 * pi**3)

    def integrated_spectrum(self, x):
        r""" integrated decay width function

        indefinite integral of the decay width (Lahiri, Pal)

        Parameters 
        ----------
        x : float
            relative energy of the produced lepton compared to its maximum energy

        Returns
        -------
        integrated_decay_width : float
            indefinite integral of the decay width
        """
        E_e = self.Energy_max * x
        mp2 = self.m_p*self.m_p
        Ee2 = E_e*E_e
        sqrt_Emp = np.sqrt((E_e - self.m_p)*(E_e + self.m_p))
        return 1.5 * mp2*mp2 * self.m_d * np.log(sqrt_Emp + E_e) + \
            sqrt_Emp * ( (self.m_d*self.m_d + mp2 - self.m_d*E_e)*(Ee2 - mp2) \
            - 1.5*self.m_d*E_e*mp2 )

    def dNdx_normed(self, x):
        r""" differential decay width
        From Lahiri Pal

        Parameters 
        ----------
        x : float
            relative energy of the produced lepton compared to its maximum energy

        Returns
        -------
        decay_width : float
            decay width
        """
        E_e = self.Energy_max * x
        return self.Energy_max * np.sqrt((E_e - self.m_p)*(E_e + self.m_p)) * \
            (self.m_d * E_e * (3 * self.m_d - 4 * E_e) + \
            self.m_p * self.m_p * (3 * E_e - 2 * self.m_d)) / \
            (self.f_max - self.f_min)

    def dNdx(self, x):
        Ee_md = self.Energy_max * x / self.m_d
        mp_md = self.m_p / self.m_d
        sqrt_Emp = np.sqrt((Ee_md - mp_md)*(Ee_md + mp_md))
        return self.prefac * self.Energy_max * sqrt_Emp * \
            (3 * Ee_md - 4 * Ee_md**2 + \
            mp_md**2 * (3 * Ee_md - 2))


def plot_diff_decay_width(PDG_clac, lp_clac,
    output='decay_spectrum_compare.pdf', normed=False):
    calc_nu = cs_calc_neutrino()
    x_arr = np.geomspace(PDG_clac.x_min, PDG_clac.x_max, 200)
    y_arr = np.logspace(-3, 0, 200)
    if normed:
        PDG_arr = calc_PDG.dNdx_normed(x_arr)
        LP_arr = calc_LP.dNdx_normed(x_arr)
        nu_arr = calc_nu.dNdx_normed(y_arr)
    else:
        PDG_arr = calc_PDG.dNdx(x_arr)
        LP_arr = calc_LP.dNdx(x_arr)
        nu_arr = calc_nu.dNdx(y_arr)

    fig = plt.figure(figsize=FIG_SIZE)
    gs = gridspec.GridSpec(3, 1)
    ax1 = fig.add_subplot(gs[:-1])
    ax2 = fig.add_subplot(gs[-1], sharex=ax1)

    ax1.plot(x_arr, PDG_arr/x_arr, label="PDG")
    ax1.plot(x_arr, LP_arr/x_arr, label="LahiriPal", ls='--')
    ax1.plot(y_arr, nu_arr/y_arr, label="Neutrino", ls=':')

    ax1.legend()
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.set_ylabel(r'diff. Decay Width $x^{-1} \mathrm{d}\Gamma / \mathrm{d}x$')
    ax1.grid()
    ax1.set_xlim(8e-3, 1.05)

    ax2.plot(x_arr, 1-LP_arr/PDG_arr, c='k',
        label=r"$1 - \frac{\mathrm{LahiriPal}}{\mathrm{PDG}}$")
    ax2.legend()
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax2.set_xlabel(r'rel. Electron Energy $x=E_e/E_{\mathrm{max}}$')
    ax2.set_ylabel(r'rel. Deviation')
    ax2.grid()

    plt.subplots_adjust(hspace=.1)
    plt.setp(ax1.get_xticklabels(), visible=False)

    plt.savefig(output, bbox_inches='tight', pad_inches=0.02)
    plt.close()



def plot_root_finding_test(PDG_clac, lp_clac):
    ran = 0.8
    x_arr = np.linspace(PDG_clac.x_min, PDG_clac.x_max, 100)
    y_PDG = PDG_clac.integrated_spectrum(x_arr) - PDG_clac.get_right_side(ran)
    y_LP = lp_clac.integrated_spectrum(x_arr) - lp_clac.get_right_side(ran)

    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    # ax2 = fig.add_subplot(212)
    ax1.plot(x_arr, y_LP*8/lp_clac.m_d**5, color="C0", label="LP")
    ax1.plot(x_arr, y_PDG, color="C1", label="PDG", ls='--')
    ax1.axhline(0., color="C2")
    ax1.legend()
    # ax1.grid()
    # fig.savefig("test_{}_to_{}.pdf".format(self.dec_str, self.prod_str))
    plt.show()

def plot_sepctrum(calc_PDG, calc_LP, types):

    fig = plt.figure()
    gs = gridspec.GridSpec(4, 1)
    ax1 = fig.add_subplot(gs[:-1])
    ax2 = fig.add_subplot(gs[-1], sharex=ax1)

    if types == "theory" or types == "both":
        energy_arr = np.linspace(calc_LP.x_min, calc_LP.x_max, 200)
        PDG_arr = calc_PDG.dNdx_normed(energy_arr)
        LP_arr = calc_LP.dNdx_normed(energy_arr)

        ax1.plot(energy_arr, LP_arr, label="LahiriPal", color="C0")
        ax1.plot(energy_arr, PDG_arr, label="PROPOSAL", color="C1")

        if types == "theory":
            ax2.plot(energy_arr, abs(1 - LP_arr/PDG_arr), label=r"LahiriPal/PROPOSAL")
            ax2.set_yscale('log')
            ax2.set_ylabel(r'|1 - ratio|')
        else:
            ax2.plot(energy_arr, LP_arr/PDG_arr, label=r"LahiriPal/PROPOSAL theory")
            ax2.set_ylabel(r'ratio')

    if types == "sampling" or types == "both":
        np.random.seed(123)
        h_LP, bins = calc_LP.norm_hist(calc_LP.sample_spectrum())
        ax1.plot(bins, np.r_[h_LP[0], h_LP], drawstyle='steps-pre', color="C0", label="LP sampled")
        # h_LP, bins, err_LP, bincenters = calc_LP.norm_hist(calc_LP.sample_spectrum(), True)
        # ax1.errorbar(bincenters, h_LP, yerr=err_LP, color="C0", fmt=',')

        np.random.seed(123)
        h_PDG, bins = calc_PDG.norm_hist(calc_PDG.sample_spectrum())
        # h_PDG, bins, err_PDG, bincenters = calc_PDG.norm_hist(calc_PDG.sample_spectrum(), True)
        # ax1.errorbar(bincenters, h_PDG, yerr=err_PDG, color="C1", fmt=',')
        ax1.plot(bins, np.r_[h_PDG[0], h_PDG], drawstyle='steps-pre', color="C1", label="PDG sampled")

        h_ratio = h_LP / h_PDG
        ax2.plot(bins, np.r_[h_ratio[0], h_ratio],
            drawstyle='steps-pre', label=r"LahiriPal/PROPOSAL sampling")

    ax1.set_ylabel(r'normed $\mathrm{d}\Gamma / \mathrm{d}x$')
    ax1.legend()
    ax1.grid()

    ax2.set_xlabel(r'$x = E_l / E_{l, \mathrm{max}}$')
    ax2.legend()
    ax2.grid()

    ax1.set_title(r"$10^{{{:.2g}}}$ {} decays: {} $\to$ {} (+ 2 $\nu$)".format(
        np.log10(calc_LP.nsteps),
        calc_LP.dec_str,
        calc_LP.dec_str_short,
        calc_LP.prod_str_short))
    plt.subplots_adjust(hspace=.0)
    plt.setp(ax1.get_xticklabels(), visible=False)
    fig.savefig("decay_spectrum_{}_to_{}.pdf".format(calc_LP.dec_str, calc_LP.prod_str))
    # plt.show()
    ax1.cla()
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-d','--decaying', type=str,
                        dest='decaying_lepton', default="muon",
                        help='lepton, that should decay')
    parser.add_argument('-p','--producing', type=str,
                        dest='producing_lepton', default="electron",
                        help='lepton, that is produced during the decay')
    parser.add_argument('-n','--nsteps', type=int,
                        dest='nsteps', default=10000,
                        help='number of bins in histogram')
    parser.add_argument('-t','--type', type=str,
                        dest='plot_type', default="theory",
                        help='plot type: theory, sampling or both')
    args = parser.parse_args()

    calc_PDG = cs_calc_PDG(args.decaying_lepton, args.producing_lepton, args.nsteps)
    calc_LP = cs_calc_LP(args.decaying_lepton, args.producing_lepton, args.nsteps)
    # plot_sepctrum(calc_PDG, calc_LP, args.plot_type)
    # plot_root_finding_test(calc_PDG, calc_LP)
    plot_diff_decay_width(calc_PDG, calc_LP)

