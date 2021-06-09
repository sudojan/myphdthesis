import os
import abc
from collections.abc import Iterable
import matplotlib.pyplot as plt
import numpy as np
from scipy import interpolate
from scipy import integrate
from matplotlib import rc

rc('font', **{'family': 'serif',
   'serif': ['Computer Modern']})
rc('text', usetex=True)
rc('text.latex', preamble=r'\usepackage{calrsfs}')

SCRIPT_PATH = os.path.dirname(os.path.realpath(__file__))

class BaseInterpolatedCrossSection(metaclass=abc.ABCMeta):

    def __init__(self, particle_type, nucleus_type, interaction_type):
        """
        Check input parameter
        """
        if ((particle_type != "nu") & (particle_type != "nubar")):
            raise NameError("particle_type must be nu or nubar, not {}".format(particle_type))
        if ((nucleus_type != "n") & (nucleus_type != "p") & (nucleus_type != "iso")):
            raise NameError("nucleus_type must be n or p or iso, not {}".format(nucleus_type))
        if ((interaction_type != "CC") & (interaction_type != "NC")):
            raise NameError("interaction_type must be CC or NC, not {}".format(interaction_type))

        self._interp = self.__build_interp__(particle_type, nucleus_type, interaction_type)

    @abc.abstractmethod
    def __build_interp__(self, particle_type, nucleus_type, interaction_type):
        """
        Build Interpolator
        """
        pass

    @abc.abstractmethod
    def __call__(self):
        """
        Returns the Interpolated Cross Section for given parameters
        """
        pass


class InterpolatedTotalCrossSection(BaseInterpolatedCrossSection):
    '''
    Load total cross section tables and interpolate them.
    '''
    def __build_interp__(self, particle_type, nucleus_type, interaction_type):
        """
        Build Interpolator with tables
        """
        _table_file_name = "totalX_{}_{}_HERAPDF1.5NLO.txt".format(particle_type, nucleus_type)
        _xsec_path = os.path.join(SCRIPT_PATH, "tables_total_cross_section", _table_file_name)

        if interaction_type == "CC":
            _use_cols = (0,1)
        else:
            _use_cols = (0,4)

        _energy, _xsec = np.loadtxt(_xsec_path,
            unpack=True,
            usecols=_use_cols,
            delimiter=',')

        return interpolate.interp1d(np.log10(_energy), np.log10(_xsec))

    def __call__(self, energy):
        """
        Call Interpolator
        """
        return 10**self._interp(np.log10(energy))


class InterpolatedDifferentialCrossSection(BaseInterpolatedCrossSection):
    '''
    Load differential cross section tables and interpolate them.
    '''
    def __build_interp__(self, particle_type, nucleus_type, interaction_type):
        '''
        Build the interpolator object on the energy and y grid.
        Interpolate in log scale to make sure that the interpolator
        doesnt get in trouble.
        '''
        _table_file_name = "dsigmady_{}_{}_{}_NLO_HERAPDF15NLO_EIG.dat".format(
            particle_type, interaction_type, nucleus_type)
        _xsec_path = os.path.join(SCRIPT_PATH, "tables_dsigma_dy", _table_file_name)

        _y_values, _xsec_values = np.loadtxt(_xsec_path, unpack=True)
        self.n_energies = 111
        self.energy_points = np.logspace(1, 12, self.n_energies)
        self.n_y_points = 100
        energy_grid_v = np.repeat(np.log10(self.energy_points), self.n_y_points)
        self.y_grid = _y_values.reshape(self.n_energies, self.n_y_points)
        self.xsec_values = _xsec_values.reshape(self.n_energies, self.n_y_points)

        return interpolate.LinearNDInterpolator(
            (energy_grid_v, np.log10(_y_values)),
            np.log10(_xsec_values),
            fill_value=0)
    
    def __call__(self, e_log, y_log):
        '''
        Call the interpolator with a logarithmic energy and logarithmic y.
        '''
        if isinstance(e_log, Iterable) and isinstance(y_log, Iterable):
            assert len(e_log) == len(y_log), \
                'shape of flattened `e_log` and `y_log` should be the same!'
        elif isinstance(e_log, Iterable) or isinstance(y_log, Iterable):
            if isinstance(e_log, Iterable):
                y_log = np.repeat(y_log, len(e_log))
            elif isinstance(y_log, Iterable):
                e_log = np.repeat(e_log, len(y_log))
        return 10**self._interp(e_log, y_log)

    def average_y(self, energy):
        exp_y_simps = np.array([
            integrate.simps(self.y_grid[idx] * self.xsec_values[idx], x=self.y_grid[idx]) / \
            (integrate.simps(self.xsec_values[idx], x=self.y_grid[idx])) for idx in range(self.n_energies)])


        interp = interpolate.interp1d(np.log10(self.energy_points), np.log10(exp_y_simps))
        return 10**interp(np.log10(energy))

    def sample_y(self, e_log, y_log=None, n_samples=1):
        if y_log is None:
            e_nu = 10**e_log
            proton_mass = 0.938 # in GeV
            # y_min = Q^2_min / s with s = 2*mp*Enu
            # Q^2_min is taken to be 1 GeV^2, below this threshold
            # perturbative QCD no longer applies.
            y_min = 1 / (2 * proton_mass * e_nu)
            y_log = np.logspace(np.log10(y_min), np.log10(1), 101)
        
        xsec = self.__call__(e_log, y_log)
        xsec_max = np.max(xsec)
        
        rand_uni = np.random.uniform(size=(n_samples, 2))
        xsec_at_u = self.__call__(e_log, np.log10(rand_uni[:, 0]))
        failed_mask = rand_uni[:, 1] > (xsec_at_u / xsec_max)
        n_failed = np.sum(failed_mask)
        while n_failed != 0:
            new_rand_uni = np.random.uniform(size=(n_failed, 2))
            rand_uni[failed_mask] = new_rand_uni
            xsec_at_u[failed_mask] = self.__call__(e_log, np.log10(new_rand_uni[:, 0]))
            failed_mask = rand_uni[:, 1] > (xsec_at_u / xsec_max)
            n_failed = np.sum(failed_mask)

        if n_samples == 1:
            return rand_uni[0, 0]
        else:
            return rand_uni[:, 0]

def glashow_resonance(energy):
    r""" Total Cross Section of the Glashow resonance
    Calculation taken from 1407.3255
    """
    m_e = 510.9989e-6 # mass in GeV
    m_W_2 = 80.379**2 # mass in GeV^2
    BR_W_enu = 0.1071 # Branching Ratio
    Gamma_W_2 = 2.085**2 # decay width in GeV^2
    sigma_res = 24 * np.pi * BR_W_enu / m_W_2
    s = 2 * energy * m_e + m_e**2
    return (Gamma_W_2 * s / ((s - m_W_2)**2 + m_W_2 * Gamma_W_2)) * sigma_res

def plot_total_xsection(filename, emin, emax):
    e_arr = np.logspace(np.log10(emin), np.log10(emax), num=100)
    convert_GeV_to_cm = 0.389e-27
    convert_picobarn_to_cm = 1e-36

    fig = plt.figure()
    ax = fig.add_subplot(111)

    total_xsection = InterpolatedTotalCrossSection("nu", "iso", "CC")
    ax.plot(e_arr, total_xsection(e_arr), label=r'CC: $\nu + N$', c='tab:blue')

    total_xsection = InterpolatedTotalCrossSection("nubar", "iso", "CC")
    ax.plot(e_arr, total_xsection(e_arr), label=r'CC: $\bar{\nu} + N$', c='tab:blue', ls='dashed')

    total_xsection = InterpolatedTotalCrossSection("nu", "iso", "NC")
    ax.plot(e_arr, total_xsection(e_arr), label=r'NC: $\nu + N$', c='tab:orange')

    total_xsection = InterpolatedTotalCrossSection("nubar", "iso", "NC")
    ax.plot(e_arr, total_xsection(e_arr), label=r'NC: $\bar{\nu} + N$', c='tab:orange', ls='dashed')
    
    e_arr = np.logspace(np.log10(emin), np.log10(emax), num=400)
    ax.plot(e_arr, glashow_resonance(e_arr) * convert_GeV_to_cm / convert_picobarn_to_cm,
        label=r'GR: $\bar{\nu} + e$', c='tab:green', ls='-.')

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.legend()
    ax.set_xlabel(r'Neutrino Energy $E_{\nu}$ / GeV')
    ax.set_ylabel(r'Neutrino Cross Section $\sigma_{\mathrm{tot}}$ / pb')
    ax.set_xlim(emin, emax)

    fig.savefig(filename, bbox_inches='tight', pad_inches=0.02)

def plot_average_y(filename, emin, emax):
    e_arr = np.logspace(np.log10(emin), np.log10(emax), num=100)

    fig = plt.figure()
    ax = fig.add_subplot(111)

    xsec = InterpolatedDifferentialCrossSection("nu", "iso", "CC")
    ax.plot(e_arr, xsec.average_y(e_arr), label=r'$\nu$ CC', c='tab:blue')
    xsec = InterpolatedDifferentialCrossSection("nubar", "iso", "CC")
    ax.plot(e_arr, xsec.average_y(e_arr), label=r'$\bar{\nu}$ CC', c='tab:blue', ls='dashed')

    xsec = InterpolatedDifferentialCrossSection("nu", "iso", "NC")
    ax.plot(e_arr, xsec.average_y(e_arr), label=r'$\nu / \bar{\nu}$ NC', c='tab:orange')
    # xsec = InterpolatedDifferentialCrossSection("nubar", "iso", "NC")
    # ax.plot(e_arr, xsec.average_y(e_arr), label=r'$\bar{\nu}$ NC', ls='dashed')

    ax.set_xscale('log')
    ax.set_xlabel(r'Neutrino Energy $E_{\nu}$ / GeV')
    ax.set_ylabel(r'Fractional Energy of hadr. Cascade $\langle y \rangle$')
    ax.set_ylim(bottom=0)
    ax.set_xlim(emin, emax)
    ax.legend()

    fig.savefig(filename, bbox_inches='tight', pad_inches=0.02)

if __name__ == '__main__':
    plot_total_xsection('nu_xsection_tot.pdf', 1e2, 1e11)
    plot_average_y('nu_xsection_average_y.pdf', 1e2, 1e11)
