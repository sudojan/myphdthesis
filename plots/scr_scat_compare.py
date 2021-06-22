
import numpy as np
import matplotlib.pyplot as plt
from scipy import constants
from scipy import stats
from tqdm import tqdm
from matplotlib import rc
import proposal as pp

rc('font', **{'family': 'serif',
   'serif': ['Computer Modern']})
rc('text', usetex=True)
rc('text.latex', preamble=r'\usepackage{calrsfs}')

FIG_SIZE = (5.78, 3.57)

m_e = constants.value('electron mass energy equivalent in MeV')
m_mu = constants.value('muon mass energy equivalent in MeV')

def sample_brems(rnd, energy, v):
    theta_star = 1
    eps_ef = v/(1 - v)
    rmax = min(1, 1/eps_ef) * energy * theta_star / m_mu
    a = rnd * rmax**2 / (1 + rmax**2)
    r = np.sqrt(a / (1-a))
    theta_photon = m_mu / energy * r
    return eps_ef * theta_photon

def sample_epair(rnd, energy, v):
    a = 8.9 / 10000.0 
    b = 1.5 / 100000.0
    c = 0.032
    nu = v/(1 - m_mu/energy)
    min_ = min(a * nu**0.25 * (1 + b * energy/1e3) + c * nu / (nu + 1), 0.1)
    rms_theta = (2.3 + np.log(energy/1e3)) / ((1 - nu) * energy/1e3) * (1 - 2 * m_e / (nu*energy))**2  * min_
    exp_sample = - np.log(1 - rnd) * rms_theta**2
    return np.sqrt(exp_sample)

def sample_photo(rnd, energy, v):
    m02 = 0.4 * 1e6 # GeV^2
    eps = energy*v
    m_p = constants.value('proton mass energy equivalent in MeV')
    t_1 = min(eps**2, m02)
    t_1_max = t_1 / (2 * m_p * eps)
    t_min = m_mu**2 * v**2 / (1 - v)
    t_p = t_1 / ((1 + t_1_max) * ((1 + t_1/t_min) / (1 + t_1_max))**rnd - 1)
    sin2 = (t_p - t_min) / (4 * (energy**2 * (1-v) - m_mu**2) - 2 * t_min)
    return 2 * np.arcsin(np.sqrt(sin2))

def calc_ioniz(energy, v):
    e_f = energy*(1-v)
    p_i = np.sqrt((energy + m_mu)*(energy - m_mu))
    p_f = np.sqrt((e_f + m_mu)*(e_f - m_mu))
    costheta = ((energy + m_e)*e_f - energy*m_e - m_mu**2) / (p_i * p_f)
    return np.arccos(costheta)

def plot_deflections(energy, rel_energy_loss, npts,
    output='scattering_deflection_compare.pdf'):
    rnd_arr = np.random.rand(npts)
    ioniz_val = calc_ioniz(energy, rel_energy_loss)
    brems_arr = sample_brems(rnd_arr, energy, rel_energy_loss)
    epair_arr = sample_epair(rnd_arr, energy, rel_energy_loss)
    photo_arr = sample_photo(rnd_arr, energy, rel_energy_loss)

    fig = plt.figure()
    ax = fig.add_subplot()

    bins = np.logspace(-9, 1, 100)
    ax.hist(brems_arr, bins=bins, histtype='step', label='Brems')
    ax.hist(epair_arr, bins=bins, histtype='step', label='epair')
    ax.hist(photo_arr, bins=bins, histtype='step', label='photo')
    ax.axvline(ioniz_val, label='ioniz', c='C3')
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.set_xlabel(r'muon deflection angle $\theta$ / rad')
    ax.legend(loc='upper left')

    plt.savefig(output, bbox_inches='tight', pad_inches=0.02)

class Scatter_wrapper(object):
    def __init__(self, pdef, medium, param_name, energy, distance):
        self.pdef = pdef
        self.medium = medium
        self.scat_param = pp.make_multiple_scattering(
            param_name, pdef, medium)
        # look in z direction to consider only the zenith angle
        self._dir_init = pp.Cartesian3D(0, 0, 1)
        self._energy = energy
        self._distance = distance

    # def calc_disp_energy(self):
    #     args = {
    #         "particle_def": self.pdef,
    #         "target": self.medium,
    #         "interpolate": False,
    #         "cuts": pp.EnergyCutSettings(np.inf, 1, False)
    #     }
    #     cross = pp.crosssection.make_std_crosssection(**args)
    #     disp_calc = pp.make_displacement(cross, True)
    #     return disp_calc.upper_limit_track_integral(
    #         self._energy, self._distance * self.medium.mass_density)

    def calc_scattering_angle(self):

        rnd1 = pp.RandomGenerator.get().random_double()
        rnd2 = pp.RandomGenerator.get().random_double()
        rnd3 = pp.RandomGenerator.get().random_double()
        rnd4 = pp.RandomGenerator.get().random_double()
        coords = self.scat_param.scatter(
            self._distance * self.medium.mass_density,
            self._energy,
            0, # dummy value, for energy_disp
            [rnd1, rnd2, rnd3, rnd4])
        direction = pp.scattering.scatter_initial_direction(
            self._dir_init, coords)[0]
        return direction.spherical_coordinates[-1] # only zenith angle

def calc_theta_0(energy, distance, pdef, medium):
    x_x0 = distance * medium.mass_density / medium.radiation_length
    beta_p = (energy - pdef.mass) * (energy + pdef.mass) / energy
    return 13.6 / beta_p * np.sqrt(x_x0) * (1 + 0.088 * np.log10(x_x0))

def plot_angles(pdef, medium, npts, energy, distance,
    output='scat_multi_compare.pdf'):

    scat_hl = Scatter_wrapper(pdef, medium, "Highland", energy, distance)
    scat_mol = Scatter_wrapper(pdef, medium, "Moliere", energy, distance)

    # energy_disp = scat_hl.calc_disp_energy()
    # #this is pre-calculated for a purely continuous 
    # # propagation of a muon with 1TeV through 10m StandardRock
    # # to shorten the calculation, without building Interpolation tables
    # energy_disp = 982567.8367161071
    # # This is only important for HighlandIntegral, not considered, here

    theta_hl = np.empty(npts)
    theta_mol = np.empty(npts)
    for idx in tqdm(range(npts)):
        theta_hl[idx] = scat_hl.calc_scattering_angle()
        theta_mol[idx] = scat_mol.calc_scattering_angle()

    # bins = np.linspace(0, np.max([theta_hl, theta_mol]))
    fig = plt.figure(figsize=FIG_SIZE)
    ax = fig.add_subplot()
    
    bins = np.linspace(0, 2, 100)
    ax.hist(theta_hl*1e3, bins=bins, histtype='step', label='Highland')
    ax.hist(theta_mol*1e3, bins=bins, histtype='step', label='Moliere')
    ax.axvline(calc_theta_0(energy, distance, pdef, medium)*1e3,
        label=r'$\theta_0$', ls='--')

    ax.set_yscale('log')
    ax.set_xlabel(r'Muon Scattering Angle $\theta$ / mrad')
    ax.legend()

    plt.savefig(output, bbox_inches='tight', pad_inches=0.02)
    plt.close()

def stochastic_deflect(energy, rel_energy_loss, v_2, npts, output='scat_stoch_compare.pdf'):
    mudef = pp.particle.MuMinusDef()
    med = pp.medium.StandardRock()
    epair_param = pp.make_stochastic_deflection('kelnerpairproduction', mudef, med)
    brems_param = pp.make_stochastic_deflection('tsaiapproximationbremsstrahlung', mudef, med)
    photo_param = pp.make_stochastic_deflection('borogpetrukhinnuclearinteraction', mudef, med)
    ioniz_param = pp.make_stochastic_deflection('naivionization', mudef, med)

    # dummy variable for last argument, sampling the azimuth
    ioniz_val = ioniz_param.stochastic_deflection(
            energy, (1-rel_energy_loss)*energy, [0.1]).zenith

    fig = plt.figure(figsize=FIG_SIZE)
    ax = fig.add_subplot()

    bins = np.logspace(-10, -1, 100)
    rnd_arr = np.random.rand(npts)

    epair_arr = np.empty(npts)
    brems_arr = np.empty(npts)
    photo_arr = np.empty(npts)

    for idx in tqdm(range(npts)):
        # second rnd is a dummy value to sample the azimuth, not relevant, here
        epair_arr[idx] = epair_param.stochastic_deflection(
            energy, (1-rel_energy_loss)*energy, [rnd_arr[idx], 0.5]).zenith
        brems_arr[idx] = brems_param.stochastic_deflection(
            energy, (1-rel_energy_loss)*energy, [rnd_arr[idx], 0.5]).zenith
        photo_arr[idx] = photo_param.stochastic_deflection(
            energy, (1-rel_energy_loss)*energy, [rnd_arr[idx], 0.5]).zenith

    ax.hist(brems_arr, bins=bins, histtype='step', label='Bremsstrahlung')
    ax.hist(epair_arr, bins=bins, histtype='step', label='$e$ pair production')
    ax.hist(photo_arr, bins=bins, histtype='step', label='Photonuclear')
    ax.axvline(ioniz_val, label='Ionization', c='C3')

    for idx in tqdm(range(npts)):
        # second rnd is a dummy value to sample the azimuth, not relevant, here
        epair_arr[idx] = epair_param.stochastic_deflection(
            energy, (1-v_2)*energy, [rnd_arr[idx], 0.5]).zenith
        brems_arr[idx] = brems_param.stochastic_deflection(
            energy, (1-v_2)*energy, [rnd_arr[idx], 0.5]).zenith
        photo_arr[idx] = photo_param.stochastic_deflection(
            energy, (1-v_2)*energy, [rnd_arr[idx], 0.5]).zenith

    ioniz_val = ioniz_param.stochastic_deflection(
            energy, (1-v_2)*energy, [0.1]).zenith

    ax.hist(brems_arr, bins=bins, color='C0', histtype='step', ls='--')
    ax.hist(epair_arr, bins=bins, color='C1', histtype='step', ls='--')
    ax.hist(photo_arr, bins=bins, color='C2', histtype='step', ls='--')
    ax.axvline(ioniz_val, c='C3', ls='--')

    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.set_xlabel(r'Muon Deflection Angle $\theta$ / rad')
    ax.legend()

    plt.savefig(output, bbox_inches='tight', pad_inches=0.02)


def multiple_scattering():
    mudef = pp.particle.MuMinusDef()
    medium = pp.medium.StandardRock()

    energy = 1e6 # MeV
    distance = 1e3 # cm

    # pp.RandomGenerator.get().set_seed(1234)
    np.random.seed(1234)
    npts = int(1e6)
    plot_angles(mudef, medium, npts, energy, distance)



def self_implemented_stochastic():

    np.random.seed(1234)
    npts = 1000000
    e_i = 1e6
    for v in [0.001, 0.01, 0.1, 0.9, 0.99]:
        plot_deflections(e_i, v, npts, "test_{:.4f}.pdf".format(v))


if __name__ == '__main__':
    # multiple_scattering()
    stochastic_deflect(1e6, 0.1, 0.001, 1000000)
    # self_implemented_stochastic()

