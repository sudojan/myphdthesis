
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.colors import LogNorm
from matplotlib import rc
from scipy.optimize import curve_fit
from tqdm import tqdm
import proposal as pp

rc('font', **{'family': 'serif',
   'serif': ['Computer Modern']})
rc('text', usetex=True)
rc('text.latex', preamble=r'\usepackage{calrsfs}')

FIG_SIZE = (5.78, 3.57)

pp.InterpolationSettings.tables_path = "tables_proposal_interpol"

def average_range_calc(energies, a, b):
    return np.log(1 + b*energies/a) / b

def calc_dEdx_params(energies, dedx_sum):
    def func(x, a, b):
        return a + b*x
    return curve_fit(func, energies, dedx_sum, sigma=dedx_sum)[0]

def create_propagator(args):

    cross_list = pp.crosssection.make_std_crosssection(**args)

    collection = pp.PropagationUtilityCollection()

    collection.displacement = pp.make_displacement(cross_list, True)
    collection.interaction = pp.make_interaction(cross_list, True)
    collection.time = pp.make_time_approximate()

    utility = pp.PropagationUtility(collection = collection)
    
    detector = pp.geometry.Sphere(pp.Cartesian3D(0,0,0), 1e20)
    density_distr = pp.density_distribution.density_homogeneous(args["target"].mass_density)
    
    return pp.Propagator(args["particle_def"], [(detector, utility, density_distr)])

def prop_mu(prop, energy, nmuons):

    init_state = pp.particle.ParticleState()
    init_state.energy = energy
    init_state.position = pp.Cartesian3D(0, 0, 0)
    init_state.direction = pp.Cartesian3D(0, 0, 1)
    init_state.time = 0
    init_state.propagated_distance = 0

    final_ranges = np.empty(nmuons)
    for idx in range(nmuons):
        track = prop.propagate(init_state)
        final_ranges[idx] = track.track()[-1].propagated_distance

    return final_ranges

def get_range_data(energies, n_muon_per_bin, data_file, args):

    if os.path.isfile(data_file):
        return np.genfromtxt(data_file)

    prop = create_propagator(args)
    range_data = np.empty((len(energies), n_muon_per_bin))
    for idx, energy in tqdm(enumerate(energies)):
        range_data[idx] = prop_mu(prop, energy, n_muon_per_bin)

    np.savetxt(data_file, range_data)
    return range_data

def plot_range_distribution(
    args,
    energy_bin_edges_mids,
    range_bin_edges_mids,
    ranges,
    output_file):

    energy_bin_edges = energy_bin_edges_mids[::2]
    energy_bin_mids = energy_bin_edges_mids[1::2]
    range_bin_edges = range_bin_edges_mids[::2]
    range_bin_mids = range_bin_edges_mids[1::2]

    args["cuts"] = pp.EnergyCutSettings(np.inf, 1, False)
    cross_list = pp.crosssection.make_std_crosssection(**args)
    xsection = np.empty((len(cross_list), len(energy_bin_mids)))

    for idx, cross in enumerate(cross_list):
        xsection[idx] = cross.calculate_dEdx(energy_bin_mids)
    dedx_sum = np.sum(xsection, axis=0)
    fit_params = calc_dEdx_params(energy_bin_mids, dedx_sum)
    dedx_ranges = average_range_calc(energy_bin_mids, *fit_params)


    fig = plt.figure()
    gs = gridspec.GridSpec(3, 1)
    ax1 = fig.add_subplot(gs[:-1])
    ax2 = fig.add_subplot(gs[-1], sharex=ax1)
    # ax1.set_title('statistic per energy bin: {}'.format(len(ranges[0])))

    ranges_hist2d = np.empty((len(energy_bin_mids), len(range_bin_mids)))
    for idx in range(len(energy_bin_mids)):
        ranges_hist2d[idx] = np.histogram(ranges[idx], bins=range_bin_edges)[0]

    Xe, Ye = np.meshgrid(energy_bin_edges, range_bin_mids)
    im = ax1.pcolormesh(Xe, Ye, np.atleast_2d(ranges_hist2d.T), norm=LogNorm())


    # average_ranges = np.average(ranges, axis=1)
    median_ranges = np.median(ranges, axis=1)
    ax1.plot(energy_bin_mids, dedx_ranges, label=r'$\langle\mathrm{d}E/\mathrm{d}X\rangle$ Fit')
    # ax1.plot(energy_bin_mids, average_ranges, label='average Simulation')
    ax1.plot(energy_bin_mids, median_ranges, label='Simulation')

    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.set_ylabel('Propagated Distance / cm')
    ax1.legend()

    ax2.plot(energy_bin_mids, dedx_ranges/median_ranges, label=r'$\langle\mathrm{d}E/\mathrm{d}X\rangle$ Fit / Simulation')
    # ax2.plot(energy_bin_mids, average_ranges/median_ranges, label='average/median')
    ax2.set_ylabel('Ratio')
    ax2.set_xlabel(r'Muon Energy $E$ / MeV')
    ax2.set_xscale('log')
    ax2.legend()
    plt.subplots_adjust(hspace=.0)
    plt.setp(ax1.get_xticklabels(), visible=False)

    fig.subplots_adjust(right=0.85)
    cbar_ax = fig.add_axes([0.9, 0.15, 0.03, 0.7])
    fig.colorbar(im, cax=cbar_ax)

    # fig.tight_layout()
    fig.savefig(output_file, bbox_inches='tight', pad_inches=0.02, dpi=300)

def main(n_energy_bins=100, n_muon_per_bin=1000,
    data_file='data_ranges_prop.txt', output_file='ranges_prop.pdf'):
    args = {
        "particle_def": pp.particle.MuMinusDef(),
        "target": pp.medium.Ice(),
        "interpolate": True,
        "cuts": pp.EnergyCutSettings(1e4, 1e-2, False)
    }
    pp.RandomGenerator.get().set_seed(1234)


    energy_bin_edges_mids = np.logspace(3, 11, num = 2*n_energy_bins + 1)
    energy_bin_mids = energy_bin_edges_mids[1::2]

    range_arr = get_range_data(energy_bin_mids, n_muon_per_bin, data_file, args)

    n_range_bins = 50
    range_bin_edges_mids = np.logspace(2, 7, num = 2*n_range_bins + 1)

    plot_range_distribution(
        args,
        energy_bin_edges_mids,
        range_bin_edges_mids,
        range_arr,
        output_file)


if __name__ == '__main__':
    main()
