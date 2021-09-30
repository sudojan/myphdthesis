import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import gridspec
import numpy as np
import gzip
from matplotlib import rc

rc('font', **{'family': 'serif',
   'serif': ['Computer Modern']})
rc('text', usetex=True)
rc('text.latex', preamble=r'\usepackage{calrsfs}')

FIG_SIZE = (5.78, 3.57)

SCRIPT_PATH = os.path.dirname(os.path.abspath(__file__))


def plot_rangedist(bin_edges, dist_base, dist_new, labels, output, xlabel=None, logscalex=False):
    hist_base = np.histogram(dist_base, bins=bin_edges)[0]
    hist_new = np.histogram(dist_new, bins=bin_edges)[0]

    hist_base = np.r_[hist_base[0], hist_base]
    hist_new = np.r_[hist_new[0], hist_new]

    fig = plt.figure(figsize=FIG_SIZE)
    gs = gridspec.GridSpec(3, 1)
    ax1 = fig.add_subplot(gs[:-1])
    ax2 = fig.add_subplot(gs[-1], sharex=ax1)

    ax1.plot(bin_edges,
            hist_base,
            linestyle='--',
            drawstyle='steps',
            label=labels[0],)
    ax1.plot(bin_edges,
            hist_new,
            linestyle=':',
            drawstyle='steps',
            label=labels[1],)

    ax1.set_xlim(0, bin_edges[-1])
    ax1.set_ylabel('Number of Muons')
    ax1.set_yscale('log')
    ax1.legend(loc='lower left')

    diff = hist_new - hist_base
    diff = np.sign(diff)*np.nan_to_num(np.log10(np.abs(diff)))

    ax2.plot(bin_edges,
            hist_base/hist_base,
            drawstyle='steps',
            alpha=0.8)
    ax2.plot(bin_edges,
            hist_new/hist_base,
            drawstyle='steps',
            label='{}/{}'.format(labels[1], labels[0]))
    if logscalex:
        ax1.set_xscale('log')
        ax2.set_xscale('log')
    if xlabel is not None:
        ax2.set_xlabel(xlabel)
    ax2.set_ylabel(r'Ratio')
    # ax2.set_ylabel(r'$\mathrm{sign}(\Delta) \log_{10}(|\Delta|)$')
    ax2.legend()
    ax2.set_ylim(top=1.1)
    # ax2.grid()

    plt.subplots_adjust(hspace=.1)
    plt.setp(ax1.get_xticklabels(), visible=False)

    fig.savefig(output, bbox_inches='tight', pad_inches=0.02)#, dpi=300)


def plot_secdist(loss_bins, heights_base, heights_new, labels, output):
    sec_labels = [
        r'$e$ Pair Production',
        'Bremsstrahlung',
        'Ionization',
        'Photonuclear',
        r'$\mu$ Pair Production',
        'Decay',
        'Weak Interaction']
    base_sum = np.sum(heights_base, axis=0)
    base_sum = np.r_[base_sum[0], base_sum]

    new_sum = np.sum(heights_new, axis=0)
    new_sum = np.r_[new_sum[0], new_sum]

    # don't plot the histograms with no events
    base_zeros = np.where((heights_base == 0).all(axis=1))
    new_zeros = np.where((heights_new == 0).all(axis=1))
    comb_zero = np.intersect1d(base_zeros, new_zeros)
    plot_idx = np.setdiff1d(range(len(sec_labels)), comb_zero)

    fig = plt.figure()
    gs = gridspec.GridSpec(3, 1)
    ax1 = fig.add_subplot(gs[:-1])
    ax2 = fig.add_subplot(gs[-1], sharex=ax1)

    for idx in plot_idx:
        ax1.plot([0,0], [0,0], label=sec_labels[idx], color='C{}'.format(idx))
        ax1.plot(loss_bins,
                np.r_[heights_base[idx][0], heights_base[idx]],
                linestyle='--',
                drawstyle='steps',
                color='C{}'.format(idx),
                alpha=0.7)
        ax1.plot(loss_bins,
                np.r_[heights_new[idx][0], heights_new[idx]],
                linestyle=':',
                drawstyle='steps',
                color='C{}'.format(idx))

    ax1.set_yscale('log')
    ax1.set_xscale('log')
    ax1.set_ylabel('Number of Secondaries')
    ax1.legend(loc='lower left')

    ax2.plot(loss_bins, new_sum/base_sum, drawstyle='steps', label='{}/{}'.format(labels[1], labels[0]))
    ax2.set_xscale('log')
    ax2.set_xlabel(r'Secondary Energy / MeV')
    ax2.set_ylabel('Ratio of Sum')
    ax2.legend()

    plt.subplots_adjust(hspace=.1)
    plt.setp(ax1.get_xticklabels(), visible=False)

    plt.savefig(output, bbox_inches='tight', pad_inches=0.02, dpi=300)
    # plt.show()


def new_range_dist():
    labels = ['KKP95', 'SSR19']

    max_propagation_len = 1e9 # cm
    statistics = int(1e6)
    energy = 1e7
    muon_energies = np.ones(statistics)*energy

    filename = os.path.join(SCRIPT_PATH, 'data_range_base.txt.gz')
    with gzip.open(filename, 'r') as file:
        ranges_base = np.genfromtxt(file)

    filename = os.path.join(SCRIPT_PATH, 'data_range_new.txt.gz')
    with gzip.open(filename, 'r') as file:
        ranges_new = np.genfromtxt(file)

    nbins = 20
    rmin = np.min([ranges_base, ranges_new]) - 0.1
    rmax = np.max([ranges_base, ranges_new]) + 0.1
    bin_edges = np.linspace(rmin, rmax, nbins+1)
    output = os.path.join(SCRIPT_PATH, 'plot_range_dist.pdf')
    plot_rangedist(bin_edges/1e5, ranges_base/1e5, ranges_new/1e5, labels, output, xlabel=r'Propagated Range / km')


def new_energy_dist():
    labels = ['KKP95', 'SSR19']
    # energy dist after range
    max_propagation_len = 1e5 # cm
    statistics = int(1e7)
    energy = 1e7
    muon_energies = np.ones(statistics)*energy

    filename = os.path.join(SCRIPT_PATH, 'data_energy_base.txt.gz')
    with gzip.open(filename, 'r') as file:
        dist_base = np.genfromtxt(file) / 1e6

    filename = os.path.join(SCRIPT_PATH, 'data_energy_new.txt.gz')
    with gzip.open(filename, 'r') as file:
        dist_new = np.genfromtxt(file) / 1e6

    nbins = 120
    rmin = np.min([dist_base, dist_new]) - 0.1
    rmax = np.max([dist_base, dist_new]) + 0.1
    # bin_edges = np.logspace(np.log10(rmin), np.log10(rmax), nbins+1)
    bin_edges = np.linspace(0, 10, nbins+1)
    output = os.path.join(SCRIPT_PATH, 'plot_energy_dist2.pdf')
    plot_rangedist(bin_edges, dist_base, dist_new, labels, output, logscalex=False, xlabel='Final Muon Energy / TeV')


def new_sec_dist():
    labels = ['KKP95', 'SSR19']
    max_propagation_len = 1e4 # cm
    statistics = int(1e7)
    energy = 1e7
    muon_energies = np.ones(statistics)*energy
    nbins = 30
    loss_bin_edges = np.logspace(np.log10(500), np.log10(energy), nbins+1)

    filename = os.path.join(SCRIPT_PATH, 'data_dNdx_base.txt')
    heights_base = np.genfromtxt(filename)

    filename = os.path.join(SCRIPT_PATH, 'data_dNdx_new.txt')
    heights_new = np.genfromtxt(filename)

    output = os.path.join(SCRIPT_PATH, 'plot_sec_dist.pdf')
    plot_secdist(loss_bin_edges, heights_base, heights_new, labels, output)

def new_sec_dist_mu():
    labels = [r'no $\mu$ pair', r'with $\mu$ pair']
    max_propagation_len = 1e4 # cm
    statistics = int(1e7)
    energy = 1e7
    muon_energies = np.ones(statistics)*energy
    nbins = 30
    loss_bin_edges = np.logspace(np.log10(500), np.log10(energy), nbins+1)

    filename = os.path.join(SCRIPT_PATH, 'data_dNdx_nomu.txt')
    heights_base = np.genfromtxt(filename)

    filename = os.path.join(SCRIPT_PATH, 'data_dNdx_mu.txt')
    heights_new = np.genfromtxt(filename)

    output = os.path.join(SCRIPT_PATH, 'plot_sec_dist_mu.pdf')
    plot_secdist(loss_bin_edges, heights_base, heights_new, labels, output)



if __name__ == "__main__":
    # new_range_dist()
    # new_sec_dist()
    # new_sec_dist_mu()
    new_energy_dist()
