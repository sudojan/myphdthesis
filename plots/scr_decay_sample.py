import numpy as np
import proposal as pp
import matplotlib.pyplot as plt
from matplotlib import gridspec
from tqdm import tqdm
from scipy import constants
from matplotlib import rc

rc('font', **{'family': 'serif',
   'serif': ['Computer Modern']})
rc('text', usetex=True)
rc('text.latex', preamble=r'\usepackage{calrsfs}')

FIG_SIZE = (5.78, 3.57)

def evaluate(particle, products):
    G_F = 1.1663787*1e-2  # MeV
    # G_F = constants.value(u'Fermi coupling constant') * 1e1

    muon = particle
    electron = products[0]
    numu = products[1]
    nuebar = products[2]

    p1 = muon.energy * nuebar.energy - (muon.momentum * muon.direction) * (nuebar.momentum * nuebar.direction)
    p2 = electron.energy * numu.energy - (electron.momentum * electron.direction) * (numu.momentum * numu.direction)

    return 64 * G_F**2 * p1 * p2

def create_decayer(products, decay_channel, matrix_evaluate=None):
    if decay_channel == 'LahiriPal':
        decayer = pp.decay.LeptonicDecayChannel(*products)
    elif decay_channel == 'PDG':
        decayer = pp.decay.LeptonicDecayChannelApprox(*products)
    elif decay_channel == 'ManyBody':
        if matrix_evaluate is not None:
            decayer = pp.decay.ManyBodyPhaseSpace(products, matrix_evaluate)
        else:
            decayer = pp.decay.ManyBodyPhaseSpace(products)
    else:
        raise NameError('unknown decay channel: {}'.format(decay_channel))
    return decayer

def calc_spectrum(particle_def,
                  products,
                  statistics,
                  decay_channel='ManyBody',
                  matrix_evaluate=None):
    decayer = create_decayer(products=products,
                             decay_channel=decay_channel,
                             matrix_evaluate=matrix_evaluate)

    init_state = pp.particle.ParticleState()
    init_state.type = particle_def.particle_type
    init_state.position = pp.Cartesian3D(0, 0, 0)
    init_state.direction = pp.Cartesian3D(1, 0, 0)
    init_state.energy = particle_def.mass

    decay_energies = np.empty((statistics, len(products)))
    for idx in tqdm(range(statistics)):

        decay_products = decayer.decay(particle_def, init_state)
        decay_energies[idx] = [prod_i.energy for prod_i in decay_products]

    return decay_energies

def plot_spectrum(particle_def, products, statistics):
    decay_channel_names = ['PDG', 'LahiriPal', 'ManyBody']
    v_max = (particle_def.mass**2 + products[0].mass**2) / (2*particle_def.mass**2)
    bins = np.linspace(0, v_max, 100)
    bincenters = 0.5*(bins[1:]+bins[:-1])
    spec_hist = np.empty((len(decay_channel_names),
                          len(products),
                          len(bincenters)))

    for idx in range(len(decay_channel_names)):

        product_energies = calc_spectrum(particle_def,
                                        products,
                                        statistics,
                                        decay_channel_names[idx],
                                        evaluate) / (particle_def.mass)

        for jdx in range(len(products)):
            spec_hist[idx, jdx] = np.histogram(product_energies[:,jdx], bins=bins)[0]

    for idx in range(len(products)):
        fig = plt.figure()
        gs = gridspec.GridSpec(2, 1, height_ratios=[2, 1])
        ax1 = fig.add_subplot(gs[0])
        ax2 = fig.add_subplot(gs[1], sharex=ax1)
        # ax1.set_title('{} decay spectrum for {}'.format(particle_def.name, products[idx].name))

        ax1.plot(bincenters, spec_hist[0, idx],
                drawstyle='steps-mid', label=decay_channel_names[0])
        ax2.plot([0,0.5], [1.0, 1.0], lw=0.5)

        for jdx in range(1, len(decay_channel_names)):
            ax1.plot(bincenters, spec_hist[jdx, idx],
                    drawstyle='steps-mid', label=decay_channel_names[jdx])


            diff = spec_hist[jdx, idx] / spec_hist[0, idx]
            diff[np.isnan(diff)] = 1.0

            ax2.plot(bincenters, diff, drawstyle="steps-mid", lw=1.0, label='{}/{}'.format(
                        decay_channel_names[jdx], decay_channel_names[0]))

        ax1.set_ylabel(r'Number of Decay Products')
        ax1.legend(loc='upper left')

        ax2.set_xlabel(r'Decay Product Energy / Primary Particle Mass')
        ax2.set_ylabel(r'Ratio')
        ax2.legend()

        plt.subplots_adjust(hspace=.1)
        plt.setp(ax1.get_xticklabels(), visible=False)

        fig.savefig(
            "decay_{}_spectrum_of_{}_decay.pdf".format(products[idx].name, particle_def.name),
            bbox_inches='tight', pad_inches=0.02)


    fig = plt.figure()
    gs = gridspec.GridSpec(3, 1)#, height_ratios=[3, 1])
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1], sharex=ax1)
    ax3 = fig.add_subplot(gs[2], sharex=ax1)
    axes = [ax1, ax2, ax3]
    lstyle = ['-', '--', ':']
    for idx in range(len(decay_channel_names)):
        # ax.set_title('{} decay spectrum for {}'.format(particle_def.name, decay_channel_names[idx]))

        for jdx in range(len(products)):
            axes[idx].plot(bins, np.r_[spec_hist[idx, jdx][0], spec_hist[idx, jdx]],
                drawstyle='steps', label=products[jdx].name, ls=lstyle[jdx])

        axes[idx].text(0.05, np.max(spec_hist)*0.75, decay_channel_names[idx], c='grey', alpha=0.5)

    axes[0].set_xlim(0, max(bins))
    axes[0].legend(bbox_to_anchor=(0.1, 1.), ncol=4, loc='lower left')
    axes[-1].set_xlabel(r'Decay Product Energy / Primary Particle Mass')
    axes[1].set_ylabel(r'Number of Decay Products')

    plt.subplots_adjust(hspace=.1)
    plt.setp(ax1.get_xticklabels(), visible=False)
    plt.setp(ax2.get_xticklabels(), visible=False)

    fig.savefig(
        'decay_{}_decay_spectrum.pdf'.format(particle_def.name),
        bbox_inches='tight', pad_inches=0.02)


if __name__ == '__main__':
    pp.RandomGenerator.get().set_seed(124)
    mu_def = pp.particle.MuMinusDef()
    tau_def = pp.particle.TauMinusDef()
    products = [pp.particle.EMinusDef(),
                pp.particle.NuMuDef(),
                pp.particle.NuEBarDef()]
    tau_mu_products = [pp.particle.MuMinusDef(),
                        pp.particle.NuTauDef(),
                        pp.particle.NuMuBarDef()]
    hadronic_products = [pp.particle.Pi0Def(),
                        pp.particle.PiMinusDef(),
                        pp.particle.NuTauDef()]

    # plot_spectrum(mu_def, products, int(1e7))
    plot_spectrum(tau_def, tau_mu_products, int(1e7))


