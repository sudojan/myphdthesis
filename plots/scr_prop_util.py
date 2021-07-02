
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib import rc
from matplotlib import lines
from scipy import stats
from tqdm import tqdm
import proposal as pp

rc('font', **{'family': 'serif',
   'serif': ['Computer Modern']})
rc('text', usetex=True)
rc('text.latex', preamble=r'\usepackage{calrsfs}')

FIG_SIZE = (5.78, 3.57)

pp.InterpolationSettings.tables_path = "tables_proposal_interpol"

def plot_dist(energy, output, args):

    cross = pp.crosssection.make_std_crosssection(**args)

    interaction = pp.make_interaction(cross, True)
    displacement = pp.make_displacement(cross, True)

    rnd = np.logspace(-8,0,200)
    e_f = interaction.energy_interaction(energy, rnd)
    dist_ef = displacement.solve_track_integral(energy, e_f)

    mean_free_path = interaction.mean_free_path(energy)
    dist_mfp = stats.expon.isf(rnd, scale=mean_free_path)

    fig = plt.figure(figsize=(6,7))
    gs = GridSpec(7, 1)
    ax1 = fig.add_subplot(gs[0:3])
    ax2 = fig.add_subplot(gs[3:6], sharex=ax1)
    ax3 = fig.add_subplot(gs[6], sharex=ax1)

    ax1.plot(rnd, e_f, label='Energy Integral')
    ax1.set_xscale('log')
    ax1.set_ylabel(r'Energy at next Interaction / MeV')
    ax1.set_yscale('log')

    ax2.plot(rnd, dist_ef, label='Tracking Integral')
    ax2.axhline(mean_free_path, c='k', ls=':')
    ax2.text(1e-8, 6000, 'Mean Free Path')
    ax2.plot(rnd, dist_mfp, ls='--',
        label=r'$\mathrm{pdf}_{\mathrm{exp}}(\mathrm{Mean Free Path})$')

    ax2.set_ylabel(r'Distance to next Interaction / $\mathrm{g}\,\mathrm{cm}^{-2}$')

    ax3.plot(rnd, dist_ef/dist_mfp, c='k',
        label=r'Tracking Integral / $\mathrm{pdf}_{\mathrm{exp}}(\mathrm{Mean Free Path})$')
    ax3.set_xlabel("Random Number")
    ax3.set_ylabel(r'Ratio')

    ax1.legend()
    ax2.legend()
    ax3.legend()
    # ax2.grid()

    plt.subplots_adjust(hspace=.1)
    plt.setp(ax1.get_xticklabels(), visible=False)
    plt.setp(ax2.get_xticklabels(), visible=False)

    plt.savefig(output, bbox_inches='tight', pad_inches=0.02)
    plt.close()


def plot_time(energy, output, args, npts=100):

    cross = pp.crosssection.make_std_crosssection(**args)

    displacement = pp.make_displacement(cross, True)
    time = pp.make_time(cross, args['particle_def'], True)
    time_approx = pp.make_time_approximate()

    e_f = np.geomspace(args['particle_def'].mass, energy, npts)
    grammage = displacement.solve_track_integral(energy, e_f)
    time_val = np.empty(npts)
    time_approx_val = np.empty(npts)
    for idx in range(npts):
        time_val[idx] = time.elapsed(
            energy, e_f[idx], grammage[idx], args["target"].mass_density)
        time_approx_val[idx] = time_approx.elapsed(
            energy, e_f[idx], grammage[idx], args["target"].mass_density)

    fig = plt.figure(figsize=FIG_SIZE)
    gs = GridSpec(4, 1)
    ax1 = fig.add_subplot(gs[:-1])
    ax2 = fig.add_subplot(gs[-1], sharex=ax1)

    ax1.plot(e_f, time_val*1e6, label=r'Time Integral')
    ax1.plot(e_f, time_approx_val*1e6, label=r'$c_0$ Approximation', ls='--')
    ax1.set_xscale('log')
    # ax1.set_yscale('log')

    ax2.plot(e_f, time_approx_val/time_val, label=r'$c_0$ Approximation / Time Integral')

    ax1.set_ylabel(r'Elapsed Time / $\mu$s')
    ax2.set_ylabel(r'Ratio')
    ax2.set_xlabel(r'Energy at the next Interaction $E_f$ / MeV')

    ax1.legend()
    ax2.legend()
    # ax2.grid()

    plt.subplots_adjust(hspace=.1)
    plt.setp(ax1.get_xticklabels(), visible=False)

    plt.savefig(output, bbox_inches='tight', pad_inches=0.02)
    plt.close()

def plot_loss(energy, output, args, npts=int(1e5), eps=1e-3):
    args = {
        "particle_def": pp.particle.MuMinusDef(),
        "target": pp.medium.Ice(),
        "interpolate": True,
        "cuts": pp.EnergyCutSettings(500, 1, False)
    }
    # TODO: there needs to be a better way to find out these hashes
    oxygen_hash = 9764564051507188568
    hydrogen_hash = 16056410032740017603
    ice_hash = 1233653330191979298

    cross = pp.crosssection.make_std_crosssection(**args)
    interaction = pp.make_interaction(cross, True)
    rates = interaction.rates(energy)
    total_rates_order = np.argsort([r.rate for r in rates])

    rates_dict = {}
    rates_dict['brems'] = {'name': 'Bremsstrahlung', 'color': 'C0'}
    rates_dict['epair'] = {'name': r'$e$ Pair Production', 'color': 'C1'}
    rates_dict['ioniz'] = {'name': 'Ionization', 'color': 'C2'}
    rates_dict['photonuclear'] = {'name': 'Photonuclear', 'color': 'C3'}

    last_hash = None
    # dummy interaction, which does not occur
    last_type = pp.particle.Interaction_Type.annihilation

    rates_arr = np.empty(npts)
    rnds = np.linspace(eps, 1-1e-12, npts)
    for idx in tqdm(range(npts)):
        loss = interaction.sample_loss(energy, rates, rnds[idx])
        rates_arr[idx] = loss.v_loss
        if (last_type.value != loss.type.value) or (last_hash != loss.comp_hash):
            if idx != 0:
                rates_dict[last_type.name][last_hash]['num'] = \
                    idx - rates_dict[last_type.name][last_hash]['start_idx']
            last_type = loss.type
            last_hash = loss.comp_hash
            rates_dict[loss.type.name][loss.comp_hash] = {'start_idx': idx}
            if loss.comp_hash == oxygen_hash:
                rates_dict[loss.type.name][loss.comp_hash]['ls'] = '--'
            elif loss.comp_hash == hydrogen_hash:
                rates_dict[loss.type.name][loss.comp_hash]['ls'] = ':'
            else:
                rates_dict[loss.type.name][loss.comp_hash]['ls'] = '-.'
    rates_dict[last_type.name][last_hash]['num'] = \
        npts - rates_dict[last_type.name][last_hash]['start_idx']


    fig = plt.figure(figsize=(7, 3.57))
    ax = fig.add_subplot()
    ax.axhline(args['cuts'].ecut/energy, c='k', alpha=0.5)
    ax.text(1.2e-4, 7e-5, r'$v_{\mathrm{cut}}=5\cdot 10^{-5}$')

    kdx = 0
    for jdx in total_rates_order:
        rate_dict = rates_dict[rates[jdx].crosssection.type.name]
        cdict = rate_dict[rates[jdx].comp_hash]
        ax.plot(
            rnds[kdx:kdx+cdict['num']],
            rates_arr[cdict['start_idx']:cdict['start_idx']+cdict['num']],
            ls=cdict['ls'], c=rate_dict['color'])
        kdx += cdict['num']

    ax.set_xlim(right=1, left=eps)
    ax.set_xlabel(r'Random Number $\xi$')
    ax.set_ylabel(r'Relative Energy Loss $v$')
    ax.set_xscale('log')
    ax.set_yscale('log')

    legend_list = []
    legend_list2 = []
    for idx in ['brems', 'epair', 'ioniz', 'photonuclear']:
        legend_list.append(lines.Line2D([0], [0], ls='-', c=rates_dict[idx]['color']))
        legend_list2.append(rates_dict[idx]['name'])
    ax.legend(legend_list, legend_list2, bbox_to_anchor=(-0.1, 1.), ncol=4, loc='lower left')

    plt.savefig(output, bbox_inches='tight', pad_inches=0.02)
    plt.close()


if __name__ == '__main__':

    args = {
        "particle_def": pp.particle.MuMinusDef(),
        "target": pp.medium.Ice(),
        "interpolate": True,
        "cuts": pp.EnergyCutSettings(500, 1, False)
    }

    # plot_dist(2e5, 'prop_util_next_int.pdf', args)
    # plot_time(1e5, 'prop_util_next_time.pdf', args)
    plot_loss(1e7, 'prop_util_stoch_loss.pdf', args, int(3e6), 1e-4)
