
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib import rc
from tqdm import tqdm
import proposal as pp

rc('font', **{'family': 'serif',
   'serif': ['Computer Modern']})
rc('text', usetex=True)
rc('text.latex', preamble=r'\usepackage{calrsfs}')

FIG_SIZE = (5.78, 3.57)

pp.InterpolationSettings.tables_path = "tables_proposal_interpol"

def create_propagator(args, cross_list, scattering=None, decay=False):

    collection = pp.PropagationUtilityCollection()

    collection.displacement = pp.make_displacement(cross_list, True)
    collection.interaction = pp.make_interaction(cross_list, True)
    if decay:
        collection.decay = pp.make_decay(cross_list, args["particle_def"], True)
    collection.time = pp.make_time_approximate()

    if scattering is not None:
        collection.scattering = scattering

    utility = pp.PropagationUtility(collection = collection)

    detector = pp.geometry.Sphere(pp.Cartesian3D(0,0,0), 1e20)
    density_distr = pp.density_distribution.density_homogeneous(args["target"].mass_density)

    return pp.Propagator(args["particle_def"], [(detector, utility, density_distr)])

def prop_mu(prop, energy, distance, nmuons):

    init_state = pp.particle.ParticleState()
    init_state.energy = energy
    init_state.position = pp.Cartesian3D(0, 0, 0)
    init_state.direction = pp.Cartesian3D(0, 0, 1)
    init_state.time = 0
    init_state.propagated_distance = 0

    final_radius = np.empty(nmuons)
    for idx in tqdm(range(nmuons)):
        track = prop.propagate(init_state, distance)
        tmp = track.track()[-1].position
        final_radius[idx] = np.sqrt(tmp.x**2 + tmp.y**2)

    return final_radius

def plot_deflection(datas, label_list, output):

    bins = np.linspace(0, 2e3, 100)
    dat_h = [np.histogram(idx, bins=bins)[0] for idx in datas]

    fig = plt.figure(figsize=FIG_SIZE)
    ax = fig.add_subplot()

    for idx in range(len(datas)):
        ax.plot(bins, np.r_[dat_h[idx][0], dat_h[idx]], drawstyle='steps',
            label=label_list[idx])

    ax.set_xlabel(r'Deflection / cm')
    ax.set_ylabel(r'Number of Muons')
    ax.set_yscale('log')
    ax.legend()

    plt.savefig(output, bbox_inches='tight', pad_inches=0.02)
    plt.close()


def calc_deflections(energy, distance, nmuons):
    args = {
        "particle_def": pp.particle.MuMinusDef(),
        "target": pp.medium.Ice(),
        "interpolate": True,
        "cuts": pp.EnergyCutSettings(500, 1, False)
    }
    pp.RandomGenerator.get().set_seed(123)

    cross_list = pp.crosssection.make_std_crosssection(**args)

    def make_scat(name):
        if name == "HighlandIntegral":
            return pp.make_multiple_scattering("HighlandIntegral",
                args["particle_def"], args["target"], cross_list, args["interpolate"])
        else:
            return pp.make_multiple_scattering(name, args["particle_def"], args["target"])

    stoch_defl = pp.make_default_stochastic_deflection(
        [pp.particle.Interaction_Type.brems,
         pp.particle.Interaction_Type.epair,
         pp.particle.Interaction_Type.ioniz,
         pp.particle.Interaction_Type.photonuclear],
        args["particle_def"], args["target"])

    scat_list = [
        None,
        pp.scattering.Scattering(make_scat("Moliere")),
        pp.scattering.Scattering(make_scat("Highland")),
        pp.scattering.Scattering(make_scat("HighlandIntegral")),
        pp.scattering.Scattering(stoch_defl),
        pp.scattering.Scattering(make_scat("HighlandIntegral"), stoch_defl),
    ]

    def get_deflections(scattering):
        prop = create_propagator(args, cross_list, scattering)
        return prop_mu(prop, energy, distance, nmuons)

    datas = [get_deflections(idx) for idx in scat_list]
    label_list = [
        # 'No Scattering',
        'Moliere',
        'Highland',
        'HighlandIntegral',
        'Stochastic Deflection',
        'Cont. + Stoch. Deflection']

    plot_deflection(datas[1:], label_list,
        'prop_scat.pdf')

def plot_track(energy, output):
    args = {
        "particle_def": pp.particle.MuMinusDef(),
        "target": pp.medium.Ice(),
        "interpolate": True,
        "cuts": pp.EnergyCutSettings(500, 1, False)
    }
    pp.RandomGenerator.get().set_seed(345)

    cross_list = pp.crosssection.make_std_crosssection(**args)
    cross_list.append(
        pp.crosssection.make_crosssection(
            pp.parametrization.mupairproduction.KelnerKokoulinPetrukhin(),
            **args))

    scat_multi = pp.make_multiple_scattering(
        "Highland", args["particle_def"], args["target"])

    stoch_defl = pp.make_default_stochastic_deflection(
        [pp.particle.Interaction_Type.brems,
         pp.particle.Interaction_Type.epair,
         pp.particle.Interaction_Type.ioniz,
         pp.particle.Interaction_Type.photonuclear],
        args["particle_def"], args["target"])

    prop = create_propagator(args, cross_list,
        pp.scattering.Scattering(scat_multi, stoch_defl),
        decay=True)

    init_state = pp.particle.ParticleState()
    init_state.energy = energy
    init_state.position = pp.Cartesian3D(0, 0, 0)
    init_state.direction = pp.Cartesian3D(0, 0, 1)
    init_state.time = 0
    init_state.propagated_distance = 0

    track = prop.propagate(init_state)
    decay = track.track()[-1] # dirty hack for decay

    stochastics = track.stochastic_losses()
    stochastics.append(decay)
    e_arr = np.empty(len(stochastics))
    t_arr = np.empty(len(stochastics))
    x_arr = np.empty(len(stochastics))
    d_arr = np.empty(len(stochastics))
    for idx in range(len(stochastics)):
        e_arr[idx] = stochastics[idx].energy
        x_arr[idx] = stochastics[idx].position.x
        d_arr[idx] = stochastics[idx].propagated_distance
        t_arr[idx] = stochastics[idx].type

    dist_arr = np.linspace(0, track.track()[-1].propagated_distance, 200)
    mu_e = np.array([track.get_state_for_distance(idx).energy for idx in dist_arr])

    fig = plt.figure(figsize=FIG_SIZE)
    gs = gridspec.GridSpec(2, 1)
    ax1 = fig.add_subplot(gs[:-1])
    ax2 = fig.add_subplot(gs[-1], sharex=ax1)

    ax1.plot(dist_arr/1e5, mu_e, label='Muon Energy')
    ax2.plot(d_arr/1e5, x_arr/1e2, label='Muon Track')

    interactions = [
        pp.particle.Interaction_Type.brems,
        pp.particle.Interaction_Type.epair,
        pp.particle.Interaction_Type.ioniz,
        pp.particle.Interaction_Type.photonuclear,
        0 # dirty hack for decay
    ]

    for idx, action in enumerate(interactions):
        mask = t_arr == int(action)
        ax1.scatter(d_arr[mask]/1e5, e_arr[mask], s=10, c=f'C{idx+1}')
        ax2.scatter(d_arr[mask]/1e5, x_arr[mask]/1e2, alpha=0.3, s=np.sqrt(e_arr[mask]), c=f'C{idx+1}')

    label_list = [
        'Bremsstrahlung',
        'Pair Production',
        'Ionization',
        'Photonuclear',
        'Decay',
    ]
    tmp = []
    for idx, label in enumerate(label_list):
        tmp.append(ax1.scatter([],[], label=label, c=f'C{idx+1}'))
    ax1.legend(bbox_to_anchor=(-0.1, 1.), ncol=3, loc='lower left')

    ax1.set_ylabel('Energy / MeV')
    ax1.set_yscale('log')
    # ax1.legend(bbox_to_anchor=(-0.1, 1.), ncol=3, loc='lower left')

    ax2.set_xlabel('Propagated Distance / km')
    ax2.set_ylabel(r'$x$ Coordinate / m')
    ax2.set_ylim(-2, 5)

    plt.subplots_adjust(hspace=.1)
    plt.setp(ax1.get_xticklabels(), visible=False)
    fig.savefig(output, bbox_inches='tight', pad_inches=0.02)
    plt.close()


if __name__ == '__main__':
    # calc_deflections(1e6, 1e5, int(1e6))
    plot_track(1e7, "prop_track.pdf")
