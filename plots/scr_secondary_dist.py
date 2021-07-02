
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib import rc
from tqdm import tqdm
import proposal as pp

rc('font', **{'family': 'serif',
   'serif': ['Computer Modern']})
rc('text', usetex=True)
rc('text.latex', preamble=r'\usepackage{calrsfs}')

FIG_SIZE = (5.78, 3.57)

pp.InterpolationSettings.tables_path = "tables_proposal_interpol"

def create_propagator(args):
    # try:
    cross_list = pp.crosssection.make_std_crosssection(**args)
    cross_list.append(
        pp.crosssection.make_crosssection(
            pp.parametrization.mupairproduction.KelnerKokoulinPetrukhin(),
            **args))
    # except TypeError:
    #     shadow = pp.parametrization.photonuclear.ShadowButkevichMikheyev()
    #     param_list = [
    #         pp.parametrization.ionization.BetheBlochRossi(args['cuts']),
    #         pp.parametrization.pairproduction.KelnerKokoulinPetrukhin(),
    #         pp.parametrization.bremsstrahlung.KelnerKokoulinPetrukhin(),
    #         pp.parametrization.photonuclear.AbramowiczLevinLevyMaor97(shadow),
    #     ]
    #     cross_list = [pp.crosssection.make_crosssection(param, **args) for param in param_list]

    collection = pp.PropagationUtilityCollection()

    collection.displacement = pp.make_displacement(cross_list, True)
    collection.interaction = pp.make_interaction(cross_list, True)
    collection.decay = pp.make_decay(cross_list, args["particle_def"], True)
    # collection.time = pp.make_time(cross_list, args["particle_def"], True)
    collection.time = pp.make_time_approximate()

    utility = pp.PropagationUtility(collection = collection)
    
    detector = pp.geometry.Sphere(pp.Cartesian3D(0,0,0), 1e20)
    density_distr = pp.density_distribution.density_homogeneous(args["target"].mass_density)
    
    return pp.Propagator(args["particle_def"], [(detector, utility, density_distr)])

def plot_secondary_dist(args, output, energy, nmuons=int(1e5), max_dist=1e5):

    prop = create_propagator(args)

    init_state = pp.particle.ParticleState()
    init_state.energy = energy
    init_state.position = pp.Cartesian3D(0, 0, 0)
    init_state.direction = pp.Cartesian3D(0, 0, 1)
    init_state.time = 0
    init_state.propagated_distance = 0

    brems = []
    epair = []
    ioniz = []
    photo = []
    mupai = []
    conti = []
    decay = []

    for idx in tqdm(range(nmuons)):
        track = prop.propagate(init_state, max_dist)
        brems.extend(track.stochastic_losses(pp.particle.Interaction_Type.brems))
        epair.extend(track.stochastic_losses(pp.particle.Interaction_Type.epair))
        ioniz.extend(track.stochastic_losses(pp.particle.Interaction_Type.ioniz))
        photo.extend(track.stochastic_losses(pp.particle.Interaction_Type.photonuclear))
        mupai.extend(track.stochastic_losses(pp.particle.Interaction_Type.mupair))
        conti.extend(track.continuous_losses())
        decay.extend(track.stochastic_losses(pp.particle.Interaction_Type.decay))

    bins = np.geomspace(1, energy, 50)
    brems_h = np.histogram([idx.energy for idx in brems], bins=bins)[0]
    epair_h = np.histogram([idx.energy for idx in epair], bins=bins)[0]
    ioniz_h = np.histogram([idx.energy for idx in ioniz], bins=bins)[0]
    photo_h = np.histogram([idx.energy for idx in photo], bins=bins)[0]
    mupai_h = np.histogram([idx.energy for idx in mupai], bins=bins)[0]
    conti_h = np.histogram([-idx.energy for idx in conti], bins=bins)[0]

    fig = plt.figure(figsize=(7, 4))
    ax = fig.add_subplot()
    
    ax.plot(bins, np.r_[brems_h[0], brems_h], drawstyle='steps', label=r'Bremsstrahlung')
    ax.plot(bins, np.r_[epair_h[0], epair_h], drawstyle='steps', label=r'$e$ Pair Production')
    ax.plot(bins, np.r_[ioniz_h[0], ioniz_h], drawstyle='steps', label=r'Ionization')
    ax.plot(bins, np.r_[photo_h[0], photo_h], drawstyle='steps', label=r'Photonuclear')
    ax.plot(bins, np.r_[mupai_h[0], mupai_h], drawstyle='steps', label=r'$\mu$ Pair Production')
    ax.plot(bins, np.r_[conti_h[0], conti_h], drawstyle='steps', label=r'Continuous Loss')

    if len(decay) > 0:
        print(len(decay))
        decay_h = np.histogram([idx.parent_particle_energy for idx in decay], bins=bins)[0]
        ax.plot(bins, np.r_[decay_h[0], decay_h], drawstyle='steps', label=r'Decay')

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel(r'Energy Loss / MeV')
    ax.set_ylabel('Number of Stochastic Losses')
    ax.legend()

    plt.savefig(output, bbox_inches='tight', pad_inches=0.02)
    plt.close()

if __name__ == '__main__':

    args = {
        "particle_def": pp.particle.MuMinusDef(),
        "target": pp.medium.Ice(),
        "interpolate": True,
        "cuts": pp.EnergyCutSettings(500, 1, False)
    }
    pp.RandomGenerator.get().set_seed(1234)
    plot_secondary_dist(args, 'prop_secondary_dist.pdf', energy=1e7, nmuons=int(1e5))

