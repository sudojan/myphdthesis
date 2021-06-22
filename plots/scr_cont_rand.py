
import numpy as np
import matplotlib.pyplot as plt
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

    cross_list = pp.crosssection.make_std_crosssection(**args)

    collection = pp.PropagationUtilityCollection()

    collection.displacement = pp.make_displacement(cross_list, True)
    collection.interaction = pp.make_interaction(cross_list, True)
    collection.time = pp.make_time_approximate()
    collection.cont_rand = pp.make_contrand(cross_list, True)

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

    final_energies = np.empty(nmuons)
    for idx in tqdm(range(nmuons)):
        track = prop.propagate(init_state, distance)
        final_energies[idx] = track.track()[-1].energy

    return final_energies

def plot_cont_rand_loss(dat_w, dat_wo, output, energy):

    bins = np.linspace(100, 5000, 300)
    dat_wo = [np.histogram(energy - e_f, bins=bins)[0] for e_f in dat_wo]
    dat_w = [np.histogram(energy - e_f, bins=bins)[0] for e_f in dat_w]

    fig = plt.figure(figsize=FIG_SIZE)
    ax = fig.add_subplot()

    label_list = [
        r'$v_{\mathrm{cut}}=10^{-3}$',
        r'$v_{\mathrm{cut}}=10^{-4}$',
        r'$v_{\mathrm{cut}}=10^{-5}$']

    for idx in range(3):
        ax.plot(bins, np.r_[dat_wo[idx][0], dat_wo[idx]], lw=.5, c=f'C{idx+1}', drawstyle='steps',
            label=label_list[idx])
        ax.plot(bins, np.r_[dat_w[idx][0], dat_w[idx]], lw=.5, c=f'C{idx+1}', ls='--', drawstyle='steps')

    ax.set_xlabel(r'Energy Lost / MeV')
    ax.set_ylabel(r'Number of Muons')
    ax.set_yscale('log')
    ax.legend()

    plt.savefig(output, bbox_inches='tight', pad_inches=0.02)
    plt.close()

def plot_cont_rand_mu(dat_w, dat_wo, output):

    bins = np.linspace(0.98*1, 1, 300)
    dat_wo = [np.histogram(e_f/1e6, bins=bins)[0] for e_f in dat_wo]
    dat_w = [np.histogram(e_f/1e6, bins=bins)[0] for e_f in dat_w]

    fig = plt.figure(figsize=FIG_SIZE)
    ax = fig.add_subplot()

    label_list = [
        r'$v_{\mathrm{cut}}=10^{-2}$',
        r'$v_{\mathrm{cut}}=10^{-3}$',
        r'$v_{\mathrm{cut}}=10^{-4}$']

    for idx in range(3):
        ax.plot(bins, np.r_[dat_wo[idx][0], dat_wo[idx]], lw=.5, c=f'C{idx}', drawstyle='steps',
            label=label_list[idx])
        ax.plot(bins, np.r_[dat_w[idx][0], dat_w[idx]], lw=.5, c=f'C{idx}', ls='--', drawstyle='steps')

    ax.set_xlabel(r'Final Muon Energy / TeV')
    ax.set_ylabel(r'Number of Muons')
    ax.set_yscale('log')
    ax.legend()

    plt.savefig(output, bbox_inches='tight', pad_inches=0.02)
    plt.close()

def main(energy, distance, nmuons):
    args = {
        "particle_def": pp.particle.MuMinusDef(),
        "target": pp.medium.Ice(),
        "interpolate": True,
        "cuts": pp.EnergyCutSettings(1, 1, False)
    }
    pp.RandomGenerator.get().set_seed(123)

    def get_energies(vcut, do_cont_rand):
        args["cuts"] = pp.EnergyCutSettings(np.inf, vcut, do_cont_rand)
        prop = create_propagator(args)
        return prop_mu(prop, energy, distance, nmuons)

    e_5_w = get_energies(1e-5, True)
    e_5_wo = get_energies(1e-5, False)
    e_4_w = get_energies(1e-4, True)
    e_4_wo = get_energies(1e-4, False)
    e_3_w = get_energies(1e-3, True)
    e_3_wo = get_energies(1e-3, False)
    e_2_w = get_energies(1e-2, True)
    e_2_wo = get_energies(1e-2, False)


    plot_cont_rand_loss([e_3_w, e_4_w, e_5_w], [e_3_wo, e_4_wo, e_5_wo],
        'prop_cont_rand_loss.pdf', energy)
    plot_cont_rand_mu([e_2_w, e_3_w, e_4_w], [e_2_wo, e_3_wo, e_4_wo],
        'prop_cont_rand_mu.pdf')

if __name__ == '__main__':
    main(1e6, 1e3, int(1e6))
