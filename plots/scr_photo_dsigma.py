import proposal as pp
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from matplotlib import rc
from matplotlib.gridspec import GridSpec

rc('font', **{'family': 'serif',
   'serif': ['Computer Modern']})
rc('text', usetex=True)
rc('text.latex', preamble=r'\usepackage{calrsfs}')

FIG_SIZE = (5.78, 3.57)


def plot_photo_dsigma(energy, npts, output='photo_dsigma.pdf'):
    mudef = pp.particle.MuMinusDef()
    medium = pp.medium.StandardRock()
    comp = medium.components[0]
    hard = True
    shadow = pp.parametrization.photonuclear.ShadowButkevichMikheyev()

    params_vmd = [
        pp.parametrization.photonuclear.BezrukovBugaev(hard),
        pp.parametrization.photonuclear.Kokoulin(hard),
        pp.parametrization.photonuclear.Zeus(hard),
        pp.parametrization.photonuclear.Rhode(hard),
    ]

    labels_vmd = [
        'BezrukovBugaev',
        'Kokoulin',
        'Zeus',
        'Rhode',
    ]

    params_regge = [
        pp.parametrization.photonuclear.AbramowiczLevinLevyMaor91(shadow),
        pp.parametrization.photonuclear.AbramowiczLevinLevyMaor97(shadow),
        pp.parametrization.photonuclear.ButkevichMikheyev(shadow),
        pp.parametrization.photonuclear.AbtFT(shadow),
    ]

    labels_regge = [
        'ALLM91',
        'ALLM97',
        'ButkevichMikheyev',
        'AbtFT',
    ]

    limits = params_vmd[0].kinematic_limits(
        mudef,
        comp,
        energy)
    v_arr = np.geomspace(limits.v_min, limits.v_max, npts)

    dsigma_vmd_arr = np.empty((len(params_vmd), npts))
    for idx in range(len(params_vmd)):
        for jdx in range(npts):
            dsigma_vmd_arr[idx, jdx] = params_vmd[idx].differential_crosssection(
                mudef, comp, energy, v_arr[jdx])

    dsigma_regge_arr = np.empty((len(params_regge), npts))
    for idx in range(len(params_regge)):
        for jdx in range(npts):
            dsigma_regge_arr[idx, jdx] = params_regge[idx].differential_crosssection(
                mudef, comp, energy, v_arr[jdx])

    # param_brems = pp.parametrization.bremsstrahlung.KelnerKokoulinPetrukhin(False)
    # dsigma_brems = [param_brems.differential_crosssection(
    #             mudef, comp, energy, v_arr[jdx]) for jdx in range(npts)]

    param_soft = pp.parametrization.photonuclear.BezrukovBugaev(False)
    dsigma_soft = np.array([param_soft.differential_crosssection(
                mudef, comp, energy, v_arr[jdx]) for jdx in range(npts)])
    param_dutta = pp.parametrization.photonuclear.AbramowiczLevinLevyMaor97(
        pp.parametrization.photonuclear.ShadowDuttaRenoSarcevicSeckel())
    dsigma_dutta = np.array([param_dutta.differential_crosssection(
                mudef, comp, energy, v_arr[jdx]) for jdx in range(npts)])

    fig = plt.figure()
    gs = GridSpec(3, 1)
    ax1 = fig.add_subplot(gs[:-1])
    ax2 = fig.add_subplot(gs[-1], sharex=ax1)

    for idx in range(len(params_regge)):
        ax1.plot(v_arr, dsigma_regge_arr[idx] * v_arr, label=labels_regge[idx])

    for idx in range(len(params_vmd)):
        ax1.plot(v_arr, dsigma_vmd_arr[idx] * v_arr, label=labels_vmd[idx], ls='--')

    # ax1.plot(v_arr, dsigma_brems * v_arr, label='Bremsstrahlung', ls=':')

    ax1.set_xscale('log')
    ax1.set_yscale('log')

    # ax1.set_xlabel(r'Relative Energy Loss $v$')
    ax1.set_ylabel(r'Differential Cross Section $v \frac{\mathrm{d}\sigma}{\mathrm{d}v}  \,/\, \mathrm{cm}^2$')

    ax1.set_xlim(right=1)
    ax1.set_ylim(bottom=1e-8)

    ax1.legend(ncol=2)
    ax1.grid()

    ax2.plot(v_arr, dsigma_dutta/dsigma_regge_arr[1], c='C1', label='ALLM97: Dutta/Butkevich')
    ax2.plot(v_arr, dsigma_soft/dsigma_vmd_arr[0], c='C4', label='BezrukovBugaev: soft/(soft+hard)')

    ax2.set_xscale('log')
    ax2.set_xlabel(r'Relative Energy Loss $v$')
    ax2.set_ylabel(r'Ratio')

    ax2.legend()

    plt.subplots_adjust(hspace=.1)
    plt.setp(ax1.get_xticklabels(), visible=False)

    plt.savefig(output, bbox_inches='tight', pad_inches=0.02)
    plt.close()

def plot_photo_dedx(energy_arr, output='photo_dedx.pdf'):
    mudef = pp.particle.MuMinusDef()
    medium = pp.medium.StandardRock()
    comp = medium.components[0]
    cuts = pp.EnergyCutSettings(np.inf, 1)
    hard = True
    shadow = pp.parametrization.photonuclear.ShadowButkevichMikheyev()
    shadow_dutta = pp.parametrization.photonuclear.ShadowDuttaRenoSarcevicSeckel()

    params_vmd = [
        pp.parametrization.photonuclear.BezrukovBugaev(hard),
        pp.parametrization.photonuclear.Kokoulin(hard),
        pp.parametrization.photonuclear.Zeus(hard),
        pp.parametrization.photonuclear.Rhode(hard),
    ]

    labels_vmd = [
        'BezrukovBugaev',
        'Kokoulin',
        'Zeus',
        'Rhode',
    ]

    params_regge = [
        pp.parametrization.photonuclear.AbramowiczLevinLevyMaor91(shadow),
        pp.parametrization.photonuclear.AbramowiczLevinLevyMaor97(shadow),
        pp.parametrization.photonuclear.ButkevichMikheyev(shadow),
        pp.parametrization.photonuclear.AbtFT(shadow),
    ]

    labels_regge = [
        'ALLM91',
        'ALLM97',
        'ButkevichMikheyev',
        'AbtFT',
    ]

    dedx_vmd = np.empty((len(params_vmd), len(energy_arr)))
    for idx in range(len(params_vmd)):
        xsection = pp.crosssection.make_crosssection(
            params_vmd[idx], mudef, medium, cuts, False)
        for jdx in range(len(energy_arr)):
            dedx_vmd[idx,jdx] = xsection.calculate_dEdx(energy_arr[jdx])

    dedx_regge = np.empty((len(params_regge), len(energy_arr)))
    for idx in tqdm(range(len(params_regge))):
        xsection = pp.crosssection.make_crosssection(
            params_regge[idx], mudef, medium, cuts, False)
        for jdx in range(len(energy_arr)):
            dedx_regge[idx,jdx] = xsection.calculate_dEdx(energy_arr[jdx])


    xsection_soft = pp.crosssection.make_crosssection(
        pp.parametrization.photonuclear.BezrukovBugaev(False), mudef, medium, cuts, False)
    dedx_soft = np.array([xsection_soft.calculate_dEdx(e_i) for e_i in energy_arr])

    xsection_dutta = pp.crosssection.make_crosssection(
        pp.parametrization.photonuclear.AbramowiczLevinLevyMaor97(
            pp.parametrization.photonuclear.ShadowDuttaRenoSarcevicSeckel()),
        mudef, medium, cuts, False)
    dedx_dutta = np.array([xsection_dutta.calculate_dEdx(e_i) for e_i in energy_arr])

    fig = plt.figure()
    gs = GridSpec(3, 1)
    ax1 = fig.add_subplot(gs[:-1])
    ax2 = fig.add_subplot(gs[-1], sharex=ax1)

    for idx in range(len(params_regge)):
        ax1.plot(energy_arr, dedx_regge[idx] / energy_arr, label=labels_regge[idx])

    for idx in range(len(params_vmd)):
        ax1.plot(energy_arr, dedx_vmd[idx] / energy_arr, label=labels_vmd[idx], ls='--')

    ax1.set_xscale('log')
    ax1.set_yscale('log')

    # ax1.set_xlabel(r'Muon Energy / MeV')
    ax1.set_ylabel(r'Average Energy Loss $\frac{1}{E} \frac{\mathrm{d}E}{\mathrm{d}X} \,/\, (\mathrm{g}^{-1} \mathrm{cm}^2)$')

    ax1.set_xlim(right=energy_arr[-1])

    ax1.legend(ncol=2)
    ax1.grid()

    ax2.plot(energy_arr, dedx_dutta/dedx_regge[1], c='C1', label='ALLM97: Dutta/Butkevich')
    ax2.plot(energy_arr, dedx_soft/dedx_vmd[0], c='C4', label='BezrukovBugaev: soft/(soft+hard)')

    ax2.set_xscale('log')
    ax2.set_xlabel(r'Muon Energy $E$ / MeV')
    ax2.set_ylabel(r'Ratio')

    ax2.legend()

    plt.subplots_adjust(hspace=.1)
    plt.setp(ax1.get_xticklabels(), visible=False)

    plt.savefig(output, bbox_inches='tight', pad_inches=0.02)
    plt.close()

if __name__ == '__main__':
    plot_photo_dsigma(1e6, 200)
    energy_arr = np.geomspace(500, 1e12, 200)
    plot_photo_dedx(energy_arr)
