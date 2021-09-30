import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc

rc('font', **{'family': 'serif',
   'serif': ['Computer Modern']})
rc('text', usetex=True)
rc('text.latex', preamble=r'\usepackage{calrsfs}')

FIG_SIZE = (5.78, 3.57)

nbins=40
bin_mids_edges = np.linspace(2,6,2*nbins+1)
bin_edges = bin_mids_edges[::2]
bin_mids = bin_mids_edges[1::2]

low = 10
up = -10
bin_edges_crop = bin_edges[low:up]
bin_mids_crop = bin_mids[low:up]

def do_fit(x_arr, y_arr):
    fit_params = np.polyfit(x_arr, y_arr, 1)
    poly_func = np.poly1d(fit_params)
    x_arr_fit = np.linspace(min(x_arr), max(x_arr), 20)
    y_arr_fit = poly_func(x_arr_fit)
    return x_arr_fit, y_arr_fit, fit_params

def compare_6_and_10_years():

    # data taken from fig 1 in 1607.08006 for 6 years of IceCube
    measured_days_6a = 2060
    ebins6, nEvents6 = np.genfromtxt('atmo_nu_6years.csv', delimiter=',', skip_header=1, skip_footer=5, unpack=True)
    # data taken from fig 3 in PoS(ICRC2019)1017 for 10 years of IceCube
    # lifetime/sec taken from Tab 1
    measured_days_10a = 259620998/60/60/24
    ebins10, nEvents10 = np.genfromtxt('atmo_nu_10years.csv', delimiter=',', skip_header=1, skip_footer=1, unpack=True)

    np.testing.assert_allclose(bin_edges[:-1], ebins6, rtol=1e-2, atol=0)
    np.testing.assert_allclose(bin_mids, ebins10, rtol=1e-2, atol=0)

    nEvents6_crop = nEvents6[low:up]
    nEvents10_crop = nEvents10[low:up]

    xfit6, yfit6, params6 = do_fit(bin_mids_crop, nEvents6_crop)
    xfit10, yfit10, params10 = do_fit(bin_mids_crop, nEvents10_crop)

    fig = plt.figure(figsize=FIG_SIZE)
    ax = fig.add_subplot(111)

    ax.plot(10**bin_edges, 10**np.r_[nEvents6[0], nEvents6],
            drawstyle='steps', label='6a atmo conv')
    ax.plot(10**bin_edges, 10**np.r_[nEvents10[0], nEvents10],
            drawstyle='steps', label='10a truncE')

    ax.plot(10**bin_mids_crop, 10**nEvents6_crop,
            linestyle='', color='k', marker='.',
            label='6a fit points {}'.format(int(sum(10**nEvents6_crop))))
    ax.plot(10**bin_mids_crop, 10**nEvents10_crop,
            linestyle='', color='k', marker='.',
            label='10a fit points {}'.format(int(sum(10**nEvents10_crop))))

    ax.plot(10**xfit6, 10**yfit6, label='6a E^{:1.3f}'.format(params6[0]))
    ax.plot(10**xfit10, 10**yfit10, label='10a E^{:1.3f}'.format(params10[0]))

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_ylabel('num events')
    ax.set_xlabel('Neutrino Energy')
    ax.grid()
    ax.set_title('6a={}days, 10a={}days'.format(measured_days_6a, int(measured_days_10a)))
    ax.legend()
    fig.savefig('plot_event_spectrum.pdf')

def final_10a_plot():
    # data taken from fig 3 in PoS(ICRC2019)1017 for 10 years of IceCube
    # lifetime/sec taken from Tab 1
    measured_days_10a = 259620998/60/60/24
    ebins10, nEvents10 = np.genfromtxt('atmo_nu_10years.csv', delimiter=',', skip_header=1, skip_footer=1, unpack=True)
    np.testing.assert_allclose(bin_mids, ebins10, rtol=1e-2, atol=0)

    nEvents10_crop = nEvents10[low:up]
    xfit10, yfit10, params10 = do_fit(bin_mids_crop, nEvents10_crop)

    fig = plt.figure(figsize=FIG_SIZE)
    ax = fig.add_subplot(111)

    ax.plot(10**bin_edges, 10**np.r_[nEvents10[0], nEvents10],
            drawstyle='steps', label='Event distribution of {} days'.format(int(measured_days_10a)))

    ax.plot(10**bin_mids_crop, 10**nEvents10_crop,
            linestyle='', color='k', marker='.',
            label='Fit points with {} events'.format(int(sum(10**nEvents10_crop))))

    ax.plot(10**xfit10, 10**yfit10, label='Power Law fit with index {:1.3f}'.format(params10[0]),)

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_ylabel('Number of Events')
    ax.set_xlabel('Reconstructed Neutrino Energy')
    ax.grid()
    ax.legend()
    fig.savefig('plot_event_spectrum.pdf', bbox_inches='tight', pad_inches=0.02)
    plt.close()

if __name__ == "__main__":
    # compare_6_and_10_years()
    final_10a_plot()
