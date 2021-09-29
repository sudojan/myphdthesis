import numpy as np
import matplotlib.pyplot as plt

# Mun Table of coefficients for hard component taken from
# Bugaev et al. / Astroparticle Physics 21 (2004) 491–509
muon_table = np.array([
    [7.174409e-4, 1.7132e-3, 4.082304e-3, 8.628455e-3, 0.01244159, 0.02204591, 0.03228755 ],
    [ -0.2436045, -0.5756682, -1.553973, -3.251305, -5.976818, -9.495636, -13.92918 ],
    [-0.2942209, -0.68615, -2.004218, -3.999623, -6.855045, -10.05705, -14.37232 ],
    [-0.1658391, -0.3825223, -1.207777, -2.33175, -3.88775, -5.636636, -8.418409 ],
    [-0.05227727, -0.1196482, -0.4033373, -0.7614046, -1.270677, -1.883845, -2.948277 ],
    [-9.328318e-3, -0.02124577, -0.07555636, -0.1402496, -0.2370768, -0.3614146, -0.5819409 ],
    [-8.751909e-4, -1.987841e-3, -7.399682e-3, -0.01354059, -0.02325118, -0.03629659, -0.059275 ],
    [-3.343145e-5, -7.584046e-5, -2.943396e-4, -5.3155e-4, -9.265136e-4, -1.473118e-3, -2.419946e-3]
])
energy_arr = np.geomspace(1e3, 1e9, 7)

def hard_component(v, energy_bin):
    tmp = 0.
    for idx in range(8):
        tmp += muon_table[idx,energy_bin] * np.log10(v)**idx
    return tmp / v

def plot_dsigma_dv(v_arr):
    for e_i in range(7):
        plt.plot(v_arr*energy_arr[e_i], [hard_component(v_i, e_i) for v_i in v_arr],
            label=r'$E=${:.2g} GeV'.format(energy_arr[e_i]))
    plt.xlabel(r'Energy Loss $Ev$ / GeV')
    plt.ylabel(r'$\left.\frac{\mathrm{d}\sigma}{\mathrm{d}v}\right|_{\mathrm{hard}}$')
    plt.xscale('log')
    plt.yscale('log')
    plt.legend()
    plt.savefig('hard_component_dsigma_dv.png', bbox_inches='tight', pad_inches=0.02, dpi=200)
    plt.close()

def plot_v_dsigma_dv(v_arr):
    for e_i in range(7):
        plt.plot(v_arr, [hard_component(v_i, e_i)*v_i for v_i in v_arr],
            label=r'$E=${:.2g} GeV'.format(energy_arr[e_i]))
    plt.xlabel(r'Relative Energy Loss $v$')
    plt.ylabel(r'$v \left.\frac{\mathrm{d}\sigma}{\mathrm{d}v}\right|_{\mathrm{hard}}$')
    plt.xscale('log')
    plt.yscale('log')
    plt.legend()
    plt.savefig('hard_component_v_dsigma_dv.png', bbox_inches='tight', pad_inches=0.02, dpi=200)
    plt.close()

rhode_x_arr = [0, 0.1, 0.144544, 0.20893, 0.301995, 0.436516,
    0.630957, 0.912011, 1.31826, 1.90546, 2.75423, 3.98107, 5.7544, 8.31764,
    12.0226, 17.378, 25.1189, 36.3078, 52.4807, 75.8577, 109.648, 158.489,
    229.087, 331.131, 478.63, 691.831, 1000, 1445.44, 2089.3, 3019.95,
    4365.16, 6309.58, 9120.12, 13182.6, 19054.6, 27542.3, 39810.8, 57544,
    83176.4, 120226, 173780, 251188, 363078, 524807, 758576, 1.09648e+06,
    1.58489e+06, 2.29086e+06, 3.3113e+06, 4.78628e+06, 6.91828e+06,
    9.99996e+06]

rhode_y_arr = [0, 0.0666667, 0.0963626, 159.74, 508.103, 215.77,
        236.403, 201.919, 151.381, 145.407, 132.096, 128.546, 125.046, 121.863,
        119.16, 117.022, 115.496, 114.607, 114.368, 114.786, 115.864, 117.606,
        120.011, 123.08, 126.815, 131.214, 136.278, 142.007, 148.401, 155.46,
        163.185, 171.574, 180.628, 190.348, 200.732, 211.782, 223.497, 235.876,
        248.921, 262.631, 277.006, 292.046, 307.751, 324.121, 341.157, 358.857,
        377.222, 396.253, 415.948, 436.309, 457.334, 479.025]

def save_table_to_tex():
    x_arr = np.logspace(-1, 7, 51).reshape(3,17)
    y_arr = np.array(rhode_y_arr[1:]).reshape(3,17)
    a = np.column_stack([x_arr[0], y_arr[0], x_arr[1], y_arr[1], x_arr[2], y_arr[2]])
    np.savetxt("table_rhode_nucl.csv", a, delimiter=' & ', fmt='%2.4e', newline=' \\\\\n')

def plot_rhode_param():
    plt.plot(x_arr, y_arr)
    plt.axvline(200)
    plt.xscale('log')
    plt.show()


if __name__ == '__main__':
    v_arr = np.geomspace(1e-6, 0.99, 100)
    # plot_dsigma_dv(v_arr)
    # plot_v_dsigma_dv(v_arr)
    save_table_to_tex()