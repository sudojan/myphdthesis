import numpy as np
import matplotlib.pyplot as plt
import ternary

class NuOsc(object):
    def __init__(self, inverted=True):
        # PDG 2020
        self.s12 = 0.307
        self.s13 = 0.0218
        self.delta = 1.36*np.pi
        self.dm21 = 7.53e-5
        if inverted:
            self.s23 = 0.547
            self.dm32 = -2.546e-3
        else:
            self.s23 = 0.545
            self.dm32 = 2.453e-3
        
        self.U_PMNS = self.create_PMNS()
        self.mass_arr = self.create_masses()

    def create_masses(self):
        m2_2 = max(self.dm21, self.dm32)
        m1_2 = m2_2 - self.dm21
        m3_2 = m2_2 + self.dm32
        return np.array([m1_2, m2_2, m3_2])

    def create_PMNS(self):
        c12 = np.sqrt(1 - self.s12)
        c13 = np.sqrt(1 - self.s13)
        c23 = np.sqrt(1 - self.s23)
        s12 = np.sqrt(self.s12)
        s13 = np.sqrt(self.s13)
        s23 = np.sqrt(self.s23)
        R23 = np.array([[1, 0, 0], [0, c23, s23], [0, -s23, c23]])
        R13 = np.array([[c13, 0, s13*np.exp(-1.j*self.delta)], [0, 1, 0], [-s13*np.exp(1.j*self.delta), 0, c13]])
        R12 = np.array([[c12, s12, 0], [-s12, c12, 0], [0, 0, 1]])
        return R23 @ R13 @ R12

    def prob(self, Alpha, Beta, LoverE):
            ots = 1.2669327621645516
            summe = np.sum(np.conjugate(self.U_PMNS[Alpha]) * self.U_PMNS[Beta] * np.exp(-2.j * ots * self.mass_arr * LoverE))
            return summe.real**2 + summe.imag**2

#     def prob_cross_check(self, Alpha, Beta, LoverE):
#         ots = 1.2669327621645516
#         dm31 = self.dm21 + self.dm32
#         m_arr = np.array([[0, self.dm21, dm31], [self.dm21, 0, self.dm32], [dm31, self.dm32, 0]])
#         kronecker_d = np.eye(3)[Alpha, Beta]
#         sum1 = 0
#         for k in range(2):
#             for j in range(3):
#                 if k >= j:
#                     continue
#                 tmp = np.conjugate(self.U_PMNS[Alpha,j]) * self.U_PMNS[Beta, j] * self.U_PMNS[Alpha,k] * np.conjugate(self.U_PMNS[Beta, k])
#                 sum1 += tmp.real * np.sin(ots*m_arr[j,k]*LoverE)**2
#         sum2 = 0
#         for k in range(2):
#             for j in range(3):
#                 if k >= j:
#                     continue
#                 tmp = np.conjugate(self.U_PMNS[Alpha,j]) * self.U_PMNS[Beta, j] * self.U_PMNS[Alpha,k] * np.conjugate(self.U_PMNS[Beta, k])
#                 sum2 += tmp.imag * np.sin(2*ots*m_arr[j,k]*LoverE)
#         return kronecker_d - 4*sum1 + 2*sum2

    def plot_osc(self, nu_i, LoverE, filename):
        figure, ax = plt.subplots()
        x_arr = np.asarray(LoverE)
        ax.plot(x_arr, np.vectorize(self.prob)(nu_i,0,x_arr), c='k', label=r'$\nu_{e}$')
        ax.plot(x_arr, np.vectorize(self.prob)(nu_i,1,x_arr), c='b', label=r'$\nu_{\mu}$')
        ax.plot(x_arr, np.vectorize(self.prob)(nu_i,2,x_arr), c='r', label=r'$\nu_{\tau}$', ls='dotted')
        ax.set_xlabel('L/E / km/GeV')
        ax.set_ylabel(r'Probability ($\nu_{e} \to \nu_{\mathrm{final}}$)')
        ax.legend()

        if filename.endswith('.pdf'):
            figure.savefig(filename, bbox_inches='tight', pad_inches=0.02)
        else:
            figure.savefig(filename, bbox_inches='tight', pad_inches=0.02, dpi=300)


    def prob_avg(self, Alpha, Beta):
        kronecker_d = np.eye(3)[Alpha, Beta]
        sum1 = 0
        for k in range(2):
            for j in range(k+1,3):
                tmp = np.conjugate(self.U_PMNS[Alpha,j]) * self.U_PMNS[Beta, j] * self.U_PMNS[Alpha,k] * np.conjugate(self.U_PMNS[Beta, k])
                sum1 += tmp.real

        return kronecker_d - 2*sum1

    def calc_flavor_ratio(self, ratio_i):
        ratio = np.empty((3,3))
        for idx in range(3):
            ratio[idx] = np.array([self.prob_avg(idx,0), self.prob_avg(idx,1), self.prob_avg(idx,2)])*ratio_i[idx]
        return np.sum(ratio, axis=0)

    def plot_flavor_triangle(self, filename):
        flavor_ratios = [[1,2,0], [0,1,0], [1,0,0], [1,1,0], [2,1,0]]
        labels = ['Pion decay', 'Muon dumped', 'Neutron beam']
        markers = ['*', 's', 'o']

        # figure, ax = plt.subplots()
        # tax = ternary.TernaryAxesSubplot(ax=ax)
        figure, tax = ternary.figure(scale=1.0)
        alen = 7
        figure.set_size_inches(alen, 0.75*alen)

        # Draw Boundary and Gridlines
        tax.boundary(linewidth=1.5)
        tax.gridlines(color="black", multiple=0.2)#, linewidth=0.5, ls='dotted')
        tax.ticks(axis='lbr', multiple=0.2, linewidth=1, tick_formats="%.1f", offset=0.02)

        fontsize = 14
        offset = 0.14
        # tax.set_title(r'$\nu_e$')
        tax.bottom_axis_label(r'$\nu_e$', fontsize=fontsize)#, offset=offset)
        tax.right_axis_label(r'$\nu_{\mu}$', fontsize=fontsize, offset=offset, rotation=0)
        tax.left_axis_label(r'$\nu_{\tau}$', fontsize=fontsize, offset=offset, rotation=0)

        tax.line(self.calc_flavor_ratio([0,1,0]),
                 self.calc_flavor_ratio([1,0,0]),
                 linewidth=7., marker=None, color='C7', linestyle="-", alpha=0.5,
                 label=r'w/o initial $\nu_{\tau}$')

        for idx in range(len(labels)):
            ratio = np.asarray(flavor_ratios[idx])/sum(flavor_ratios[idx])
            tax.line(ratio, self.calc_flavor_ratio(ratio),
                    linewidth=3., marker=markers[idx], markersize=10,
                    linestyle=":", color='C{}'.format(idx),
                    label=labels[idx])

        tax.legend(fontsize=10, loc='upper left')

        # Remove default Matplotlib Axes
        tax.clear_matplotlib_ticks()
        tax.get_axes().axis('off')

        # redraw labels by hand, otherwise they are not visible
        # seems to be a bug
        # https://github.com/marcharper/python-ternary/issues/36
        tax._redraw_labels()

        if filename.endswith('.pdf'):
            figure.savefig(filename, bbox_inches='tight', pad_inches=0.02)
        else:
            figure.savefig(filename, bbox_inches='tight', pad_inches=0.02, dpi=300)

if __name__ == '__main__':
    osc = NuOsc()
    # # print(osc.prob(1,0,400) + osc.prob(1,1,400) + osc.prob(1,2,400))
    # # print(osc.prob_avg(1,0) + osc.prob_avg(1,1) + osc.prob_avg(1,2))
    # # osc.plot_osc(0, np.linspace(0, 35000, 500), 'nu_osc_len.pdf')
    osc.plot_flavor_triangle('nu_flavor_triangle.pdf')
