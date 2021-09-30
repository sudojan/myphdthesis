Single-differential cross-sections dsigma/dy in the perturbative DGLAP
formalism at NLO, using HERAPDF1.5
cf. also: Cooper-Sarkar, Mertsch and Sarkar, JHEP08 (2011) 042
Philipp Mertsch (pmertsch@stanford.edu)
28 March 2015

The 12 files

dsigmady_nu_CC_n_NLO_HERAPDF15NLO_EIG.dat
dsigmady_nu_CC_p_NLO_HERAPDF15NLO_EIG.dat
dsigmady_nu_CC_iso_NLO_HERAPDF15NLO_EIG.dat
dsigmady_nu_NC_n_NLO_HERAPDF15NLO_EIG.dat
dsigmady_nu_NC_p_NLO_HERAPDF15NLO_EIG.dat
dsigmady_nu_NC_iso_NLO_HERAPDF15NLO_EIG.dat
dsigmady_nubar_CC_n_NLO_HERAPDF15NLO_EIG.dat
dsigmady_nubar_CC_p_NLO_HERAPDF15NLO_EIG.dat
dsigmady_nubar_CC_iso_NLO_HERAPDF15NLO_EIG.dat
dsigmady_nubar_NC_n_NLO_HERAPDF15NLO_EIG.dat
dsigmady_nubar_NC_p_NLO_HERAPDF15NLO_EIG.dat
dsigmady_nubar_NC_iso_NLO_HERAPDF15NLO_EIG.dat

contain the neutrino/antineutrino CC/NC single-differential
cross-sections on neutrons/protons/isoscalar targets. Each file
consists of 11100 lines, with consecutive blocks of 100 lines for each
of the 111 neutrino energies. The neutrino energies are the same as in
the production for ANIS, i.e. from 10 GeV to 10^12 GeV in logarithmic
steps of 0.1 (i.e. 10^1 GeV, 10^1.1 GeV etc.) for a total of 111
energies. Please note that for neutrino energies <~ 100 GeV the
interaction is not in the perturbative regime and thus the
cross-sections are not reliable. For each neutrino energy the
cross-section is computed at 100 y-points, also in logarithmic steps,
i.e. from y = ymin * 10^dlgy to y=1 where ymin = Q2/s and dlgy =
-log10(ymin) / 100 (i.e. ymin^0.99, ymin^0.98 etc.). The first
(second) number in each line is y (dsigma/dy in pb).
