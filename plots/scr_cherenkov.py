import numpy as np
from scipy.constants import physical_constants

alpha = physical_constants['fine-structure constant'][0] # 1./137
r_e = physical_constants['classical electron radius'][0] * 1e2 # 2.8e-13 cm
m_e = physical_constants['electron mass energy equivalent in MeV'][0] * 1e6 # 0.5e6 eV

class Cherenkov_calc(object):
    def __init__(self, refraction_index):
        self.refraction_index = refraction_index
        self.lorentz_beta = 1. # assuming particle propagates with c

    @property
    def cherenkov_angle(self):
        """Get Cherenkov angle of medium in degree"""
        return np.rad2deg(np.arccos(1/(self.lorentz_beta * self.refraction_index)))

    def num_photons(self, lambda1, lambda2):
        r""" Frank-Tamm Formula
        Calculates the number of photons per centi meter
        $
        \frac{\mathrm{d}N}{\mathrm{d}X} = 
            \int_{\lambda_1}^{\lambda_2} \mathrm{d}\lambda
                \frac{\mathrm{d}^2N}{\mathrm{d}X\mathrm{d}\lambda}
            2 \pi \alpha \sin^2 \theta_c =
                \left( \frac{1}{\lambda_1} - \frac{1}{\lambda_2} \right)
        $

        Parameters
        ----------
        lambda1 : float
            Lower wavelength limit in nm.
        lambda2 : float
            Upper wavelength limit in nm.
        """
        sin2_tc = 1 - (self.refraction_index * self.lorentz_beta)**(-2)
        return 2 * np.pi * alpha * sin2_tc * (1e7/lambda1 - 1e7/lambda2)

    def energy_loss(self):
        """Energy loss in eV/cm
        """
        sin2_tc = 1 - (self.refraction_index * self.lorentz_beta)**(-2)
        return alpha**2 / (r_e * m_e) * sin2_tc


if __name__ == '__main__':
    cherenkov = Cherenkov_calc(1.35) # for ice
    print('cherenkov angle: {} rad'.format(cherenkov.cherenkov_angle))
    print('cherenkov energy loss: {} eV/cm'.format(cherenkov.energy_loss()))
    print('number of photons: {} /cm'.format(cherenkov.num_photons(400, 700)))