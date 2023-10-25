import scipy.special as sc
import numpy as np
import mpmath as mp
import scipy.constants as const

class ScalarTensorPhase:
    def __init__(self,M1,M2,q1,q2,p1,p2,ms,c3):
        '''
        INPUT PARAMETERS
        :param M1/M2: Neutron star massse
        :param q1/q2: Scalar charges
        :param p1/p2: Induced scalar charges
        :param ms: Scalar mass parameter in units of 1/s
        :param c3: Scalar cubic self-interaction parameter
        '''
        self.M1 = M1
        self.M2 = M2
        self.q1 = q1
        self.q2 = q2
        self.p1 = p1
        self.p2 = p2
        self.ms = ms
        self.c3 = c3

        '''
        NEWTON'S CONSTANT + SPEED OF LIGHT
        '''
        self.G = const.G          #UNITS: m^3/(kg*s)
        self.c = const.c          #UNITS: m/s
        self.hbar_SI = const.hbar #UNITS: J s

        '''
        DEFINE PARAMETERS
        '''
        self.M = M1 + M2
        self.delta_q = (M2 * q1 - M1 * q2) / self.M ** 2
        self.nu = M1 * M2 / (M1 + M2) ** 2
        self.Q = (q1 + q2) / self.M
        self.QBar = q1 * q2 / (M1 * M2)
        self.msBar = (self.G * self.M * self.ms) / self.c ** 3
        self.c3Bar = c3 * self.G * self.M ** 2 * self.Q * self.QBar
        self.xi_q = (M2 ** 2 * q1 + M1 ** 2 * q2) / (self.M ** 3)
        self.xi_p = (M1 * p2 * q1 - M2 * p1 * q2) / (self.M ** 3) * self.delta_q
        self.delta_M = M1 - M2
        self.g1 = (M2 ** 2 * q1 - M1 ** 2 * q2) / (self.M ** 3)
        self.g2 = (M2 ** 3 * q1 - M1 ** 3 * q2) / (self.M ** 4)
        self.xi_c = c3 * self.QBar * self.delta_M * self.delta_q / self.M

    def Psi_E(self, f):
        '''
        :param f: Orbital frequency
        :return: 0PN correction due to modified Kepler relation
        '''
        v = (np.pi * self.M * self.G * f / self.c**3)**(1/3)

        first_term = 5 * self.QBar / (6 * self.nu * self.msBar ** (5 / 2)) * (sc.gammainc(5 / 2, self.msBar / v ** 2) + sc.gammainc(7 / 2, self.msBar / v ** 2) + 2 * sc.gammainc(9 / 2,self.msBar / v ** 2))
        second_term = v ** 3 * 5 * self.QBar / (6 * self.nu * self.msBar ** 4) * (sc.gammainc(4, self.msBar / v ** 2) + sc.gammainc(5, self.msBar / v ** 2) + 2 * sc.gammainc(6,self.msBar / v ** 2))
        return float(first_term + second_term)

    def Psi_Minus1PN(self, f):
        '''
        :param f: Orbital frequency
        :return: Leading order phase correction due to scalar dipole radiation (-1PN order)
        '''
        v = (np.pi * self.M * self.G * f / self.c**3)**(1/3)

        first_term = -1/v**7 * mp.hyp3f2(-3/2, 5/3, 7/6, 8/3, 13/6, self.msBar**2 / v**6)
        second_term = -90 * 2**(1/3) * 3**(1/2) * v**3 * sc.gamma(2/3)**3 / (247 * np.pi * self.msBar**(10/3))
        third_term = 63 * sc.gamma(1/3)**3 / (256 * 2**(1/3) * np.pi * self.msBar**(7/3))

        return float(np.real(np.heaviside((v**3/self.msBar) - 1, 1) * (self.delta_q**2 * 5 * self.G/(896 * self.nu**3) * (first_term + second_term + third_term))))

    def Psi_0PN_q(self, f):
        '''
        :param f: Orbital frequency
        :return: Phase correction due to scalar radiation at 0PN, contribution from scalar charges q1, q2
        '''
        v = (np.pi * self.M * self.G * f / self.c**3)**(1/3)

        first_term = (-15 * self.g2 * self.delta_q * self.msBar**2) / ((9856 * self.nu**3 * v**11) * mp.hyp3f2(-3/2, 7/3, 11/6, 10/3, 17/6, self.msBar**2 / v**6))
        second_term = (45 * sc.gamma(2/3)**3 * self.delta_q * (-1680 * self.g1 - 966 * self.g2 + 5 * (1064 * self.nu + 659) * self.delta_q))/(1605632 * (2**(2/3)) * np.pi * self.nu**3 * self.msBar**(5/3))
        third_term = (self.delta_q * (1680 * self.g1 + 1008 * self.g2 - 5 * (1064 * self.nu + 659) * self.delta_q)) /((86016 * self.nu**3 * v**5) * mp.hyp3f2(-3/2, 4/3, 5/6, 7/3, 11/6, self.msBar**2 / v**6))
        fourth_term = (5 * np.sqrt(3) * v**3 * (sc.gamma(1/3)**3) * self.delta_q * (7728 * self.g1 + 4368 * self.g2 - 23 * (1064 * self.nu + 659) * self.delta_q))/(30829568 * (2**(1/3)) * np.pi * self.nu**3 * self.msBar**(8/3))

        return float((first_term + second_term + third_term + fourth_term) * np.real(np.heaviside((v**3/self.msBar) - 1, 1)))

    def Psi_0PN_p(self, f):
        '''
        :param f: Orbital frequency
        :return: Phase correction due to scalar radiation at 0PN, contribution from induced scalar charges p1, p2
        '''
        v = (np.pi * self.M * self.G * f / self.c**3)**(1/3)

        first_term = -1/v**5 * mp.hyp3f2(-3/2, 4/3, 5/6, 7/3, 11/6, self.msBar**2 / v**6))
        second_term = - (6 * 2**(1/3) * 3**(1/2) * v**3 * sc.gamma(1/3)**3) / (187 * np.pi * self.msBar**(8/3))
        third_term = (135 * sc.gamma(2/3)**3) / (56 * 2**(1/3) * np.pi * self.msBar**(5/3))

        return float(np.real(np.heaviside((v**3/self.msBar) - 1, 1) * (self.xi_p * 5 / (16 * self.nu**3) * (first_term + second_term + third_term))))

    def Psi_0PN_c3(self, f):
        '''
        :param f: Orbital frequency
        :return: Phase correction due to scalar radiation at 0PN, contribution from cubic scalar coupling c3
        '''
        v = (np.pi * self.M * self.G * f / self.c**3)**(1/3)

        Psi_0PN_c3_1 = 1/v**7 * mp.hyp3f2(-3/2, 5/3, 7/6, 8/3, 13/6, self.msBar**2 / v**6) + 90 * 2**(1/3) * 3**(1/2) * v**3 * sc.gamma(2/3)**3 / (247 * np.pi * self.msBar**(10/3)) - 63 * sc.gamma(1/3)**3 / (256 * 2**(1/3) * np.pi * self.msBar**(7/3))
        Psi_0PN_c3_2 = -1/v**9 * mp.hyp3f2(-3/2, 2, 3/2, 3, 5/2, self.msBar**2 / v**6) - 24 * v**3 / (35 * self.msBar**4) + 3 * np.pi/(8 * self.msBar**3)

        first_term = 1/(self.c * self.hbar_SI) * self.xi_c * (5 * self.G * self.M1 * self.M2)/(7168 * np.pi * self.msBar * self.nu**3) * Psi_0PN_c3_1
        second_term = 1/(self.c * self.hbar_SI) * self.xi_c * (25 * self.G * self.M1 * self.M2)/(55296 * np.pi * self.nu**3) * Psi_0PN_c3_2
        return float(np.real(np.heaviside((v**3/self.msBar) - 1, 1) * (first_term + second_term)))

    def Psi_0PN_l2(self, f):
        v = (np.pi * self.M * self.G * f / self.c**3)**(1/3)

        first_term = -1/v**5 * mp.hyp3f2(-5/2, 4/3, 5/6, 7/3, 11/6, self.msBar**2 / (4*v**6))
        second_term = - 720 * 2**(1/3) * 3**(1/2) * v**3 * sc.gamma(1/3)**3 / (4301 * np.pi * self.msBar**(8/3))
        third_term = 405 * sc.gamma(2/3)**3 / (112 * np.pi * self.msBar**(5/3))
        return float(np.real(np.heaviside((v**3/self.msBar) - 1, 1/2) * self.xi_q**2 /(32*self.nu**3) * (first_term + second_term + third_term)))

    def Psi_ST_total(self, f):
        return float(self.Psi_E(f) + self.Psi_Minus1PN(f) + self.Psi_0PN_q(f) + self.Psi_0PN_p(f) + self.Psi_0PN_c3(f) + self.Psi_0PN_l2(f))