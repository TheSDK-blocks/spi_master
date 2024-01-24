"""
========
FIR
========

FIR model The System Development Kit

Initially written by Otto Simoa, otto.simola@aalto.fi, 2023.

"""

import os
import sys
if not (os.path.abspath('../../thesdk') in sys.path):
    sys.path.append(os.path.abspath('../../thesdk'))

import scipy.signal as sig
import numpy as np

class FIR_model():
    def __init__(self):
        self.resolution = 16

        self.n = 40
        self.H = []

        self.gainBits = 10
        self.scale = 1

    def weighted_sum(self, x_vector):
        ''' Calculate one value for FIR filtering

        '''
        y = []

        for i in range(0, self.n):
            y.append(self.H[i] * x_vector[::-1][i])

        return sum(y) 

    def calc_FIR(self, dut_array):
        ''' Calculate values for a set of FIR filtering

        '''
        y = []
        p = len(dut_array) + self.n
        x_array = np.concatenate(([0] * self.n, np.copy(dut_array[:, 0]),  [0] * self.n))

        for i in range(0, p):
            x_vector = x_array[i:i + self.n]
            y.append(self.weighted_sum(x_vector) * self.scale)
    
        return np.array(y[:len(dut_array)], copy=True)

    def calc_coeffs(self, n):
	    desired = np.array([1, 0]) #Low-pass
	    bandwidth = 0.47
	    bands = np.array([0, bandwidth, 0.499, 0.5])
	    coeffs = sig.remez(n + 1, bands, desired, fs = 1)

	    return coeffs

if __name__=="__main__":
    import matplotlib.pyplot as plt
    from  model import *

    input()
