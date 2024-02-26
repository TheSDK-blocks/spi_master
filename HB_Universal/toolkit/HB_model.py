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
import pdb

class HB_model():
    def __init__(self):
        self.resolution = 16

        self.n = 40
        self.H = []

        self.gainBits = 10
        self.scale = 1

    def calc_coeffs(self, n, bands):
        desired = np.array([1, 0]) #Low-pass

        coeffs = sig.remez(n, np.array(bands), desired, fs = 1)

        hb = np.zeros((2 * n - 1, 1))
        hb[0::2, 0] = coeffs
        hb[n - 1, 0] = 1

        return hb

if __name__=="__main__":
    import matplotlib.pyplot as plt
    from  model import *

    input()
