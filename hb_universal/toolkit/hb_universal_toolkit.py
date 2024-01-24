"""
========
hb_universal toolkit
========

hb_universal toolkit for The System Development Kit

Initially written by Otto Simoa, otto.simola@aalto.fi, 2023.

"""

import os
import sys
if not (os.path.abspath('../../thesdk') in sys.path):
    sys.path.append(os.path.abspath('../../thesdk'))

from FIR_model import FIR_model
from HB_model import HB_model

#from plot_PSD import *
#from NR_signal_generator import *
from itertools import chain
#import plot_format

import matplotlib.pyplot as plt
import numpy as np
import math
import yaml
import pdb

class hb_universal_toolkit():
    def __init__(self, *arg):
        self.FIR = FIR_model()
        self.HB = HB_model()

    def generate_Hfiles(self, root):
        intHB1 = HB_model()
        intHB2 = HB_model()
        intHB3 = HB_model()
        decHB1 = HB_model()
        decHB2 = HB_model()
        decHB3 = HB_model()

        intHB1.H = intHB1.calc_coeffs(40, [0, 0.45, 0.499, 0.5])
        intHB2.H = intHB2.calc_coeffs(8, [0, 0.225, 0.499, 0.5])
        intHB3.H = intHB3.calc_coeffs(6, [0, 0.1125, 0.499, 0.5])
        decHB1.H = decHB1.calc_coeffs(6, [0, 0.1125, 0.499, 0.5])
        decHB2.H = decHB2.calc_coeffs(8, [0, 0.225, 0.499, 0.5])
        decHB3.H = decHB3.calc_coeffs(40, [0, 0.45, 0.499, 0.5])

        self.export_yaml(intHB1, root + "f2_interpolator/hb_interpolator/configs/hb1-config.yml")
        self.export_yaml(intHB2, root + "f2_interpolator/hb_interpolator/configs/hb2-config.yml")
        self.export_yaml(intHB3, root + "f2_interpolator/hb_interpolator/configs/hb3-config.yml")
        self.export_yaml(decHB1, root + "f2_decimator/hb_decimator/configs/hb1-config.yml")
        self.export_yaml(decHB2, root + "f2_decimator/hb_decimator/configs/hb2-config.yml")
        self.export_yaml(decHB3, root + "f2_decimator/hb_decimator/configs/hb3-config.yml")

    def load_H(self, fname):
        with open(fname, 'r') as file:
            yaml_f = yaml.safe_load(file)

        ymlReso = yaml_f['resolution']
        ymlH = yaml_f['H']

        H = []          

        for tap in ymlH:
            H.append(tap * (math.pow(2, ymlReso - 1) - 1))

        return H

    def load_yaml(self, model, fname):
        with open(fname, 'r') as file:
            yaml_f = yaml.safe_load(file)

        new_reso = yaml_f['resolution']
        new_gB = yaml_f['gainBits']
        ymlH = yaml_f['H']

        if model == "HB":
            self.HB.resolution = new_reso
            self.HB.gainBits = new_gB
  
            H = []          
  
            for tap in ymlH:
                H.append(tap * (math.pow(2, self.HB.resolution - 1) - 1))
            
            self.HB.n = (len(H) + 1) / 2
            self.HB.H = H
        else:
            self.FIR.resolution = new_reso
            self.FIR.gainBits = new_gB
  
            H = []          
  
            for tap in ymlH:
                H.append(tap * (math.pow(2, self.FIR.resolution - 1) - 1))
            
            self.FIR.n = len(H)
            self.FIR.H = H
            

    def export_yaml(self, model, fname):
        tapfile = os.path.dirname(os.path.realpath(__file__)) + "/" + fname

        fid = open(tapfile, 'w')

        msg = "#Generated by TheSDK/fir.export_yaml\n"
        fid.write(msg)

        msg = "syntax_version: 2\n"
        fid.write(msg)

        H = model.H
        msg = "resolution: " + str(model.resolution) + "\n"
        fid.write(msg)

        msg = "gainBits: "+ str(model.gainBits) + "\n"
        fid.write(msg)

        msg = "H: ["
        fid.write(msg)
        lines = H.shape[0]

        for k in range(lines - 1):
            fid.write("%0.32f,\n" %(H[k]))

        fid.write("%0.32f]\n" %(H[lines - 1]))

        fid.close()

    def plot_coeff_fft(self, coeffs1, coeffs2, coeffs3):
        signal1 = np.concatenate((np.zeros(int((1024 - len(coeffs1)) / 2)), coeffs1, np.zeros(int((1024 - len(coeffs1)) / 2))))
        fft_coeffs1 =  np.fft.fft(signal1)
        freq1 = np.fft.fftfreq(len(signal1), d=1)

        signal2 = np.concatenate((np.zeros(int((1024 - len(coeffs2)) / 2)), coeffs2, np.zeros(int((1024 - len(coeffs2)) / 2))))
        fft_coeffs2 =  np.fft.fft(signal2)
        freq2 = np.fft.fftfreq(len(signal2), d=1)

        signal3 = np.concatenate((np.zeros(int((1024 - len(coeffs3)) / 2)), coeffs3, np.zeros(int((1024 - len(coeffs3)) / 2))))
        fft_coeffs3 =  np.fft.fft(signal3)
        freq3 = np.fft.fftfreq(len(signal3), d=1)

        figure, axis = plt.subplots(2, 3, figsize=(10,6))

        axis[0, 0].plot(np.linspace(0, 1, len(signal1)), signal1, 'b-')
        axis[0, 0].set_title("HB1 Coefficients")  
        axis[0, 0].set_xlim([0.45, 0.55])
         
        axis[1, 0].plot(freq1[:int(len(fft_coeffs1)/2)], 20*np.log10(fft_coeffs1)[:int(len(fft_coeffs1)/2)], 'g')
        axis[1, 0].set_title("HB1 Frequency response")


        axis[0, 1].plot(np.linspace(0, 1, len(signal2)), signal2, 'b-')
        axis[0, 1].set_title("HB2 Coefficients")  
        axis[0, 1].set_xlim([0.48, 0.52])
         
        axis[1, 1].plot(freq2[:int(len(fft_coeffs2)/2)], 20*np.log10(fft_coeffs2)[:int(len(fft_coeffs2)/2)], 'g')
        axis[1, 1].set_title("HB2 Frequency response")


        axis[0, 2].plot(np.linspace(0, 1, len(signal3)), signal3, 'b-')
        axis[0, 2].set_title("HB3 Coefficients")  
        axis[0, 2].set_xlim([0.48, 0.52])
         
        axis[1, 2].plot(freq3[:int(len(fft_coeffs3)/2)], 20*np.log10(fft_coeffs3)[:int(len(fft_coeffs3)/2)], 'g')
        axis[1, 2].set_title("HB3 Frequency response")

        figure.show()

    def plot_sig_fft(self, vec, signals, modes):
        #pdb.set_trace()
        figure, axis = plt.subplots(1, len(modes), figsize=(8,2))

        figure.suptitle("Frequency responses")

        sig_count = len(signals)

        for i in range(sig_count):
            if vec == "I":
                signal = signals[i][0]
            else:
                signal = signals[i][1]

            sig_len = signal.size
            fft_signal = np.fft.fft(signal)
            freq = np.fft.fftfreq(len(signal), d=1)
             
            axis.plot(freq[:int(sig_len/2)], 20*np.log10(fft_signal)[:int(len(fft_signal)/2)], 'g')
            axis.set_title(f"mode: {modes[i]}")
            
        figure.show()

    def plot_simple_signals(self, vecs, input_signal, output_signals, modes):
        xlims_py = [0, len(output_signals[0][0])] #For plotting
        pdb.set_trace()

        if ("I" in vecs): 
            Ifigure, Iaxis = plt.subplots(2, 3, figsize=(8,4))
            Ifigs = list(chain.from_iterable(Iaxis))

            fg0 = Ifigs[0]
            fg0.plot(np.arange(0, len(input_signal), 1), input_signal, 'r-') 
            fg0.set_title("Input")
 
        if ("Q" in vecs):
            Qfigure, Qaxis = plt.subplots(2, 3, figsize=(8,4))
            Qfigs = list(chain.from_iterable(Qaxis))

            fg0 = Qfigs[0]
            fg0.plot(np.arange(0, len(input_signal), 1), input_signal, 'r-') 
            fg0.set_title("Input") 

        for i in range(0, len(modes)):
            if ("I" in vecs):
                fgI = Ifigs[i + 1]
                fgI.plot(np.arange(0, len(output_signals[i][0]), 1), output_signals[i][0], 'g-')
                fgI.set_title("hb_universal I vector [sv, " + modes[i] + "]")
        
            if ("Q" in vecs):
                fgQ = Qfigs[i + 1]
                fgQ.plot(np.arange(0, len(output_signals[i][1]), 1), output_signals[i][1], 'g-')
                fgQ.set_title("hb_universal Q vector [sv, " + modes[i] + "]")


if __name__=="__main__":
    import matplotlib.pyplot as plt
    from  model import *
    
    model = model()
    input()
