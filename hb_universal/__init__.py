"""
=========
hb_universal
=========

hb_universal model template The System Development Kit
Used as a template for all TheSyDeKick Entities.

Current docstring documentation style is Numpy
https://numpydoc.readthedocs.io/en/latest/format.html

This text here is to remind you that documentation is important.
However, youu may find it out the even the documentation of this 
entity may be outdated and incomplete. Regardless of that, every day 
and in every way we are getting better and better :).

Initially written by Marko Kosunen, marko.kosunen@aalto.fi, 2017.

"""

import os
import sys
import pdb

if not (os.path.abspath("../../thesdk") in sys.path):
    sys.path.append(os.path.abspath("../../thesdk"))

sys.path.append('toolkit/')

from thesdk import IO, thesdk
from rtl import rtl, rtl_iofile, rtl_connector_bundle

from NR_signal_generator import *
from receiver import *
from plot_PSD import *


import numpy as np
import math

class hb_universal(thesdk,rtl):
    @property
    def _classfile(self):
        return os.path.dirname(os.path.realpath(__file__)) + "/"+__name__

    def __init__(self,*arg): 
        self.print_log(type='I', msg='Inititalizing %s' %(__name__)) 
        self.proplist = [ 'Rs' ];    # Properties that can be propagated from parent
        self.Rs =  100e6;            # Sampling frequency
        self.IOS=Bundle()            # Pointer for input data
        self.IOS.Members["convmode"] = IO()
        self.IOS.Members["scale"] = IO()
        self.IOS.Members["output_switch"] = IO()

        self.IOS.Members["iptr_A"] = IO()
        self.IOS.Members["Z"] = IO()

        self.IOS.Members['control_write']= IO() 
        #self.IOS.Members["clock"] = IO()
        #self.IOS.Members["reset"] = IO()
        self.model='py';             # Can be set externally, but is not propagated
        self.par= False              # By default, no parallel processing
        self.queue= []               # By default, no parallel processing

        if len(arg)>=1:
            parent=arg[0]
            self.copy_propval(parent,self.proplist)
            self.parent =parent;

        self.init()

    def init(self):
        pass #Currently nohing to add

    def main(self):
        '''Guideline. Isolate python processing to main method.
        
        To isolate the interna processing from IO connection assigments, 
        The procedure to follow is
        1) Assign input data from input to local variable
        2) Do the processing
        3) Assign local variable to output

        '''
        pass
    def define_io_conditions(self):
        """When iofiles start to roll
        """
        #Inputs roll after initdone
        self.iofile_bundle.Members['scale'].rtl_io_condition='initdone'
        self.iofile_bundle.Members['output_switch'].rtl_io_condition='initdone'
        self.iofile_bundle.Members['convmode'].rtl_io_condition='initdone'
        self.iofile_bundle.Members['iptr_A'].rtl_io_condition='initdone'

        #Outputs roll after initdone and default conds
        self.iofile_bundle.Members['Z'].rtl_io_condition_append(cond='&& initdone')

    def run(self,*arg):
        '''Guideline: Define model depencies of executions in `run` method.

        '''
        if self.model == "py":
            self.main()
        elif self.model in [ 'sv', 'icarus' ]:
            #Inputs
            _=rtl_iofile(self, name='scale', dir='in', iotype='sample', datatype='int', ionames=['io_control_hb1scale'])
            _=rtl_iofile(self, name='output_switch', dir='in', iotype='sample', datatype='int', ionames=['io_control_hb3output_switch'])
            _=rtl_iofile(self, name='mode', dir='in', iotype='sample', datatype='int', ionames=['io_control_mode'])
            _=rtl_iofile(self, name='convmode', dir='in', iotype='sample', datatype='int', ionames=['io_control_convmode'])
            _=rtl_iofile(self, name='iptr_A', dir='in', iotype='sample', datatype='scomplex', ionames=['io_in_iptr_A_real', 'io_in_iptr_A_imag'])

            #Outputs
            _=rtl_iofile(self, name='Z', dir='out', iotype='sample', datatype='scomplex', ionames=['io_out_Z_real', 'io_out_Z_imag'])

            self.rtlparameters=dict([ ('g_Rs', (float, self.Rs)), ]) #Freq for sim clock
            self.run_rtl()
        else:
            self.print_log(type='F', msg='Requested model currently not supported')
def gen_data(sig_type, sig_len, buffer_len):
    """ Method for simple signal generation
        -------
    """
    n_cycles = sig_len                        # How long the pulse is
    buffer_cycles = np.zeros(buffer_len)  # How long the zero datafeed after pulse is
    prebuffer_cycles = np.zeros(buffer_len)   # How long the zero datafeed before pulse is

    if sig_type == "Triangle":
        hp = int(n_cycles/2)
        data = np.concatenate((prebuffer_cycles, np.linspace(0, 1, hp), np.linspace(0, 1, hp)[::-1], buffer_cycles))
    elif sig_type == "Sine":
        sine_cycles = 2 # how many sine cycles
        sine_resolution = 200 # how many datapoints to generate
        sine_wave = np.sin(np.arange(0, np.pi * 2 * sine_cycles, np.pi * 2 * sine_cycles / sine_resolution))
        data = np.concatenate((prebuffer_cycles, sine_wave, buffer_cycles))
    elif sig_type == "Square":
        data = np.concatenate((prebuffer_cycles, np.ones(n_cycles*3), buffer_cycles))
    elif sig_type == "Impulse":
        data = np.concatenate((prebuffer_cycles, np.ones(1), buffer_cycles))

    return data


def calc_EVM(signal_gen, **kwargs):
    """ Method for printing/return EVM and plotting/return constellation points of testbecnch's signal

        Parameters
        ----------
        no_plot : binary (0,1) 
           if 0, constellation points are plotted.
           if 1, no plotting occures

        Example
        -------
        self.calc_EVM(no_plot=1)

    """

    text = f'EVM:{np.round(signal_gen.EVM[0][0]*100,2)}%'
    print(text)
    no_plot = 0
    if no_plot == 0:
        for i in range(0, signal_gen.BW.size):
           for j in range(0, len(signal_gen.BWP[i])):
               if signal_gen.EVM[i].any():
                   if signal_gen.EVM[i][j] != 0:
                       plt.figure()
                       plt.plot(signal_gen.rxDataSymbols[i][j].real, signal_gen.rxDataSymbols[i][j].imag, 'o')
                       plt.plot(signal_gen.cnstl[i][j].real, signal_gen.cnstl[i][j].imag, 'o')
                       plt.title("EVM = " + str(np.round(signal_gen.EVM[0][0] * 100, 2)) + "%")
                       plt.show(block=False)

    return signal_gen.EVM, signal_gen.cnstl, signal_gen.rxDataSymbols


def decimate(x,Fs, Fs_bb):
    pdb.set_trace()
    decim=Fs/Fs_bb
    decim2=Fs/Fs_bb
    fact=int(np.log2(decim)/1)
    order=100
    s_fil=x
    while decim>1:
        Fs=Fs/2
        decim=decim/2
        b=sig.remez(int(order+1),[0,Fs*0.24,Fs*0.28,0.5*Fs],[1,0],Hz=int(Fs))
        s_fil=np.convolve(s_fil.reshape((-1,1))[:,0],b,mode='same').reshape((-1,1))
        s_fil=s_fil[0::2]
    Fs=decim2
    t=np.arange(0,len(s_fil))/Fs
    s_out=np.transpose(np.vstack((np.real(s_fil[:,0]),np.imag(s_fil[:,0]))))
    # Normalize to 1
    test_matrix=np.concatenate((np.absolute(np.real(s_out)),np.absolute(np.imag(s_out))))
    max_value=test_matrix.max()
    s_out=s_out/max_value
    s_out=np.transpose(np.vstack((t,s_out[:,0],s_out[:,1])))
    return s_out


def plot_5G_output(vecs, signal_gen, output_list, descaling, modes,decim):
    Rs_bb = signal_gen.s_struct['Fs']

    for i in range(0, len(modes)):
        interpolation_factor = 2**i
        if decim == True:
            IQ_decim = decimate((output_list[i][0].T / descaling + 1j* output_list[i][1].T / descaling).reshape(-1,1), Rs_bb * interpolation_factor,Rs_bb)[:,1:]
        else: 
            I_decim=(output_list[i][0] /descaling)
            Q_decim=(output_list[i][1] /descaling)
            IQ_decim= np.transpose(np.vstack((I_decim,Q_decim)))

        signal_gen.IOS.Members['in_dem'].Data = IQ_decim

        signal_gen.run_dem()
        signal_gen.run_EVM()
            
        I_mod = output_list[i][0].T / descaling
        Q_mod = output_list[i][1].T / descaling

        calc_EVM(signal_gen, no_plot = 0)

        if ("I" in vecs): 
            _, ACLR = plot_PSD(signal=IQ_decim[:,0], Fc=0, Fs=Rs_bb, double_sided=True, f_span=10 * signal_gen.BW, BW=signal_gen.BW, zoom_plt=1, legend="decimated I"+str(i), PSD_min=-100, BW_conf=signal_gen.BW_conf, ACLR_BW=signal_gen.ACLR_BW, no_plot=0)
        if ("Q" in vecs):     
            _, ACLR = plot_PSD(signal=IQ_decim[:,1], Fc=0, Fs=Rs_bb, double_sided=True, f_span=10 * signal_gen.BW, BW=signal_gen.BW, zoom_plt=1, legend="decimated Q"+str(i), PSD_min=-100, BW_conf=signal_gen.BW_conf, ACLR_BW=signal_gen.ACLR_BW, no_plot=0)


if __name__=="__main__":
    import argparse
    import matplotlib.pyplot as plt
    from itertools import chain
    import plot_format

    import matplotlib.pyplot as plt
    from  hb_universal import *
    from  hb_universal.controller import controller as hb_universal_controller
    import pdb
    import math
    plot_format.set_style('isscc')

    #Setup sim controller
    hb_universal_controller = hb_universal_controller()
    hb_universal_controller.Rs = 200e9
    hb_universal_controller.reset()
    hb_universal_controller.step_time()
    hb_universal_controller.start_datafeed()

    #Sim controls
    isInteractive = [False,  True,  False,  False,  False] 
    isInteractive_decim = [False,  False,  False,  False,  False] 
    model = 'sv'
    lang='sv'

    interpolation_factor = 8
    resolution = 16 
    modes = [ 'bypass', 'two', 'four', 'eight', "more" ]
    simulation_type = 'interp' # interp , interp_decim
    input_list = [ ]
    output_list = [ ]
    output_list_decim = [ ]


    buffer_len = 150

    #Urc toolbox
    hb_universal_tk = hb_universal_tk()

    #Input, output and intermediate scaling
    scaling = (math.pow(2, resolution - 2) - 1)
    descaling = (math.pow(2, resolution - 2) - 1)
    block_scales = [4, 4, 4, 512]
    block_scales_decim = [2, 2, 2, 512]

 
    # 5G, Impulse, Sine, Square, Triangle
    sig_type = "5G"
    # Signals to be printed
    vecs = ["I"] 
    #If coeffs will be plotted
    plot_coeffs = False  
    #If plot sig FFT 
    plot_sig_fft = True

    if simulation_type == "interp":
        hb1_H = np.array(hb_universal_tk.load_H("../chisel/f2_interpolator/hb_interpolator/configs/hb1-config.yml"))
        hb2_H = np.array(hb_universal_tk.load_H("../chisel/f2_interpolator/hb_interpolator/configs/hb2-config.yml"))
        hb3_H = np.array(hb_universal_tk.load_H("../chisel/f2_interpolator/hb_interpolator/configs/hb3-config.yml"))
    else:    
        hb1_H = np.array(hb_universal_tk.load_H("../chisel/f2_decimator/hb_decimator/configs/hb1-config.yml"))
        hb2_H = np.array(hb_universal_tk.load_H("../chisel/f2_decimator/hb_decimator/configs/hb2-config.yml"))
        hb3_H = np.array(hb_universal_tk.load_H("../chisel/f2_decimator/hb_decimator/configs/hb3-config.yml"))

    #Plot coeffs and FFT
    if plot_coeffs:
        hb_universal_tk.plot_coeff_fft(hb1_H, hb2_H, hb3_H)
   

    if sig_type == "5G":
        #Setup Signal gen and preplot
        signal_gen = NR_signal_generator()
        signal_gen.QAM = "64QAM"
        signal_gen.osr = 1
        signal_gen.BWP = np.array([[[4,4,0,1]]]) #subcarrier spacing = my, 15*10^3 * 2^(my), OFDM symbols, Bandwith parts low to high 
        signal_gen.BW = np.array([150e6])
        signal_gen.in_bits = np.array([["max"]])
        signal_gen.Rs_bb = 0 #Baseband sample rate

        signal_gen.run_gen()

        Rs_bb = signal_gen.s_struct['Fs']
        signal_gen.IOS.Members['in_dem'] = signal_gen.IOS.Members['out']

        signal_gen.run_dem()
        signal_gen.run_EVM()

        calc_EVM(signal_gen, no_plot = 0)
        if ("I" in vecs):
            _, ACLR = plot_PSD(signal=signal_gen.IOS.Members['out'].Data[:, 0], Fc=0, Fs=Rs_bb, double_sided=True, f_span=10 * signal_gen.BW, BW=signal_gen.BW, zoom_plt=1, legend="Original I", PSD_min=-100, BW_conf=signal_gen.BW_conf, ACLR_BW=signal_gen.ACLR_BW, no_plot=0)
        if ("Q" in vecs):
            _, ACLR = plot_PSD(signal=signal_gen.IOS.Members['out'].Data[:, 1], Fc=0, Fs=Rs_bb, double_sided=True, f_span=10 * signal_gen.BW, BW=signal_gen.BW, zoom_plt=1, legend="Original Q", PSD_min=-100, BW_conf=signal_gen.BW_conf, ACLR_BW=signal_gen.ACLR_BW, no_plot=0)
        
        #Scale to resolution and add buffer zeros
        I_sig = (np.concatenate((np.floor(signal_gen.IOS.Members['out'].Data[:, 0] * scaling), np.zeros(buffer_len)))).astype(int)
        Q_sig = (np.concatenate((np.floor(signal_gen.IOS.Members['out'].Data[:, 1] * scaling), np.zeros(buffer_len)))).astype(int)
        
        vec_len = len(I_sig)

    else:
        #Signal params
        data = gen_data(sig_type, 4, buffer_len)

    for i in range(len(modes)):
        print("MODE: " + modes[i])
        dut_interp = hb_universal()
        dut_interp.model = model
        dut_interp.lang = lang
        dut_interp.runname = modes[i]
        dut_interp.interactive_rtl = isInteractive[i]

        if simulation_type == 'interp_decim':
            dut_decim = hb_universal()
            dut_decim.model = model
            dut_decim.lang = lang
            dut_decim.runname = modes[i]
            dut_decim.interactive_rtl = isInteractive_decim[i]
            dut_decim.IOS.Members["iptr_A"] =dut_interp.IOS.Members["Z"]


        interpolation_factor = 2**i

        #Input data
        if sig_type == "5G":
            input_data = np.repeat(np.insert((I_sig + 1j * Q_sig),0,np.zeros(100)), interpolation_factor)
            vec_len = len(input_data)
        else:
            scaled_data = np.repeat(np.rint(data * scaling), interpolation_factor)
            vec_len = len(scaled_data)
            input_data = (scaled_data + 1j * scaled_data)
            
        input_list.append(input_data)
        dut_interp.IOS.Members["iptr_A"].Data = input_data.reshape(-1, 1)

        rst = np.full(vec_len, 0)
        until_hb3_out_ready = (hb1_H.size * 2 + hb2_H.size * 2 + hb3_H.size * 2) * 4
        rst[:until_hb3_out_ready] = 1 
        dut_interp.IOS.Members["reset_loop"].Data = rst.reshape(-1, 1)

        dut_interp.IOS.Members["mode"].Data = np.full(vec_len, i).reshape(-1, 1)

        #These are constants
        dut_interp.IOS.Members["cic3shift"].Data = np.full(vec_len, 0).reshape(-1, 1)
        dut_interp.IOS.Members["convmode"].Data = np.full(vec_len, 0).reshape(-1, 1)

        if modes[i] in ["bypass","two" ]:    
            dut_interp.IOS.Members["ndiv"].Data = np.full(vec_len, 1).reshape(-1, 1)
            dut_interp.IOS.Members["hb1output_switch"].Data = np.full(vec_len,1).reshape(-1, 1)
            dut_interp.IOS.Members["hb2output_switch"].Data = np.full(vec_len,0).reshape(-1, 1)
            dut_interp.IOS.Members["hb3output_switch"].Data = np.full(vec_len,0).reshape(-1, 1)
        elif modes[i] in ["four" ]:    
            dut_interp.IOS.Members["ndiv"].Data = np.full(vec_len, 1).reshape(-1, 1)
            dut_interp.IOS.Members["hb1output_switch"].Data = np.full(vec_len,0).reshape(-1, 1)
            dut_interp.IOS.Members["hb2output_switch"].Data = np.full(vec_len,1).reshape(-1, 1)
            dut_interp.IOS.Members["hb3output_switch"].Data = np.full(vec_len,0).reshape(-1, 1)
        elif modes[i] in ["eight" ]:    
            dut_interp.IOS.Members["ndiv"].Data = np.full(vec_len, 1).reshape(-1, 1)
            dut_interp.IOS.Members["hb1output_switch"].Data = np.full(vec_len,0).reshape(-1, 1)
            dut_interp.IOS.Members["hb2output_switch"].Data = np.full(vec_len,0).reshape(-1, 1)
            dut_interp.IOS.Members["hb3output_switch"].Data = np.full(vec_len,1).reshape(-1, 1)
        else:
            dut_interp.IOS.Members["ndiv"].Data = np.full(vec_len, 2).reshape(-1, 1) 
            dut_interp.IOS.Members["hb1output_switch"].Data = np.full(vec_len,0).reshape(-1, 1)
            dut_interp.IOS.Members["hb2output_switch"].Data = np.full(vec_len,0).reshape(-1, 1)
            dut_interp.IOS.Members["hb3output_switch"].Data = np.full(vec_len,0).reshape(-1, 1)
        
        #Scales
        dut_interp.IOS.Members["hb1scale"].Data = np.full(vec_len, block_scales[0]).reshape(-1, 1)
        dut_interp.IOS.Members["hb2scale"].Data = np.full(vec_len, block_scales[1]).reshape(-1, 1)
        dut_interp.IOS.Members["hb3scale"].Data = np.full(vec_len, block_scales[2]).reshape(-1, 1)
        dut_interp.IOS.Members["cic3scale"].Data = np.full(vec_len, block_scales[3]).reshape(-1, 1)
 
        #These are clocks
        dut_interp.IOS.Members['control_write'] = hb_universal_controller.IOS.Members['control_write']     


        if simulation_type == 'interp_decim':
            rst = np.full(vec_len, 0)
            until_hb3_out_ready = (hb1_H.size * 2 + hb2_H.size * 2 + hb3_H.size * 2) * 4
            rst[:until_hb3_out_ready] = 1 
            dut_decim.IOS.Members["reset_loop"].Data = rst.reshape(-1, 1)

            dut_decim.IOS.Members["mode"].Data = np.full(vec_len, i).reshape(-1, 1)

            #These are constants
            dut_decim.IOS.Members["cic3shift"].Data = np.full(vec_len, 0).reshape(-1, 1)
            dut_decim.IOS.Members["convmode"].Data = np.full(vec_len, 1).reshape(-1, 1)

            if modes[i] in ["bypass","two" ]:    
                dut_decim.IOS.Members["ndiv"].Data = np.full(vec_len, 1).reshape(-1, 1)
                dut_decim.IOS.Members["hb1output_switch"].Data = np.full(vec_len,1).reshape(-1, 1)
                dut_decim.IOS.Members["hb2output_switch"].Data = np.full(vec_len,0).reshape(-1, 1)
                dut_decim.IOS.Members["hb3output_switch"].Data = np.full(vec_len,0).reshape(-1, 1)

            elif modes[i] in ["four" ]:    
                dut_decim.IOS.Members["ndiv"].Data = np.full(vec_len, 1).reshape(-1, 1)
                dut_decim.IOS.Members["hb1output_switch"].Data = np.full(vec_len,0).reshape(-1, 1)
                dut_decim.IOS.Members["hb2output_switch"].Data = np.full(vec_len,1).reshape(-1, 1)
                dut_decim.IOS.Members["hb3output_switch"].Data = np.full(vec_len,0).reshape(-1, 1)
            elif modes[i] in ["eight" ]:    
                dut_decim.IOS.Members["ndiv"].Data = np.full(vec_len, 1).reshape(-1, 1)
                dut_decim.IOS.Members["hb1output_switch"].Data = np.full(vec_len,0).reshape(-1, 1)
                dut_decim.IOS.Members["hb2output_switch"].Data = np.full(vec_len,0).reshape(-1, 1)
                dut_decim.IOS.Members["hb3output_switch"].Data = np.full(vec_len,1).reshape(-1, 1)

            else:
                dut_decim.IOS.Members["ndiv"].Data = np.full(vec_len, 2).reshape(-1, 1) 
                dut_decim.IOS.Members["hb1output_switch"].Data = np.full(vec_len,0).reshape(-1, 1)
                dut_decim.IOS.Members["hb2output_switch"].Data = np.full(vec_len,0).reshape(-1, 1)
                dut_decim.IOS.Members["hb3output_switch"].Data = np.full(vec_len,0).reshape(-1, 1)

            #Scales
            dut_decim.IOS.Members["hb1scale"].Data = np.full(vec_len, block_scales_decim[0]).reshape(-1, 1)
            dut_decim.IOS.Members["hb2scale"].Data = np.full(vec_len, block_scales_decim[1]).reshape(-1, 1)
            dut_decim.IOS.Members["hb3scale"].Data = np.full(vec_len, block_scales_decim[2]).reshape(-1, 1)
            dut_decim.IOS.Members["cic3scale"].Data = np.full(vec_len, block_scales_decim[3]).reshape(-1, 1)
     
            #These are clocks
            dut_decim.IOS.Members['control_write'] = hb_universal_controller.IOS.Members['control_write']


        dut_interp.run()

        #Read output to output_list
        sv_I = dut_interp.IOS.Members["Z"].Data.real[: ,0].astype("int16")
        sv_Q = dut_interp.IOS.Members["Z"].Data.imag[:, 0].astype("int16")

        
        if sig_type == "5G":
            #pdb.set_trace()
            trimmed_sv_I = np.trim_zeros(sv_I) #Trim zeros
            trimmed_sv_Q = np.trim_zeros(sv_Q) #Trim zeros
            trimmed_sv_Q = sv_Q #Trim zeros
            trimmed_sv_I = sv_I #Trim zeros

            longer = max(len(trimmed_sv_I), len(trimmed_sv_Q))

            #Concat the shorter to match the longer
            if longer == len(trimmed_sv_I):
                scaled_sv_I = trimmed_sv_I #/ np.amax(sv_I) #Normalization to one
                scaled_sv_Q = np.concatenate(( trimmed_sv_Q , np.zeros(longer - len(trimmed_sv_Q)))) #/ np.amax(sv_Q) #Normalization to one
            else:
                scaled_sv_I = np.concatenate(( trimmed_sv_I , np.zeros(longer - len(trimmed_sv_I)))) #/ np.amax(sv_I) #Normalization to one
                scaled_sv_Q = trimmed_sv_Q #/ np.amax(sv_Q) #Normalization to one
        else:
            scaled_sv_I = sv_I
            scaled_sv_Q = sv_Q

        #Append to list
        output_list.append([scaled_sv_I, scaled_sv_Q])



        if simulation_type == 'interp_decim':
            #pdb.set_trace()
            #sig=np.zeros(len(dut_decim.IOS.Members["iptr_A"].Data))
            #sig[int(len(sig)/2):int(len(sig)/2+interpolation_factor)]=1*scaling
            #dut_decim.IOS.Members["iptr_A"].Data=(sig+1j*sig).reshape(-1,1)
            dut_decim.run()
            #pdb.set_trace()
            sv_I = dut_decim.IOS.Members["Z"].Data.real[: ,0].astype("int16")[::interpolation_factor]
            sv_Q = dut_decim.IOS.Members["Z"].Data.imag[:, 0].astype("int16")[::interpolation_factor]

            #Append to list
            output_list_decim.append([sv_I, sv_Q])

           
    if sig_type != "5G":
        hb_universal_tk.plot_simple_signals(vecs, data, output_list, modes) 
    else:
        if simulation_type == 'interp_decim':
            plot_5G_output(vecs, signal_gen, output_list_decim, descaling, modes, False)
        else:
            plot_5G_output(vecs, signal_gen, output_list, descaling, modes, True)

    if plot_sig_fft:
        if ("I" in vecs):
            hb_universal_tk.plot_sig_fft("I", output_list[1:], modes[1:])

        if ("Q" in vecs):
            hb_universal_tk.plot_sig_fft("Q", output_list[1:], modes[1:])

    plt.show()

    #py = hb_universal_tk.FIR.calc_FIR(py_data.reshape(-1, 1)) / descaling
    #fg1 = axis[0, 1]          
    #axis[0, 1].plot(np.arange(0, len(py), 1), py, 'b-') 
    #axis[0, 1].set_title("FIR [py]")   
    #axis[0, 1].set_xlim(xlims_py)

    input()
