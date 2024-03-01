"""
=========
hb_universal
=========

hb_universal model with python and chisel

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
from dsp_toolkit import *
import plot_format
plot_format.set_style('isscc')

from hb_universal.hb_model import *

import matplotlib.pyplot as plt
import numpy as np
import math

class hb_universal(rtl, thesdk):
    @property
    def _classfile(self):
        return os.path.dirname(os.path.realpath(__file__)) + "/"+__name__

    def __init__(self,*arg): 
        self.print_log(type='I', msg='Inititalizing %s' %(__name__)) 
        self.proplist = [ 'Rs' ];   # Properties that can be propagated from parent
        self.Rs = 100e6;            # Sampling frequency
        self.IOS = Bundle()         # Pointer for input data

        self.IOS.Members["convmode"] = IO()
        self.IOS.Members["scale"] = IO()
        self.IOS.Members["output_switch"] = IO()

        self.IOS.Members["iptr_A"] = IO()
        self.IOS.Members["Z"] = IO()

        self.IOS.Members['control_write']= IO()
        self.IOS.Members["clock_slow"] = IO() 
        self.IOS.Members["clock"] = IO()
        self.IOS.Members["reset"] = IO()

        self.py_model = hb_model()

        self.model = 'py';             # Can be set externally, but is not propagated
        self.par = False              # By default, no parallel processing
        self.queue = []               # By default, no parallel processing

        if len(arg) >= 1:
            parent = arg[0]
            self.copy_propval(parent, self.proplist)
            self.parent  = parent;



    def init(self, **kwargs):
        """ Method for initializing the HB, these values can be changed manually afterwards 

            Parameters
            ----------
            model: string
                py       = Python
                sv       = SystemVerilog

            lang: string
                py       = Python
                sv       = SystemVerilog

            conversion_mode: string
                interp   = interpolation mode
                decim    = decimation mode

            conversion_factor: integer(0, 2, 4, 8, 16, ...)
                conversion factor, decides mode

            is_interactive: bool
                set if the simulation in interactive

            Example
            -------
            self.init()

        """
        model = kwargs.get('model', 'py')
        lang = kwargs.get('lang', 'py')
        conversion_mode = kwargs.get('conversion_mode', 'interp')
        conversion_factor = kwargs.get('conversion_factor', 16)
        is_interactive = kwargs.get('is_interactive', False)
            
        interp_scale = 2
        decim_scale = 1

        self.model = model
        self.lang = lang
        self.interactive_rtl = is_interactive

        self.IOS.Members["output_switch"].Data = np.array([[0]])

        #These are constants
        if conversion_mode == 'interp':
            self.runname = str(conversion_factor) + "_interp"

            self.IOS.Members["convmode"].Data = np.array([[0]])

            #Scales
            self.IOS.Members["scale"].Data = np.array([[interp_scale]])
        else:
            self.runname = str(conversion_factor) + "_decim"

            #These are constants
            self.IOS.Members["convmode"].Data = np.array([[1]])

            #Scales
            self.IOS.Members["scale"].Data = np.array([[decim_scale]])


    def main(self):
        '''For running the PY model
        '''
        self.IOS.Members["Z"].Data = self.py_model.interpolation(self.IOS.Members["iptr_A"].Data)


    def define_io_conditions(self):
        """When iofiles start to roll
        """
        self.iofile_bundle.Members['clock_slow'].rtl_io_condition='initdone'

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
            _=rtl_iofile(self, name='scale', dir='in', iotype='sample', datatype='int', ionames=['io_in_scale'])
            _=rtl_iofile(self, name='output_switch', dir='in', iotype='sample', datatype='int', ionames=['io_in_output_switch'])
            _=rtl_iofile(self, name='convmode', dir='in', iotype='sample', datatype='int', ionames=['io_in_convmode'])
            _=rtl_iofile(self, name='clock_slow', dir='in', iotype='sample', datatype='int', ionames=['io_clock_slow'])
            _=rtl_iofile(self, name='iptr_A', dir='in', iotype='sample', datatype='scomplex', ionames=['io_in_iptr_A_real', 'io_in_iptr_A_imag'])

            #Outputs
            _=rtl_iofile(self, name='Z', dir='out', iotype='sample', datatype='scomplex', ionames=['io_out_Z_real', 'io_out_Z_imag'])

            self.rtlparameters=dict([ ('g_Rs', (float, self.Rs)), ]) #Freq for sim clock
            self.run_rtl()
        else:
            self.print_log(type='F', msg='Requested model currently not supported')


if __name__=="__main__":
    import argparse
    from  hb_universal import *
    from  hb_universal.controller import controller as dut_controller

    dsp_tk = dsp_toolkit()

    # General params
    dut_model = 'sv'
    conversion_mode = 'interp'
    signal_type = '5G'
    modes = [16]

    # Interactive control        
    is_interactive_interp = 0 
    is_interactive_decim = 0 

    # Plot toggles
    vecs = ["I"]
    plot_fft = 0
    plot_evm = 0
    plot_psd = 0
    plot_raw = 1

    # Params for DUT
    resolution = 16
    conversion_factor = 2**1

    # Params for signal generation
    signal_gen = None
    QAM = "64QAM"
    BWP = [[[4,4,0,1]]]
    osr = 1
    BW = [200e6]
    in_bits = [["max"]]
    Rs_bb = 0

    prebuffer_len = 100
    buffer_len = 151
    signal_len = 4
    scaling = (math.pow(2, resolution - 2) - 1)

    ### Initial values
    #For plotting
    inputs = [ ]
    interp_outputs = [ ]
    interp_decim_outputs = [ ]

    #Setup sim controller
    dut_controller = dut_controller()
    dut_controller.Rs = 200e9
    dut_controller.reset()
    dut_controller.step_time()
    dut_controller.start_datafeed()

    if signal_type == "5G":
        #Setup Signal gen and preplot
        signal_gen, I, Q = dsp_tk.init_NR_siggen(vecs=vecs, \
                                                 QAM=QAM, \
                                                 osr=osr, \
                                                 BWP=BWP, \
                                                 BW=BW, \
                                                 in_bits=in_bits, \
                                                 Rs_bb=Rs_bb, \
                                                 lz=prebuffer_len, \
                                                 tz=buffer_len, \
                                                 plot_evm=plot_evm, \
                                                 plot_psd=plot_psd)            
        I_scaled = np.floor(I * scaling)
        Q_scaled = np.floor(Q * scaling)
        data = (I_scaled + 1j * Q_scaled) 
    else:
        #Signal params
        simple = dsp_tk.gen_simple_signal(signal_type, \
                                          signal_len, \
                                          buffer_len)
        data = np.floor(simple * scaling)

    interp = hb_universal()
    interp.init(model=dut_model, \
                lang=dut_model, \
                conversion_mode='interp', \
                conversion_factor=conversion_factor, \
                is_interactive=is_interactive_interp == 1)

    if conversion_mode == 'interp_decim':
        decim = hb_universal()
        decim.init(model=dut_model, \
                   lang=dut_model, \
                   conversion_mode='decim', \
                   conversion_factor=conversion_factor, \
                   is_interactive=is_interactive_decim == 1)

        decim.IOS.Members["iptr_A"] = interp.IOS.Members["Z"]
    
    #Input data
    if dut_model == 'sv':
        if signal_type == "5G":
            input_data = np.repeat(data, conversion_factor)
        else:
            input_data = np.repeat((data + 1j * data), conversion_factor)

    elif dut_model == 'py':
        scaled_data = (np.rint(data * scaling)).astype(int)
        if signal_type == "5G":
            input_data = scaled_data
        else:
            input_data = (scaled_data + 1j * scaled_data)

    interp.IOS.Members["iptr_A"].Data = input_data.reshape(-1, 1)

    inputs.append(input_data)

    #Setup controller
    interp.IOS.Members['clock_slow'].Data = np.resize([1,0], len(input_data)).reshape(-1, 1)
    interp.IOS.Members['control_write'] = dut_controller.IOS.Members['control_write']     

    ## Run interp
    interp.run()

    #Read output to output_list
    sv_I = interp.IOS.Members["Z"].Data.real[:,0].astype("int16") / scaling
    sv_Q = interp.IOS.Members["Z"].Data.imag[:,0].astype("int16") / scaling
    
    #Append to list
    interp_outputs.append([sv_I, sv_Q, conversion_factor])

    if conversion_mode == 'interp_decim':
        #Setup controller
        decim.IOS.Members['clock_slow'].Data = np.resize([1,0], len(input_data)).reshape(-1, 1)
        decim.IOS.Members['control_write'] = dut_controller.IOS.Members['control_write']

        ## Run decim
        decim.run()

        #Read output to output_list
        sv_I = decim.IOS.Members["Z"].Data.real[:,0].astype("int16")[::conversion_factor] / scaling
        sv_Q = decim.IOS.Members["Z"].Data.imag[:,0].astype("int16")[::conversion_factor] / scaling

        #Append to list
        interp_decim_outputs.append([sv_I, sv_Q, conversion_factor])

    ### Plotting
    if signal_type == "5G":
        dsp_tk.plot_5G_signals(vecs=vecs, \
                                    convmode=conversion_mode, \
                                    modes=modes, \
                                    signal_gen=signal_gen, \
                                    outputs=interp_outputs, \
                                    Fc=0, \
                                    plot_evm=plot_evm, \
                                    plot_psd=plot_psd, \
                                    plot_raw=plot_raw)
        if conversion_mode == 'interp_decim':
            dsp_tk.plot_5G_signals(vecs=vecs, \
                                        convmode=conversion_mode, \
                                        modes=modes, \
                                        signal_gen=signal_gen, \
                                        outputs=interp_decim_outputs, \
                                        Fc=0, \
                                        plot_evm=plot_evm, \
                                        plot_psd=plot_psd, \
                                        plot_raw=plot_raw)
    else:
        if conversion_mode == 'interp_decim':
            plot_data = interp_decim_outputs  
        else:
            plot_data = interp_outputs                            
        dsp_tk.plot_simple_signals(vecs, conversion_mode, modes, data, 1, plot_data) 
        
    if plot_fft:
        for vec in vecs:
            if conversion_mode == 'interp_decim':
                plot_data = interp_decim_outputs  
            else:
                plot_data = interp_outputs                            
            dsp_tk.plot_sig_fft(vec=vec, modes=modes, output=plot_data)

    plt.show()
    input()
