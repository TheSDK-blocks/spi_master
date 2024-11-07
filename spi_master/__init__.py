"""
=========
spi_master
=========

spi_master model with python and chisel

"""

import os
import sys
import pdb

if not (os.path.abspath("../../thesdk") in sys.path):
    sys.path.append(os.path.abspath("../../thesdk"))

sys.path.append('toolkit/')

from thesdk import *
from rtl import rtl, rtl_iofile, rtl_connector_bundle

import matplotlib.pyplot as plt
import numpy as np
import math


class spi_master(rtl, thesdk):
    @property
    def _classfile(self):
        return os.path.dirname(os.path.realpath(__file__)) + "/"+__name__

    def __init__(self,*arg):
        """HB parameters and attributes"""
        self.print_log(type='I', msg='Initializing %s' %(__name__))
        self.IOS.Members["cpol"] = IO()
        self.IOS.Members["cpha"] = IO()
        self.IOS.Members["mosi"] = IO()

        self.IOS.Members["miso"] = IO()
        self.IOS.Members["sclk"] = IO()
        self.IOS.Members["cs"] = IO()
        
        # Removed ready-valid handshake signals
        # self.IOS.Members["masterData_ready"] = IO()
        # self.IOS.Members["masterData_valid"] = IO()
        # self.IOS.Members["masterData_bits"] = IO()
        # self.IOS.Members["slaveData_ready"] = IO()
        # self.IOS.Members["slaveData_valid"] = IO()
        # self.IOS.Members["slaveData_bits"] = IO()

        # Added new IO signals without handshake
        self.IOS.Members["slaveData"] = IO()        # Input data to transmit
        self.IOS.Members["slaveDataValid"] = IO()   # Indicates when slaveData is valid (optional)
        self.IOS.Members["masterData"] = IO()       # Output data received
        

        self.IOS.Members['control_write']= IO()
        
        self.IOS.Members["clock"] = IO()
        self.IOS.Members["reset"] = IO()

        self.py_model = None

        self.proplist = [ 'Rs' ];   # Properties that can be propagated from parent
        self.Rs = 100e6;            # Sampling frequency

        self.model = 'py';             # Can be set externally, but is not propagated
        self.par = False              # By default, no parallel processing
        self.queue = []               # By default, no parallel processing

        if len(arg) >= 1:
            parent = arg[0]
            self.copy_propval(parent, self.proplist)
            self.parent  = parent;

    def init(self, **kwargs):
        return

    def main(self):
        '''For running the PY model
        '''
        return

    def define_io_conditions(self):
        """When iofiles start to roll
        """
        # Inputs roll after initdone
        self.iofile_bundle.Members['cpol'].rtl_io_condition='initdone'
        self.iofile_bundle.Members['cpha'].rtl_io_condition='initdone'
        self.iofile_bundle.Members['miso'].rtl_io_condition='initdone'
        self.iofile_bundle.Members['slaveData'].rtl_io_condition='initdone'
        self.iofile_bundle.Members['slaveDataValid'].rtl_io_condition='initdone'

        # Outputs roll after initdone and default conds
        self.iofile_bundle.Members['mosi'].rtl_io_condition_append(cond='&& initdone')
        self.iofile_bundle.Members['sclk'].rtl_io_condition_append(cond='&& initdone')
        self.iofile_bundle.Members['masterData'].rtl_io_condition_append(cond='&& initdone')
        self.iofile_bundle.Members['cs'].rtl_io_condition_append(cond='&& initdone')
        

    def run(self,*arg):
        '''Guideline: Define model dependencies of executions in `run` method.
        '''
        if self.model == "py":
            self.main()
        elif self.model in [ 'sv', 'icarus' ]:
            # Inputs
            _=rtl_iofile(self, name='cpol', dir='in', iotype='sample', datatype='int', ionames=['io_cpol'])
            _=rtl_iofile(self, name='cpha', dir='in', iotype='sample', datatype='int', ionames=['io_cpha'])
            _=rtl_iofile(self, name='mosi', dir='out', iotype='sample', datatype='int', ionames=['io_mosi'])
            _=rtl_iofile(self, name='miso', dir='in', iotype='sample', datatype='int', ionames=['io_miso'])
            _=rtl_iofile(self, name='sclk', dir='out', iotype='sample', datatype='int', ionames=['io_sclk'])
            _=rtl_iofile(self, name='cs', dir='out', iotype='sample', datatype='int', ionames=['io_cs'])

            # Updated IO files without handshake
            _=rtl_iofile(self, name='slaveData', dir='in', iotype='sample', datatype='int', ionames=['io_slaveData'])
            _=rtl_iofile(self, name='slaveDataValid', dir='in', iotype='sample', datatype='int', ionames=['io_slaveDataValid'])
            _=rtl_iofile(self, name='masterData', dir='out', iotype='sample', datatype='int', ionames=['io_masterData'])
           

            # Outputs
            self.rtlparameters=dict([ ('g_Rs', (float, self.Rs)), ]) # Freq for sim clock
            self.run_rtl()
        else:
            self.print_log(type='F', msg='Requested model currently not supported')

if __name__=="__main__":
    import argparse
    from  spi_master import *
    from  spi_master.controller import controller as dut_controller
    from  spi_slave import *
    from  spi_slave.controller import controller as dut1_controller
    import sys

    sys.path.insert(0, '/home/pro/masters/ntabatab/a-core_thesydekick/Entities/spi_slave')

    # General params
    dut_model = 'sv'
    dut_lang = 'sv'
  
    # Interactive control        
    is_interactive = True
    
    # Setup sim controller
    dut_controller = dut_controller()
    dut_controller.Rs = 200e9
    dut_controller.reset()
    dut_controller.step_time()
    dut_controller.start_datafeed()

    dut1_controller = dut1_controller()
    dut1_controller.Rs = 200e9
    dut1_controller.reset()
    dut1_controller.step_time()
    dut1_controller.start_datafeed()

    dut = spi_master()
    dut.model=dut_model
    dut.lang=dut_lang
    dut.interactive_rtl=False

    dut1 = spi_slave()
    dut1.model=dut_model
    dut1.lang=dut_lang
    dut1.interactive_rtl=is_interactive

    dut.IOS.Members["cpol"].Data = np.array([[0]])  ##from master to core 
    #pdb.set_trace()
    dut1.IOS.Members["cs"] = dut.IOS.Members["cs"].Data   # Master `cs` to Slave `cs` 
    dut.IOS.Members["cpha"].Data = np.array([[0]])   ##from master to core
    dut.IOS.Members["miso"].Data = np.array([[0]])  
    #dut.IOS.Members["miso"] = dut1.IOS.Members["miso"]  # Slave `miso` to Master `miso`
    dut.IOS.Members["slaveDataValid"].Data = np.array([[1]])
    dut.IOS.Members["slaveData"].Data = np.array([[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,10,20,30,40,50,60,70,80,90,100,110,10,20,30,40,50,60,70,80,1,2,3,4,5,6,7,7,7,4,89,9,0,0,6,17]])
    dut1.IOS.Members["sclk"].Data = dut.IOS.Members["sclk"].Data   # Master `sclk` to Slave `sclk`
    dut1.IOS.Members["mosi"].Data = dut.IOS.Members["mosi"].Data   # Master `mosi` to Slave `mosi`
    dut1.IOS.Members["monitor_in"].Data = np.array([[1,0,1,0,1,0,1,0,1,0,1,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]])

    dut.IOS.Members['control_write'] = dut_controller.IOS.Members['control_write']    
    dut1.IOS.Members['control_write'] = dut1_controller.IOS.Members['control_write']    
    # Run simulation
    dut.run()
    pdb.set_trace()
    dut1.run()

    input()
