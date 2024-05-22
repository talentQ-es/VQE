import qibo
import numpy as np
from qibo import models

class VQEQibo:
    
    def __init__(self, circuit = None, hamiltonian = None):
        
        self.circuit = circuit
        self.hamiltonian = hamiltonian
        self._optimizer = None
        self._func_callback = None
        self._options = None
        
        self._vqe = models.VQE(self.circuit, self.hamiltonian)
        
    def minimize(self, 
                 initial_parameters = None, 
                 callback = None, 
                 method = None, 
                 options = None, 
                 compile = False):
        
        return self._internal_minimize(initial_parameters, 
                                     callback = callback, 
                                     method = method, 
                                     options = options, 
                                     compile = compile,
                                     optimize = False)
        
    def _internal_minimize(self,
                 initial_parameters = None, 
                 callback = None, 
                 method = None, 
                 options = None, 
                 compile = False,
                 optimize = False):
        
        if self._optimizer is None:
            self._optimizer = method
            
        if self._func_callback is None: 
            self._func_callback = callback
        
        if self._options is None: 
            self._options = options
            
        if options and optimize:
            options['disp'] = False
            
        energy, params, extra = self._vqe.minimize(initial_parameters, 
                                               callback = callback if optimize else self._inner_callback, 
                                               method = method if optimize else self._optimizer, 
                                               options = options if optimize else self._options, 
                                               compile = compile)

        return energy, params, extra

    def _inner_callback(self, p):

        if self._func_callback is not None:

            energy, params, extra = self._internal_minimize(p, 
                                                         callback = None, 
                                                         method = self._optimizer, 
                                                         options = self._options, 
                                                         compile = False,
                                                         optimize = True)
            
            return self._func_callback(params, energy, extra, self._optimizer)