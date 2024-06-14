"""
Inspired by: https://github.com/WBanner/Test-VQE-Repository
Created by: QuantumSpain (https://quantumspain-project.es/)
"""
from qiskit.algorithms.optimizers.optimizer import Optimizer, OptimizerSupportLevel, OptimizerResult
import numpy as np
from typing import Dict, Any

class Rotosolve(Optimizer):
    """Create an instance of the Rotosolve optimizer.
    Parameters
    ----------
    max_steps : int
        Maximum number of steps to take in the optimizer. This is the number
        of times to loop through all the parameters in the objective function.
    alt_convention : bool
        The Rotosolve paper uses the convention that there is a `1/2`
        in the exponent. This convention corresponds to `alt_convention=False`.
        In other algorithms this factor is
        not present, in which case, use `alt_convention=True`.
    step_size : int
        The step count in the value optimization loop.
    See: https://arxiv.org/abs/1905.09692
    """

    CONFIGURATION = {
        'name': 'RotoSolve',
        'description': 'RotoSolve Optimizer',
        'input_schema': {
            '$schema': 'http://json-schema.org/schema#',
            'id': 'GP_schema',
            'type': 'object',
            'properties': {},
            'additionalProperties': False
        },
        'support_level': {
            'gradient': OptimizerSupportLevel.ignored,
            'bounds': OptimizerSupportLevel.ignored,
            'initial_point': OptimizerSupportLevel.supported
        },
        'options': [],
        'optimizer': [],
        'maxiter': 1024
    }

    def __init__(self, max_steps: int = 10, step_size: int = 1, alt_convention: bool = False):
        
        super().__init__()
        
        self._max_steps = max_steps
        self._step_size = step_size
        
        # alt_convention == False:
        # e^{-i H theta}
        # alt_convention == True
        # e^{-i H theta/2}
        self._alt_convention = alt_convention

    def optimize(self, num_vars, obj_function, gradient_function=None, variable_bounds=None, initial_point=None):
        """See `qiskit.algorithms.optimizers.optimizer.Optimizer` documentation."""
        super().optimize(num_vars, obj_function, gradient_function, variable_bounds, initial_point)

        if initial_point is None:
            initial_point = np.random.uniform(-np.pi, +np.pi, num_vars)
            
        return self.minimize(fun = obj_function, x0 = initial_point)

    def minimize(self, fun, x0, bounds = None, jac = None):
    
        factor = 2 if self._alt_convention else 1

        def f(x):
            return fun(x / factor)
        
        theta_min, f_min, f_evals, f_values =  self._rotosolve(f, x0, self._max_steps, self._step_size)
        
        self.energy_values = f_values
        
        result = OptimizerResult()
        result.x = theta_min # optimal parameters
        result.fun = f_min # optimal function value
        result.nfev = f_evals

        return result
    
    @staticmethod
    def _rotosolve(f, initial_point: np.array, max_steps: int, step_size: int):
        
        D = len(initial_point)
        theta = initial_point
        f_evals = 0
        
        def f_counter(*args, **kwargs):
            return f(*args, **kwargs)

        f_current = f_counter(initial_point)
        f_evals += 1
        
        converged = False
        steps = 0

        theta_values = []
        f_values = []
        
        print("Rotosolve algorithm for optimizing a given value")
        print("================================================")
        
        while not converged:
            
            for d in range(D):
                
                phi = np.random.uniform(-np.pi, +np.pi)
                theta_d = phi
                theta[d] = theta_d
                m_vals = {
                    'phi+0': 0,
                    'phi+pi/2': 0,
                    'phi-pi/2': 0
                }

                m_vals['phi+0'] = f_counter(theta)
                f_evals += 1
                theta[d] = theta[d] + np.pi/2
                
                m_vals['phi+pi/2'] = f_counter(theta)
                f_evals += 1
                theta[d] = theta[d] - np.pi
                
                m_vals['phi-pi/2'] = f_counter(theta)
                f_evals += 1
                theta[d] = phi - np.pi/2 - np.arctan2(
                    2*m_vals['phi+0'] - m_vals['phi+pi/2'] - m_vals['phi-pi/2'], 
                    m_vals['phi+pi/2'] - m_vals['phi-pi/2']
                )

                phi = 0
                theta_d = 0

            theta_values.append(theta)

            expectation_value = f(theta)
            f_values.append(expectation_value)
            
            print('Step {step}. Current expectation value: {ev: .8f}'.format(step = steps, ev = expectation_value))
            
            steps += step_size
            
            if steps >= max_steps:
                converged = True

        f_current = f_counter(theta)
        f_evals += 1

        min_index = np.argmin(f_values)
        f_min = f_values[min_index]
        theta_min = theta_values[min_index]

        return theta_min, f_min, f_evals, f_values
    
    @property
    def settings(self) -> Dict[str, Any]:
        conf = self.CONFIGURATION
        return conf

    def estimate_stddev(self):
        """Estimate the standard deviation."""
        ar = np.array(self.energy_values)
        x = np.abs(ar - ar.mean())**2
        std = np.sqrt(np.sum(x) / len(ar))
    
        return std

    def get_support_level(self):
        """Get the support level dictionary."""
        return self.CONFIGURATION['support_level']