import pennylane as qml
from pennylane import numpy as np
from qiskit import QuantumCircuit

def print_pennylane_circuit(f, params, n_wires, expand_params = False, qiskit = True, qsstyle = False):

    device = qml.device('default.qubit', wires = n_wires) # pennylane simulator
        
    @qml.qnode(device) 
    def pansatz(p):
        if expand_params:
            f(*p)
        else:
            f(p)
        return [qml.expval(qml.PauliZ(wires=i)) for i in device.wires] # dummy measurement, mandatory to print
        
    if qiskit:
        
        # Dibujamos el circuito con qiskit
        pansatz(params)
        qasm_circ = pansatz.qtape.to_openqasm()
        circuit = QuantumCircuit.from_qasm_str(qasm_circ)
        
        if qsstyle:
            from styles.style import qspain
            return circuit.draw('mpl', style= qspain())

        return circuit.draw('mpl')

    fig, _ = qml.draw_mpl(pansatz, style="default", fontsize = "medium")(params)
    return fig.show() # Dibujamos el circuito con pennylane

def print_pennylane_circuit2(n_wires, f, init_params, extra_params, expand_params = False, qiskit = True, qsstyle = False):

    device = qml.device('default.qubit', wires = n_wires) # pennylane simulator
 
    @qml.qnode(device) 
    def pansatz(n_wires, init_params, extra_params):

        if expand_params:
            f(n_wires, init_params, *extra_params)
        else:
            f(n_wires, init_params, extra_params)

        return [qml.expval(qml.PauliZ(wires=i)) for i in device.wires] # dummy measurement, mandatory to print
        
    if qiskit:
        
        # Dibujamos el circuito con qiskit
        pansatz(n_wires, init_params, extra_params)
        qasm_circ = pansatz.qtape.to_openqasm()
        circuit = QuantumCircuit.from_qasm_str(qasm_circ)
        
        if qsstyle:
            from styles.style import qspain
            return circuit.draw('mpl', style= qspain())
        
        return  circuit.draw('mpl')
            
    fig, _ = qml.draw_mpl(pansatz, style="default", fontsize = "medium")(n_wires, init_params, extra_params)
    return fig.show() # Dibujamos el circuito con pennylane

def convert_vqe_to_dict(job, program_id):
    
    d = {'job_id': str(job.__dict__['_job'].__dict__['_job_id']).encode("utf-8"),
         'backend': str(job.__dict__['_job'].__dict__['_backend']).encode("utf-8"),
         'program_id': str(program_id).encode("utf-8"),
         'success': str(job.__dict__['_job'].__dict__['_results']['success']).encode("utf-8"),
         'hamiltonian': str(job.__dict__['_job'].__dict__['_params']['hamiltonian']).encode("utf-8"),
         'optimizer': str(job.__dict__['_job'].__dict__['_params']['optimizer']).encode("utf-8"),
         'optimizer_config': str(job.__dict__['_job'].__dict__['_params']['optimizer_config']).encode("utf-8"),
         'shots': str(job.__dict__['_job'].__dict__['_params']['shots']).encode("utf-8"),
         'use_measurement_mitigation': str(job.__dict__['_job'].__dict__['_params']['use_measurement_mitigation']).encode("utf-8"),
         'params': str(job.__dict__['_job'].__dict__['_params']).encode("utf-8"),
         'intermediate_results': str(job.intermediate_results['function']).encode("utf-8"),
    }
    
    return d