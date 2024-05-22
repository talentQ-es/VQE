"""
Created by: QuantumSpain (https://quantumspain-project.es/)
"""
def ising_chain_ham(n, gam, pennylane = False):
    
    # Esta función nos devuelve el Hamiltoniano en términos utilizables por los algoritmos de Qiskit
    # n = number of spin positions
    # gam = transverse field parameter
    if pennylane:
        import pennylane as qml
        from pennylane import numpy as np
    
    from qiskit.opflow import X, Z, I
        
    for i in range(n):
        vecX = [I] * n
        vecZ = [I] * n
        vecX[i] = X
        vecZ[i] = Z
        
        if i == n - 1:
            vecX[0] = X
        else:
            vecX[i+1] = X

        auxX = vecX[0]
        auxZ = vecZ[0]
        
        for a in vecX[1:n]:
            auxX = auxX ^ a
        for b in vecZ[1:n]:
            auxZ = auxZ ^ b
            
        if i == 0:
            H = (auxX) + (gam * auxZ)
        else:
            H = H + (auxX) + (gam * auxZ)

    if pennylane:
        h_matrix = np.matrix(H.to_matrix_op().primitive.data)
        return qml.pauli.pauli_decompose(h_matrix)
    
    return H
