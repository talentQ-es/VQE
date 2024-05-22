def ising_chain_ham_qibo(n, gam):
    
    # Esta función nos devuelve el Hamiltoniano en términos utilizables por los algoritmos de Qiskit
    # n = number of spin positions
    # gam = transverse field parameter
    
    import numpy as np
    from qibo import hamiltonians as ham, matrices as m
    
    for i in range(n):
        vecX = [m.I] * n
        vecZ = [m.I] * n
        vecX[i] = m.X
        vecZ[i] = m.Z
        
        if i == n - 1:
            vecX[0] = m.X
        else:
            vecX[i+1] = m.X

        auxX = vecX[0]
        auxZ = vecZ[0]
        
        for a in vecX[1:n]:
            auxX = np.kron(auxX, a)
        for b in vecZ[1:n]:
            auxZ = np.kron(auxZ, b)
            
        if i == 0:
            H = (auxX) + (gam * auxZ)
        else:
            H = H + (auxX) + (gam * auxZ)

    return ham.Hamiltonian(n, H)