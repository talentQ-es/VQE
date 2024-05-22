# The following functions are used to perform the Jordas-Wigner transformation on the fermionic operators
# The transformation is done analitically and without using matrix computation, which gives us a higher computation speed
# The code is prepared only for single and double excitations

import numpy as np
from functools import reduce
from itertools import product

PAULIS = {'X': np.array([[0,1],[1,0]]), 'Y': np.array([[0,-1j],[1j,0]]), 'Z': np.array([[1,0],[0,-1]]), 'I': np.array([[1,0],[0,1]])}

def PauliSymbolProd(S1, S2):

    # S1 and S2 are tuples of the type (coefficient, "Pauli Operator")
    # The coefficient is/can be a complex number
    # The Pauli operator is a string that must be 'I', 'X', 'Y' or 'Z'.
    # The output is a tuple of the same type (coefficient, "Pauli Operator")
    # The output coefficient and operator represent the result of the matrix dot product S1·S2

    sign = +1.0

    if type(S1) is tuple:
        sign *= S1[0]
        S1 = S1[1]

    if type(S2) is tuple:
        sign *= S2[0]
        S2 = S2[1]

    Paulis = ['I', 'X', 'Y', 'Z']

    assert S1 in Paulis, "Symbol S1 does not correspond with a Pauli Matrix"
    assert S2 in Paulis, "Symbol S2 does not correspond with a Pauli Matrix"
    
    A = Paulis.index(S1)
    B = Paulis.index(S2)

    if A == B:
        return (+1.0*sign, Paulis[0])
    elif A*B == 0:
        return (+1.0*sign, Paulis[A]) if B==0 else (+1.0*sign, Paulis[B])   
    else:
        return ((-1)**(1 + int(B-A+0.1)) * sign * 1j, Paulis[6 - A - B])
    

def PauliChain(C1,C2):

    # C1 and C2 are tuples of the type (coefficient, "Pauli Chain")
    # The coefficient is/can be a complex number
    # The Pauli Chain is a string that must be composed by symbols 'I', 'X', 'Y' or 'Z'.
    # The output is a tuple of the same type (coefficient, "Pauli Chain")
    # The output coefficient and operator represent the tensor product C1⊗C2

    sign = +1.0

    if type(C1) is tuple:
        sign *= C1[0]
        C1 = C1[1]

    if type(C2) is tuple:
        sign *= C2[0]
        C2 = C2[1]

    return sign, ''.join([C1,C2])
        

def JWelement_up(el, pos):

    # Given a fermionic CREATION operator a^{\dagger}_{pos}, ince we need to go through a chain of qubits, this generates an output consisting of two tuples
    # They can be either I, (X-iY)/2 or Z
    # This depend on the relative situation of the qubit we are analyzing with respect to the position in which we are creating a particle

    if pos > el:
        return [(+1.0, 'I'),(+1.0, 'I')]
    elif pos < el:
        return [(+1.0, 'Z'),(+1.0, 'Z')]
    else:
        return [(+0.5, 'X'), (-0.5j, 'Y')]
    
    
def JWelement_down(el, pos):

    # Given a fermionic DESTRUCTION operator a^{\dagger}_{pos}, ince we need to go through a chain of qubits, this generates an output consisting of two tuples
    # They can be either I, (X+iY)/2 or Z
    # This depend on the relative situation of the qubit we are analyzing with respect to the position in which we are destructing a particle

    if pos > el:
        return [(+1.0, 'I'),(+1.0, 'I')]
    elif pos < el:
        return [(+1.0, 'Z'),(+1.0, 'Z')]
    else:
        return [(+0.5, 'X'), (+0.5j, 'Y')]
    

def JW(ups, downs, num_qubits, coeff):

    # Given a single excitation, which is an operator of the form a+ a, this generates the dictionary of Pauli Chains
    # with the appropriate coefficients for a JW transformation.

    paulistrings = []

    sol = {}

    for actual_qubit in reversed(range(num_qubits)): # Using reversed here results in the qubit ordering used by Qiskit
        elsu = [JWelement_up(actual_qubit, up) for up in ups]

        elsd = [JWelement_down(actual_qubit, down) for down in downs]

        paulistrings.append(list(product(*elsu,*elsd)))

    for position in range(2**(len(ups)+len(downs))):
        tag = []
        for qubit in range(num_qubits):
            tag.append(reduce(PauliSymbolProd,paulistrings[qubit][position]))
        act = reduce(PauliChain, tag)
        sol.update({act[1]: act[0]*coeff})

    return sol


def jordanwigner(dictionary, num_qubits):

    # Given a dictionary of single and double excitation fermionic operators and the number of qubits we need for our circuit,
    # this function performs the needed JW transformation, resulting in a dictionary with the Pauli Chains for the whole excitation

    # The keys of the dictionary is adapted to work with the one given by our FermionicOp class:
    # '+_u +_u -_d -_d'
    # Where the 'u's are the positions where we create particles and the 'd's the positions where we destruct

    transdict = {}

    for tag in list(dictionary.keys()):

        ops = tag.split(" ")   

        u = []
        d = []

        for op in ops:

            s,n = op.split("_")
            n = int(n)

            if s == '+':
                u = u + [n]
            elif s == '-':
                d = d + [n]

        new = JW(u, d, num_qubits, dictionary[tag])

        transdict = {key: new.get(key, 0) + transdict.get(key, 0) for key in set(new) | set(transdict)}

    transdict = {key:val for key, val in transdict.items() if abs(val) != 0}

    return transdict


def matrix_jw(dictionary,num_qubits):

    fin = np.zeros([2**num_qubits,2**num_qubits])

    for tag in list(dictionary.keys()):

        mat = dictionary[tag]

        for symm in tag:

            mat = np.kron(mat, PAULIS[symm])

        fin = fin + mat

    return fin