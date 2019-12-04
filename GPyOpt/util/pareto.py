import numpy as np

def ParetoFront(P):
    #################
    # From "A Fast Algorithm for Finding the Non Dominated Set in Multi objective Optimization" 
    #################
    O = np.array(sorted(P, key=lambda x: x[0]))
    S = np.array([O[0]])
    O = np.delete(O,0,0)
    
    for n in range(len(O)):
        M = S < O[0]
        Cp_all = []
        Cp_any = []
        
        for x in M:
            Cp_aux = np.all(x)
            Cp_aux2 = np.any(x)
            Cp_all.append(Cp_aux)
            Cp_any.append(Cp_aux2)
        
        if (np.any(Cp_all)==True):
            O = np.delete(O,0,0)

        elif (np.all(Cp_any)==False):
            Z = np.argwhere(np.array(Cp_any) == False)[::-1]
            for z in Z:
                S = np.delete(S, z[0], 0)
            S = np.vstack((S, O[0]))
            O = np.delete(O,0,0)

        else:
            S = np.vstack((S, O[0]))
            O = np.delete(O,0,0)

        if (S.size == 0):
            S = np.append(S, O[0])
            O = np.delete(O,0,0)
    return S