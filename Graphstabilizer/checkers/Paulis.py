def check_is_Paulistr(Pauli, raiseorFalse = 'raise'):
    '''
    Checks if the Pauli is a proper Pauli string, i.e. 'I', 'X', 'Y', 'Z', or a list of these.
    
    raiseorFalse is 'raise' or 'false' and decides if the function throws an exception or if it returns false.
    '''
    if not (raiseorFalse == 'raise') or (raiseorFalse == 'false'):
        raise ValueError(f"Input {raiseorFalse} should be either 'raise' or 'false'.")
    
    ## Iteratively go through all in list if list
    if type(Pauli) is list:
        for Paul in Pauli:
            check_Pauli(Paul)
        return
    
    if raiseorFalse == 'raise':
        if Pauli not in ['X','Y','Z']:
            raise ValueError(f"Pauli {Pauli} is not a string either 'I', 'X', 'Y' or 'Z'")
        else: return
    
    elif raiseorFalse == 'false':
        if Pauli not in ['X','Y','Z']:
            return False
        else: return True