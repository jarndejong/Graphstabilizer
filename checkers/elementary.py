### Input checker functions
def check_is_naturalnr(number, raiseorFalse = 'raise'):
    '''
    Checks if the input is a valid natural number, i.e. an integer larger than 0.
    
    raiseorFalse is 'raise' or 'false' and decides if the function throws an exception or if it returns false.
    '''
    if not (raiseorFalse == 'raise') or (raiseorFalse == 'false'):
        raise ValueError(f"Input {raiseorFalse} should be either 'raise' or 'false'.")
    
    if raiseorFalse == 'raise':
        if not type(number) is int:
            raise TypeError(f"Number parameter is type {type(number)}, not 'int64' or 'int32'.")
        else: return
    
    elif raiseorFalse == 'false':
        if not type(number) is int:
            return False
        else: return True
    
def check_is_node_index(size, node, raiseorFalse = 'raise'):
    '''
    Checks if the node index is in the allowed range, or of it's a list of valid indices.
    
    raiseorFalse is 'raise' or 'false' and decides if the function throws an exception or if it returns false.
    '''
    if not (raiseorFalse == 'raise') or (raiseorFalse == 'false'):
        raise ValueError(f"Input {raiseorFalse} should be either 'raise' or 'false'.")
    
    # Iteratively go through all in list if list
    if type(node) is list:
        for nod in node:
            check_is_node_index(size, nod, raiseorFalse = raiseorFalse)
        return
    
    check_is_naturalnr(node+1, raiseorFalse = 'raise')
    
    if raiseorFalse == 'raise':
        if node < 0 or node >= size:
            raise ValueError(f"{node} is not an index of {size}-sized list.")
        else: return
    
    elif raiseorFalse == 'false':
        if node < 0 or node >= size:
            return False
        else: return True
    
def check_is_Boolvar(boolvar, raiseorFalse = 'raise'):
    '''
    Checks if the boolvar is indeed a boolean.
    
    raiseorFalse is 'raise' or 'false' and decides if the function throws an exception or if it returns false.
    '''
    if not (raiseorFalse == 'raise') or (raiseorFalse == 'false'):
        raise ValueError(f"Input {raiseorFalse} should be either 'raise' or 'false'.")
    
    if raiseorFalse == 'raise':
        if not type(boolvar) is bool:
            raise TypeError('Variable is {type(boolvar)}, not bool.')
        else: return
    
    elif raiseorFalse == 'false':
        if not type(boolvar) is bool:
            return False
        else: return True

def check_is_list(l, raiseorFalse = 'raise'):
    '''
    Checks if the input l is a list.
    
    raiseorFalse is 'raise' or 'false' and decides if the function throws an exception or if it returns false.
    '''
    if not (raiseorFalse == 'raise') or (raiseorFalse == 'false'):
        raise ValueError(f"Input {raiseorFalse} should be either 'raise' or 'false'.")
    
    if raiseorFalse == 'raise':
        if not type(l) is list:
            raise TypeError('Variable is {type(l)}, not list.')
        else: return
    
    elif raiseorFalse == 'false':
        if not type(l) is list:
            return False
        else: return True