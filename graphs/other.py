#%% ## Check against graphs
# def check_in_GHZ_orbit(adjecency_matrix, nr_qubits = None):
#     '''
#     Checks if the given adjecency matrix is in the LC orbit of a graph state.
#      - First count all entries in the adjecency matrix, set to ntot
#      - if ntot = n^2 - n, it is fully connected, and thus a GHZ
#      - if ntot is not 2*(n-1), it cannot be a star, and thus no GHZ
#     '''
#     # Get the total number of qubits if its not given
#     if nr_qubits is None:
#         nr_qubits = shape(adjecency_matrix)[0]
    
#     # Calculate ntot
#     ntot = npsum(npsum(adjecency_matrix))
    
#     # Compare against the criteria
#     # Fully connected?
#     if ntot == nr_qubits**2 - nr_qubits:
#         return True
#     # Possibly star?
#     elif ntot == 2*(nr_qubits - 1):
#         # Check if any row has n-1 terms and all rows are above 0 terms
#         if any(sum(adjecency_matrix) == nr_qubits - 1) & all(sum(adjecency_matrix) >= 0):
#             return True
#         else: return False
#     else: return False