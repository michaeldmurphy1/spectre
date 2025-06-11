import numpy as np
import h5py
import os

def is_file_empty(file_path):
  if not os.path.exists(file_path):
    return False
  return os.path.getsize(file_path) == 0

if __name__ == "__main__":
    f = h5py.File("GrmhdSmoothFlowSubcellP5Lev4.h5")
    
    errors = np.array(f['Errors.dat'])
    print(errors)
    
    refinement_i = f.attrs['InputSource.yaml'][0].find("InitialRefinement: [") + len("InitialRefinement: [")
    refinement = int(f.attrs['InputSource.yaml'][0][refinement_i:refinement_i+2].replace(',',']').split(']')[0])
    grid_i = f.attrs['InputSource.yaml'][0].find("InitialGridPoints: [") + len("InitialGridPoints: [")
    grid = int(f.attrs['InputSource.yaml'][0][grid_i:grid_i+2].replace(',',']').split(']')[0])
    
    path = "convergence.txt"
    empty = is_file_empty(path)
    with open(path, 'a') as file:
        if empty: file.write("P_n, N_x, L_2(Error(rho))\n")
        file.write(str(2**refinement)+","+str(grid-1)+",")
        file.write(str(errors[1,3]))
        file.write("\n")