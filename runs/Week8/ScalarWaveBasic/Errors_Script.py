import numpy as np
import h5py
import os

def is_file_empty(file_path):
  if not os.path.exists(file_path):
    return False
  return os.path.getsize(file_path) == 0

if __name__ == "__main__":
    f = h5py.File("ScalarWavePlaneWave1DEventsAndTriggersBasicReductions.h5")
    
    errors = np.array(f['Errors.dat'])
    
    refinement_i = f.attrs['InputSource.yaml'][0].find("InitialRefinement: [") + len("InitialRefinement: [")
    refinement = int(f.attrs['InputSource.yaml'][0][refinement_i:refinement_i+2].split(']')[0])
    grid_i = f.attrs['InputSource.yaml'][0].find("InitialGridPoints: [") + len("InitialGridPoints: [")
    grid = int(f.attrs['InputSource.yaml'][0][grid_i:grid_i+2].split(']')[0])
    
    path = "convergence.txt"
    empty = is_file_empty(path)
    with open(path, 'a') as file:
        if empty: file.write("Coord, Scalar, Spacial Deriv., Conj. Momentum\n")
        file.write(str(2**refinement * grid)+"\n")
        for i,slice in enumerate(errors[:,[3,4,5]]):
            file.write(f"{i}, {slice[0]}, {slice[1]}, {slice[2]}\n") 
        file.write("\n")
