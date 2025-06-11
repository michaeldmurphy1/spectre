import numpy as np
import h5py
import os

def is_file_empty(file_path):
  if not os.path.exists(file_path):
    return True
  return os.path.getsize(file_path) == 0

def modify_yaml_after_marker(filename, marker_string, new_value):
    with open(filename, 'r') as f:
        lines = f.readlines()
    # Find the marker line index
    marker = 0
    val = ""
    for i, line in enumerate(lines):
        if marker_string in line:
            marker = i
            first, second = line.split(':')
            print(first+": "+str(new_value)+'\n')
            val = first+": "+str(new_value)+'\n'
            break
    lines[marker] = val
    with open(filename, 'w') as f:
        f.writelines(lines)

if __name__ == "__main__":
    f = h5py.File("ScalarWavePlaneWave1DEventsAndTriggersModReductions.h5")
    
    errors = np.array(f['Errors.dat'])
    
    refinement_i = f.attrs['InputSource.yaml'][0].find("InitialRefinement: [") + len("InitialRefinement: [")
    refinement = int(f.attrs['InputSource.yaml'][0][refinement_i:refinement_i+2].split(']')[0])
    grid_i = f.attrs['InputSource.yaml'][0].find("InitialGridPoints: [") + len("InitialGridPoints: [")
    grid = int(f.attrs['InputSource.yaml'][0][grid_i:grid_i+2].split(']')[0])

    order_i = f.attrs['InputSource.yaml'][0].find("Order: ") + len("Order: ")
    order = int(f.attrs['InputSource.yaml'][0][order_i:order_i+2].split(']')[0])
    
    timestep_i = f.attrs['InputSource.yaml'][0].find("InitialTimeStep: ") + len("InitialTimeStep: ")
    timestep = float(f.attrs['InputSource.yaml'][0][timestep_i:timestep_i+5].split('\n')[0].strip())

    
    path = "convergence.txt"
    empty = is_file_empty(path)
    with open(path, 'a') as file:
        if empty: file.write("Coord, Scalar, Spacial Deriv., Conj. Momentum\n")
        # if empty: file.write("grid, order, Scalar error\n")
        # file.write(str(grid)+", "+str(order)+", "+str(errors[-1,3])+"\n")
        file.write(str(2**refinement * grid)+", init_Tstep: "+str(timestep)+"\n")
        for i,slice in enumerate(errors[:,[0,3,4,5]]):
            # if i%2 != 0: continue
            file.write(f"{slice[0]}, {slice[1]}, {slice[2]}, {slice[3]}\n") 

        file.write("\n")

    modify_yaml_after_marker('PlaneWave1D.yaml', "InitialTimeStep", 0.1*timestep)
    # modify_yaml_after_marker('PlaneWave1D.yaml', "InitialTimeStep", 0.1*timestep)

    
    # with open(path, 'a') as file:
    #     if empty: file.write("Coord, Scalar, Spacial Deriv., Conj. Momentum\n")
    #     file.write(str(2**refinement * grid)+", order: "+str(order)+"\n")
    #     for i,slice in enumerate(errors[:,[0,3,4,5]]):
    #         # if i%2 != 0: continue
    #         file.write(f"{slice[0]}, {slice[1]}, {slice[2]}, {slice[3]}\n") 

    #     file.write("\n")


