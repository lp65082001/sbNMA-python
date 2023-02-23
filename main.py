import numpy as np
import parmed as pmd
from bin import load_file, potential, hessian_build
from parmed.charmm import CharmmParameterSet 
import time


if __name__ == '__main__':

    # Input loading 
    psf = pmd.charmm.psf.CharmmPsfFile('./example/gap_p.psf')
    pdb = pmd.load_file('./example/gap_p.pdb')
    model_info = load_file.load_psf_pdb_file(psf,pdb)
    model_info.load()
    
    # get model info list #
    model_index = model_info.get_table()
    
    # get potential parameter #
    potent = potential.charmm_potential(model_info.get_table(),psf)
    model_potential = potent.parameter_table()
    
    # build hessian matrix #
    
    hessian_ = hessian_build.Hessian(model_index,model_potential)
    hessian_martix = hessian_.build_matrix()
    
    start = time.time()
    eignvalue,eignvector = hessian_.solve_Hessian(hessian_martix)
    
    endtime = time.time()
    print (endtime - start)
    print(hessian_martix)

    # solve eigenvalue and eigenvector #

