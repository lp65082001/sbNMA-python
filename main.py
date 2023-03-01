import numpy as np
import parmed as pmd
from bin import load_file, potential, hessian_build, energy_minimze, correlation
from parmed.charmm import CharmmParameterSet 
import time


if __name__ == '__main__':
    topology = './example/ubq.pdb'
    bonding = './example/ubq.psf'

    # Input loading 
    psf = pmd.charmm.psf.CharmmPsfFile(bonding)
    pdb = pmd.load_file(topology)
    model_info = load_file.load_psf_pdb_file(psf,pdb)
    model_info.load()
    
    # get model info list #
    model_index = model_info.get_table()

    # get CA index #
    ca_index = model_info.get_reduce_index()

    # get potential parameter #
    potent = potential.charmm_potential(model_info.get_table(),psf)
    model_potential = potent.parameter_table()
    
    # Energy minimization #
    em_pos = energy_minimze.min(topology,bonding)
    
    # build hessian matrix #
    
    hessian_ = hessian_build.Hessian(model_index,model_potential,em_pos)
    hessian_martix = hessian_.build_matrix()

    # solve eigenvalue and eigenvector #
    eignvalue,eignvector = hessian_.solve_Hessian(hessian_martix)

    # calculate dynamics coupling #
    dc = correlation.dynamics_coupling(eignvalue,eignvector,ca_table=ca_index)
    dc.run(reduce=True,ccm = True)
    dc.plot_matrix()


    

