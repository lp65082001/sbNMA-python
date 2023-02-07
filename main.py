import numpy as numpy
import parmed as pmd
from bin import load_file, potential, hessian_build
from parmed.charmm import CharmmParameterSet    


if __name__ == '__main__':

    # Input loading 
    psf = pmd.charmm.psf.CharmmPsfFile('./example/gap_p.psf')
    pdb = pmd.load_file('./example/gap_p.pdb')
    model_info = load_file.load_psf_pdb_file(psf,pdb)
    model_info.load()
    
    # get model info list #
    model_info.get_table()
    
    # get potential parameter #
    potent = potential.charmm_potential(model_info.get_table(),psf)
    potent.parameter_table()

    # build hessian matrix #


    # solve eigenvalue and eigenvector #