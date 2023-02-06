#!/usr/bin/env python
# coding: utf-8
from parmed.charmm import CharmmParameterSet
import numpy as np


class charmm_potential:
    def __init__(self,x,y,mode = 'bonding_only'):
        self.params = CharmmParameterSet('./par_file/par_all22_prot.prm')
        if (mode == 'bonding_only'):
            self.bond_index = x[0]
            self.angle_index = x[1]
            self.dihedral_index = x[2]
            self.improper_index = x[3]
            self.nonbond_index = x[4]
            self.nonbond_table = x[5]
            self.position = x[6]
            self.type = x[7]
            self.structure = y
        else:
            print('Error XD')

    #def parameter_table(self):
        
        
    def bond_parameter_table(self):
        bond_par = []
        for i in range(0,self.bond_index.shape[0]):
            bond_par.append(self.params.bond_types[(self.type[self.bond_index[i,0]], self.type[self.bond_index[i,1]])])
            
        return np.array(bond_par)




