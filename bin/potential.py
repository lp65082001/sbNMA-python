#!/usr/bin/env python
# coding: utf-8
from parmed.charmm import CharmmParameterSet
import numpy as np


class charmm_potential:
    def __init__(self,x,y,mode = 'bad'):
        self.params = CharmmParameterSet('./par_file/par_all22_prot.prm')
        if (mode == 'bad'):
            self.bond_index = x[0]
            self.angle_index = x[1]
            self.dihedral_index = x[2]
            self.improper_index = x[3]
            self.nonbond_index = x[4]
            self.nonbond_table = x[5]
            #self.position = x[5]
            self.type = x[6]  
            self.structure = y
        else:
            print('Error XD')

    def parameter_table(self):
        return [self.bond_parameter_table(),self.angle_parameter_table(),self.dihedral_parameter_table(),self.improper_parameter_table()]
            
    def bond_parameter_table(self):
        bond_par = []
        for i in range(0,self.bond_index.shape[0]):
            bond_par.append([self.params.bond_types[(self.type[self.bond_index[i,0]], self.type[self.bond_index[i,1]])].k,
            self.params.bond_types[(self.type[self.bond_index[i,0]], self.type[self.bond_index[i,1]])].req])
            
        return np.array(bond_par).reshape(-1,2)
    
    def angle_parameter_table(self):
        angle_par = []
        UB_par = []
        for i in range(0,self.angle_index.shape[0]):
            angle_par.append([self.params.angle_types[(self.type[self.angle_index[i,0]], self.type[self.angle_index[i,1]], self.type[self.angle_index[i,2]])].k,
            self.params.angle_types[(self.type[self.angle_index[i,0]], self.type[self.angle_index[i,1]], self.type[self.angle_index[i,2]])].theteq])

        for i in range(0,self.angle_index.shape[0]):
            UB_par.append([self.params.urey_bradley_types[(self.type[self.angle_index[i,0]], self.type[self.angle_index[i,1]], self.type[self.angle_index[i,2]])].k,
            self.params.urey_bradley_types[(self.type[self.angle_index[i,0]], self.type[self.angle_index[i,1]], self.type[self.angle_index[i,2]])].req])

        return np.array(angle_par).reshape(-1,2), np.array(UB_par).reshape(-1,2)
    
    def dihedral_parameter_table(self):
        dihedral_par = []
        for i in range(0,self.dihedral_index.shape[0]):
            dihedral_par.append([self.params.dihedral_types[(self.type[self.dihedral_index[i,0]], self.type[self.dihedral_index[i,1]], self.type[self.dihedral_index[i,2]], self.type[self.dihedral_index[i,3]])][0].phi_k,
            self.params.dihedral_types[(self.type[self.dihedral_index[i,0]], self.type[self.dihedral_index[i,1]], self.type[self.dihedral_index[i,2]], self.type[self.dihedral_index[i,3]])][0].per,
            self.params.dihedral_types[(self.type[self.dihedral_index[i,0]], self.type[self.dihedral_index[i,1]], self.type[self.dihedral_index[i,2]], self.type[self.dihedral_index[i,3]])][0].phase])
        return np.array(dihedral_par).reshape(-1,3)
    
    def improper_parameter_table(self):
        improper_par = []
        for i in range(0,self.improper_index.shape[0]):

            improper_par.append([self.params.improper_types[(self.type[self.improper_index[i,0]], self.type[self.improper_index[i,1]], self.type[self.improper_index[i,2]], self.type[self.improper_index[i,3]])].psi_k,
            self.params.improper_types[(self.type[self.improper_index[i,0]], self.type[self.improper_index[i,1]], self.type[self.improper_index[i,2]], self.type[self.improper_index[i,3]])].psi_eq])
        return np.array(improper_par).reshape(-1,2)


        


