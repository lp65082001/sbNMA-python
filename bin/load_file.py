#!/usr/bin/env python
# coding: utf-8
from parmed.charmm import CharmmParameterSet
import parmed as pmd
import re
import numpy as np


class load_psf_pdb_file:
    def __init__(self,psf,pdb,mode="bonding_only"):
        self.params = CharmmParameterSet('./par_file/par_all22_prot.prm')
        self.structure = psf
        self.coordinate = pdb
        self.mode_ = mode

    def load(self):   
        if (self.mode_ == "bonding_only"):
            print('Load only bonding potential')
            self.bond_index, self.bond_atom = self.create_bond_list(self.structure)
            self.angle_index, self.angle_atom = self.create_angle_list(self.structure)
            self.dihedral_index, self.dihedral_atom = self.create_dihedral_list(self.structure)
            self.improper_index, self.improper_atom = self.create_improper_list(self.structure)
            self.nonbonded_index, self.nonbond_par = self.create_nonbonded_table(self.params)
            self.position = self.create_position_table(self.coordinate)
            print('Done')
        else:
            print('Error XD')

    def create_bond_list(self,x):
        # create bond list (index and atom name)
        index_list = np.zeros((len(x.bonds),2)).astype('int')
        atom_list = np.zeros((len(x.bonds),2)).astype('str')
        for i in range(0,len(x.bonds)):
            # tranform to str and split 
            t_list = list(filter(None,re.split('<|>|:|--|;',str(x.bonds[i]))))[1:5]
            t_list_1 = list(filter(None,re.split(r'\s',str(t_list[0]))))
            t_list_2 = list(filter(None,re.split(r'\s',str(t_list[2]))))

            index_list[i,0] = t_list_1[2][1:-1]
            atom_list[i,0] = t_list_1[1]
            index_list[i,1] = t_list_2[2][1:-1]
            atom_list[i,1] = t_list_2[1]
        return np.array(index_list), np.array(atom_list)

    def create_angle_list(self,x):
        # create angle list (index and atom name)
        index_list = np.zeros((len(x.angles),3)).astype('int')
        atom_list = np.zeros((len(x.angles),3)).astype('str')
        for i in range(0,len(x.angles)):
            # tranform to str and split 
            t_list = list(filter(None,re.split(r'<|>|:|--|;',str(x.angles[i]))))[1:7]
            t_list_1 = list(filter(None,re.split(r'\s',str(t_list[0]))))
            t_list_2 = list(filter(None,re.split(r'\s',str(t_list[2]))))
            t_list_3 = list(filter(None,re.split(r'\s',str(t_list[4]))))

            index_list[i,0] = t_list_1[2][1:-1]
            atom_list[i,0] = t_list_1[1]
            index_list[i,1] = t_list_2[2][1:-1]
            atom_list[i,1] = t_list_2[1]
            index_list[i,2] = t_list_3[2][1:-1]
            atom_list[i,2] = t_list_3[1]
        return np.array(index_list), np.array(atom_list)

    def create_dihedral_list(self,x):
        # create angle list (index and atom name)
        index_list = np.zeros((len(x.dihedrals),4)).astype('int')
        atom_list = np.zeros((len(x.dihedrals),4)).astype('str')
        for i in range(0,len(x.dihedrals)):
            # tranform to str and split 
            t_list = list(filter(None,re.split(r'<|>|:|--|;',str(x.dihedrals[i]))))[2:10]
            t_list_1 = list(filter(None,re.split(r'\s',t_list[0])))
            t_list_2 = list(filter(None,re.split(r'\s',t_list[2])))
            t_list_3 = list(filter(None,re.split(r'\s',t_list[4])))
            t_list_4 = list(filter(None,re.split(r'\s',t_list[6])))

            index_list[i,0] = t_list_1[2][1:-1]
            atom_list[i,0] = t_list_1[1]
            index_list[i,1] = t_list_2[2][1:-1]
            atom_list[i,1] = t_list_2[1]
            index_list[i,2] = t_list_3[2][1:-1]
            atom_list[i,2] = t_list_3[1]
            index_list[i,3] = t_list_4[2][1:-1]
            atom_list[i,3] = t_list_4[1]
        return np.array(index_list), np.array(atom_list)

    def create_improper_list(self,x):
        # create angle list (index and atom name)
        index_list = np.zeros((len(x.impropers),4)).astype('int')
        atom_list = np.zeros((len(x.impropers),4)).astype('str')
        for i in range(0,len(x.impropers)):
            # tranform to str and split 
            t_list = list(filter(None,re.split(r'<|>|:|--|;|,',str(x.impropers[i]))))[2:10]
            t_list_1 = list(filter(None,re.split(r'\s',t_list[0])))
            t_list_2 = list(filter(None,re.split(r'\s',t_list[3])))
            t_list_3 = list(filter(None,re.split(r'\s',t_list[5])))
            t_list_4 = list(filter(None,re.split(r'\s',t_list[7])))

            # colume is center atom
            index_list[i,0] = t_list_1[2][1:-1]
            atom_list[i,0] = t_list_1[1]
            index_list[i,1] = t_list_2[2][1:-1]
            atom_list[i,1] = t_list_2[1]
            index_list[i,2] = t_list_3[2][1:-1]
            atom_list[i,2] = t_list_3[1]
            index_list[i,3] = t_list_4[2][1:-1]
            atom_list[i,3] = t_list_4[1]
        return np.array(index_list), np.array(atom_list)

    def create_nonbonded_table(self,y):
        all_nonbond_table =list(y.atom_types)  
        table = np.zeros((len(all_nonbond_table),2))
        for i in range(0,len(all_nonbond_table)):
            table[i,0] = y.atom_types_str[all_nonbond_table[i]].epsilon
            table[i,1] = y.atom_types_str[all_nonbond_table[i]].sigma
        return np.array(all_nonbond_table), np.array(table)

    def create_position_table(self,z):
        return z.coordinates

    def get_table(self):
        if (self.mode_ == "bonding_only"):
            return [self.bond_index, self.bond_atom,
            self.angle_index, self.angle_atom,
            self.dihedral_index, self.dihedral_atom,
            self.improper_index, self.improper_atom,
            self.nonbonded_index, self.nonbond_par,
            self.position]
 
        else:
            print("Error XD")