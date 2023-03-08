#!/usr/bin/env python
# coding: utf-8
from parmed.charmm import CharmmParameterSet
import parmed as pmd
import re
import numpy as np
import MDAnalysis as mda


class load_psf_pdb_file:
    def __init__(self,psf,pdb,mode="bad"):
        self.params = CharmmParameterSet('./par_file/par_all22_prot.prm')
        self.structure = psf
        self.coordinate = pdb
        self.mode_ = mode
        u = mda.Universe(psf, pdb)
        self.real_type = u.select_atoms("all").types

    def load(self):   
        if (self.mode_ == "bad"):
            print('Structure and Potential initialization!')
            self.bond_index, self.bond_atom = self.create_bond_list(self.structure)
            self.angle_index, self.angle_atom = self.create_angle_list(self.structure)
            self.dihedral_index, self.dihedral_atom = self.create_dihedral_list(self.structure)
            self.improper_index, self.improper_atom = self.create_improper_list(self.structure)
            self.nonbonded_index, self.nonbond_par = self.create_nonbonded_table(self.params)
            self.position = self.create_position_table(self.coordinate)
            self.mass_table = self.create_mass_table()
            
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

    def create_mass_table(self):
        # custom #
        mass_table = []
        for i in self.real_type:
            if(i[0]=='H'):
                mass_table.append(1.008)
            elif(i[0]=='C'):
                mass_table.append(12.011)
            elif(i[0]=='N'):
                mass_table.append(14.007)
            elif(i[0]=='O'):
                mass_table.append(15.999)
            elif(i[0]=='S'):
                mass_table.append(32.06)
        return np.array(mass_table)
    
    def get_reduce_index(self,name="CA"):
        red_index = []
        for i in self.structure.atoms:
            red_index.append(i.name)
            #print(i.name)
        red_index = np.array(red_index)
        return np.where(red_index==name)[0]
    
    def get_position(self):
        return self.position

    def get_table(self):
        if (self.mode_ == "bad"):
            return [self.bond_index,
            self.angle_index, 
            self.dihedral_index,
            self.improper_index,
            self.nonbonded_index, self.nonbond_par,
            #self.position,
            self.real_type,
            self.mass_table]
 
        else:
            print("Error XD")