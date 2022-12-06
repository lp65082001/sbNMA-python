import numpy as numpy
import re

class load_lammps_data:
    def __init__(self,filename):
        print("lammps_data_format reader initialization")
        self.path = filename

    def load(self):
        print("Load file")
        with open(self.path) as f:
            lines = f.readlines()
            #for line in lines:
                #print(list(filter(self.not_empty,re.split(r' |#',line))))


        print("done")


    def not_empty(self,string):
        return string and string.strip()




'''
class load_psf_pdb_file:
    def __init__(self,par_file):

    def load(self,filename):   
'''
