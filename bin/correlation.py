import numpy as np

class dynamics_coupling:
    def __init__(self,value,vector,mode=20):
        self.eignvalue = value
        self.eignvector = vector
        self.mode_ = mode

    def non_degenerate_modes(self):
        self.positive_eignvalue = self.eignvalue[np.where(self.eignvalue>0)[0]]
        self.new_index = np.argsort(self.positive_eignvalue)
        self.positive_eignvector = self.eignvector[:,self.new_index ]
    
    def cross_correlation(self):
        ccm = np.zeros([int(self.eigenvalue.shape[0])/3,int(self.eigenvalue.shape[0])/3])

        return

    def run(self,ccm = None):
        self.non_degenerate_modes()
        if (ccm==True):
            self.cross_correlation()