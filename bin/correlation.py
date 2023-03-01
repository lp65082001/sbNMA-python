import numpy as np
from tqdm import tqdm
from numba import jit
import matplotlib.pyplot as plt
# figure setting #
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['figure.dpi'] = 300
# Parameter #
kb = 1.987e-3
T = 310

class dynamics_coupling:
    def __init__(self,value,vector,mode = 20,ca_table = None):
        self.eignvalue = value
        self.eignvector = vector
        self.mode_ = mode
        self.ca = ca_table
    
    #def get_result(self):
    
    def plot_matrix(self,title="ccm"):
        if(title=="ccm"):
            fig, ax = plt.subplots()
            im = ax.imshow(self.ccm_,vmin=-1, vmax=1, cmap='cool')
            fig.colorbar(im, ax=ax, label='Cross correlation')
            ax.invert_yaxis()
            plt.savefig('./result/test.png',bbox_inches='tight')
            #plt.show()
    
    def vector_table(self,n):    
        return self.positive_eignvector[:,n].reshape(-1,3)

    def non_degenerate_modes(self):
        self.positive_eignvalue = self.eignvalue[np.where(self.eignvalue>0)[0]]
        self.new_index = np.argsort(self.positive_eignvalue)
        self.positive_eignvector = self.eignvector[:,self.new_index]
    
    @jit
    def cross_correlation(self,reduce = None):
        ccm_ref = np.zeros([int(self.eignvalue.shape[0]/3),int(self.eignvalue.shape[0]/3)])
        ccm = np.zeros([int(self.eignvalue.shape[0]/3),int(self.eignvalue.shape[0]/3)])

        print("Build cross-correlation martix")
        ccmt = tqdm(total=self.mode_,ncols=100)
        for feq in range(self.mode_):
            ccm_ref += np.tensordot(self.vector_table(feq),self.vector_table(feq).T,axes=([1],[0]))/self.positive_eignvalue[feq]
            ccmt.update()
        ccmt.close()
        ccm_ref = kb*T*ccm_ref
        print("Done")
        print("Normalized")
        ccmt2 = tqdm(total=int(self.eignvalue.shape[0]/3),ncols=100)
        for i in range(int(self.eignvalue.shape[0]/3)):
            ccm[i,:] = np.diagonal(ccm_ref)*ccm_ref[i,i]
            ccmt2.update()
        ccm_ = ccm_ref/(ccm)**0.5
        if (reduce ==True):
            self.ccm_ = ccm_[np.ix_(self.ca,self.ca)]
        else:
            self.ccm_ = ccm_
        ccmt2.close()
        print("Done")

    def get_ccm(self):
        return self.ccm_
        
    def run(self,reduce=True, ccm = None):
        self.output_list = [ccm]
        self.non_degenerate_modes()
        if (ccm==True):
            if(reduce==True):
                self.cross_correlation(reduce)
            else:
                self.cross_correlation()
        




        