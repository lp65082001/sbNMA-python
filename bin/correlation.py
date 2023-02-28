import numpy as np
kb = 1.987e-3
T = 310


class dynamics_coupling:
    def __init__(self,value,vector,mode = 20,ca_table = None):
        self.eignvalue = value
        self.eignvector = vector
        self.mode_ = mode
        self.ca = ca_table
    
    def get_result(self):

    def non_degenerate_modes(self):
        self.positive_eignvalue = self.eignvalue[np.where(self.eignvalue>0)[0]]
        self.new_index = np.argsort(self.positive_eignvalue)
        self.positive_eignvector = self.eignvector[:,self.new_index ]
    
    def cross_correlation(self):
        ccm_ref = np.zeros([int(self.eigenvalue.shape[0])/3,int(self.eigenvalue.shape[0])/3])
        ccm = np.zeros([int(self.eigenvalue.shape[0])/3,int(self.eigenvalue.shape[0])/3])

        print("Build cross-correlation martix")
        ccmt = tqdm(total=self.mode_,ncols=100)
        for feq in range(self.mode_):
            for pair1 in range(int(self.eigenvalue.shape[0])/3):
                for pair2 in range(int(self.eigenvalue.shape[0])/3):
                    ui = self.positive_eignvector[[3*pair1,3*pair1+1,3*pair1+2],feq]
                    uj = self.positive_eignvector[[3*pair2,3*pair2+1,3*pair2+2],feq]
                    ccm_ref[pair1,pair2] += np.dot(ui,uj)/self.positive_eignvalue
            ccmt.update()
        ccmt.close()
        print("Done")
        print("Normalized")
        for pair1 in range(int(self.eigenvalue.shape[0])/3):
            for pair2 in range(int(self.eigenvalue.shape[0])/3):
                ccm[pair1,pair2] = ccm_ref[pair1,pair2]/(ccm_ref[pair1,pair1]*ccm_ref[pair2,pair2])**0.5
        self.ccm = ccm
        print("Done")
        

    def run(self, reduce=True, ccm = None):
        self.output_list = [ccm]
        self.non_degenerate_modes()
        if (ccm==True):
            self.cross_correlation()


        