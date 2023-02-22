import numpy as np
from sympy import symbols, acos, diff, sin, cos
from tqdm import tqdm


class Hessian:

    def __init__(self,index,potential,mode='bad'):
        if (mode == 'bad'):
            self.bond_index = index[0]
            self.angle_index = index[1]
            self.dihedral_index = index[2]
            self.nonbonded_index = index[3]
            self.nonbond_par = index[4]
            self.position = index[5]
            self.real_type = index[6]
            self.mass_type = index[7]
            self.bond_par = potential[0]
            self.angle_par = potential[1]
            self.dihedral_par = potential[2]

        else:
            print('Error XD')
    
    def second_deriavete_element_two_body(self,k,b,m1,m2,x1,x2,mode='bond'):
        if(mode=='bond'):
            # define sympy #
            xi,yi,zi,xj,yj,zj = symbols('xi yi zi xj yj zj', real=True)

            # define potential #
            rij = ((xj-xi)**2+(yj-yi)**2+(zj-zi)**2)**0.5
            potential_form = k*(rij-b)**2

            rij_xi_xi = (potential_form.diff(xi).diff(xi)).subs(xi,x1[0]).subs(yi,x1[1]).subs(zi,x1[2]).subs(xj,x2[0]).subs(yj,x2[1]).subs(zj,x2[2])
            rij_yi_yi = (potential_form.diff(yi).diff(yi)).subs(xi,x1[0]).subs(yi,x1[1]).subs(zi,x1[2]).subs(xj,x2[0]).subs(yj,x2[1]).subs(zj,x2[2])
            rij_zi_zi = (potential_form.diff(zi).diff(zi)).subs(xi,x1[0]).subs(yi,x1[1]).subs(zi,x1[2]).subs(xj,x2[0]).subs(yj,x2[1]).subs(zj,x2[2])
            rij_xi_yi = (potential_form.diff(xi).diff(yi)).subs(xi,x1[0]).subs(yi,x1[1]).subs(zi,x1[2]).subs(xj,x2[0]).subs(yj,x2[1]).subs(zj,x2[2])
            rij_xi_zi = (potential_form.diff(xi).diff(zi)).subs(xi,x1[0]).subs(yi,x1[1]).subs(zi,x1[2]).subs(xj,x2[0]).subs(yj,x2[1]).subs(zj,x2[2])
            rij_yi_zi = (potential_form.diff(yi).diff(zi)).subs(xi,x1[0]).subs(yi,x1[1]).subs(zi,x1[2]).subs(xj,x2[0]).subs(yj,x2[1]).subs(zj,x2[2])
        
            rij_xj_xj = (potential_form.diff(xj).diff(xj)).subs(xi,x1[0]).subs(yi,x1[1]).subs(zi,x1[2]).subs(xj,x2[0]).subs(yj,x2[1]).subs(zj,x2[2])
            rij_yj_yj = (potential_form.diff(yj).diff(yj)).subs(xi,x1[0]).subs(yi,x1[1]).subs(zi,x1[2]).subs(xj,x2[0]).subs(yj,x2[1]).subs(zj,x2[2])
            rij_zj_zj = (potential_form.diff(yj).diff(yj)).subs(xi,x1[0]).subs(yi,x1[1]).subs(zi,x1[2]).subs(xj,x2[0]).subs(yj,x2[1]).subs(zj,x2[2])
            rij_xj_yj = (potential_form.diff(xj).diff(yj)).subs(xi,x1[0]).subs(yi,x1[1]).subs(zi,x1[2]).subs(xj,x2[0]).subs(yj,x2[1]).subs(zj,x2[2])
            rij_xj_yj = (potential_form.diff(xj).diff(yj)).subs(xi,x1[0]).subs(yi,x1[1]).subs(zi,x1[2]).subs(xj,x2[0]).subs(yj,x2[1]).subs(zj,x2[2])
            rij_xj_zj = (potential_form.diff(xj).diff(zj)).subs(xi,x1[0]).subs(yi,x1[1]).subs(zi,x1[2]).subs(xj,x2[0]).subs(yj,x2[1]).subs(zj,x2[2])
            rij_yj_zj = (potential_form.diff(yj).diff(zj)).subs(xi,x1[0]).subs(yi,x1[1]).subs(zi,x1[2]).subs(xj,x2[0]).subs(yj,x2[1]).subs(zj,x2[2])

            rij_xi_xj = (potential_form.diff(xi).diff(xj)).subs(xi,x1[0]).subs(yi,x1[1]).subs(zi,x1[2]).subs(xj,x2[0]).subs(yj,x2[1]).subs(zj,x2[2])
            rij_xi_yj = (potential_form.diff(xi).diff(yj)).subs(xi,x1[0]).subs(yi,x1[1]).subs(zi,x1[2]).subs(xj,x2[0]).subs(yj,x2[1]).subs(zj,x2[2])
            rij_xi_zj = (potential_form.diff(xi).diff(zj)).subs(xi,x1[0]).subs(yi,x1[1]).subs(zi,x1[2]).subs(xj,x2[0]).subs(yj,x2[1]).subs(zj,x2[2])
            rij_yi_xj = (potential_form.diff(yi).diff(xj)).subs(xi,x1[0]).subs(yi,x1[1]).subs(zi,x1[2]).subs(xj,x2[0]).subs(yj,x2[1]).subs(zj,x2[2])
            rij_yi_yj = (potential_form.diff(yi).diff(yj)).subs(xi,x1[0]).subs(yi,x1[1]).subs(zi,x1[2]).subs(xj,x2[0]).subs(yj,x2[1]).subs(zj,x2[2])
            rij_yi_zj = (potential_form.diff(yi).diff(zj)).subs(xi,x1[0]).subs(yi,x1[1]).subs(zi,x1[2]).subs(xj,x2[0]).subs(yj,x2[1]).subs(zj,x2[2])
            rij_zi_xj = (potential_form.diff(zi).diff(xj)).subs(xi,x1[0]).subs(yi,x1[1]).subs(zi,x1[2]).subs(xj,x2[0]).subs(yj,x2[1]).subs(zj,x2[2])
            rij_zi_yj = (potential_form.diff(zi).diff(yj)).subs(xi,x1[0]).subs(yi,x1[1]).subs(zi,x1[2]).subs(xj,x2[0]).subs(yj,x2[1]).subs(zj,x2[2])
            rij_zi_zj = (potential_form.diff(zi).diff(zj)).subs(xi,x1[0]).subs(yi,x1[1]).subs(zi,x1[2]).subs(xj,x2[0]).subs(yj,x2[1]).subs(zj,x2[2])

            two_body_element = np.array([[rij_xi_xi,rij_xi_yi,rij_xi_zi,rij_xi_xj,rij_xi_yj,rij_xi_zj],
                                         [rij_xi_yi,rij_yi_yi,rij_yi_zi,rij_yi_xj,rij_yi_yj,rij_yi_zj],
                                         [rij_xi_zi,rij_yi_zi,rij_zi_zi,rij_zi_xj,rij_zi_yj,rij_zi_zj],
                                         [rij_xi_xj,rij_yi_xj,rij_zi_xj,rij_xj_xj,rij_xj_yj,rij_xj_zj],
                                         [rij_xi_yj,rij_yi_yj,rij_zi_yj,rij_xj_yj,rij_yj_yj,rij_yj_zj],
                                         [rij_xi_zj,rij_yi_zj,rij_zi_zj,rij_xj_zj,rij_yj_zj,rij_zj_zj]])
            mass_reduce = np.array([[(m1**0.5*m1**0.5),(m1**0.5*m1**0.5),(m1**0.5*m1**0.5),(m2**0.5*m1**0.5),(m2**0.5*m1**0.5),(m2**0.5*m1**0.5)],
                                    [(m1**0.5*m1**0.5),(m1**0.5*m1**0.5),(m1**0.5*m1**0.5),(m2**0.5*m1**0.5),(m2**0.5*m1**0.5),(m2**0.5*m1**0.5)],
                                    [(m1**0.5*m1**0.5),(m1**0.5*m1**0.5),(m1**0.5*m1**0.5),(m2**0.5*m1**0.5),(m2**0.5*m1**0.5),(m2**0.5*m1**0.5)],
                                    [(m2**0.5*m1**0.5),(m2**0.5*m1**0.5),(m2**0.5*m1**0.5),(m2**0.5*m2**0.5),(m2**0.5*m2**0.5),(m2**0.5*m2**0.5)],
                                    [(m2**0.5*m1**0.5),(m2**0.5*m1**0.5),(m2**0.5*m1**0.5),(m2**0.5*m2**0.5),(m2**0.5*m2**0.5),(m2**0.5*m2**0.5)],
                                    [(m2**0.5*m1**0.5),(m2**0.5*m1**0.5),(m2**0.5*m1**0.5),(m2**0.5*m2**0.5),(m2**0.5*m2**0.5),(m2**0.5*m2**0.5)]])
            
            return (two_body_element/mass_reduce).astype('float')
        #elif(mode=='pair'):
        else:
            print('error XD')

    def bond_K(self,kb,b0,m1,m2,x1,x2):
        return self.second_deriavete_element_two_body(kb,b0,m1,m2,x1,x2)

    #def angle_K(self,ka,t0,x1,x2,x3):

    #def dihedral_K(self,x1,x2,x3,x4):
    
    #def nonbonded_K(self,x1,x2)

    def build_matrix(self):
        hm = np.zeros((self.mass_type.shape[0]*3,self.mass_type.shape[0]*3))
    
        num_cout = 0
        bond_times = tqdm(total=self.bond_index.shape[0])
        for i,j in self.bond_index*3:

            k_n = self.bond_K(self.bond_par[num_cout,0],self.bond_par[num_cout,1],self.mass_type[int(i/3)],self.mass_type[int(j/3)],self.position[int(i/3)],self.position[int(j/3)])
            #print(k_n.shape)
            hm[i,i:i+3] += k_n[0,0:3]
            hm[i,j:j+3] += k_n[0,3:6]

            
            hm[i+1,i:i+3] += k_n[1,0:3]
            hm[i+1,j:j+3] += k_n[1,3:6]
            
            hm[j,i:i+3] += k_n[2,0:3]
            hm[j,j:j+3] += k_n[2,3:6]
            
            hm[j+1,i:i+3] += k_n[3,0:3]
            hm[j+1,j:j+3] += k_n[3,3:6]
            
            bond_times.update(1)
            num_cout += 1
        
        '''
        num_cout = 0
        for i,j,k in self.angle_index*3:
            ka_n = self.angle_K(self.angle_par[num_cout,0],np.deg2rad(self.angle_par[num_cout,1]),self.mass_type[int(i/3)],self.mass_type[int(j/3)],self.mass_type[int(k/3)],self.position[int(i/3)],self.position[int(j/3)],self.position[int(k/3)])
            hm[i,i:i+3] += ka_n[0,0:3]
            hm[i,j:j+3] += ka_n[0,3:6]
            hm[i,k:k+3] += ka_n[0,6:9]
            
            hm[i+1,i:i+3] += ka_n[1,0:3]
            hm[i+1,j:j+3] += ka_n[1,3:6]
            hm[i+1,k:k+3] += ka_n[1,6:9]
            
            hm[j,i:i+3] += ka_n[2,0:3]
            hm[j,j:j+3] += ka_n[2,3:6]
            hm[j,k:k+3] += ka_n[2,6:9]
            
            hm[j+1,i:i+3] += ka_n[3,0:3]
            hm[j+1,j:j+3] += ka_n[3,3:6]
            hm[j+1,k:k+3] += ka_n[3,6:9]
            
            hm[k,i:i+3] += ka_n[4,0:3]
            hm[k,j:j+3] += ka_n[4,3:6]
            hm[k,k:k+3] += ka_n[4,6:9]
            
            hm[k+1,i:i+3] += ka_n[5,0:3]
            hm[k+1,j:j+3] += ka_n[5,3:6]
            hm[k+1,k:k+3] += ka_n[5,6:9]
            
            num_cout += 1
        '''
        return hm

    #def energy_minimize(self):

    #def solve_Hessian(self,h):
    #   h_val, h_vec = np.linalg.eig(h)
    #   return h_val, h_vec
    #def PCA_frequence(self,n=20):



