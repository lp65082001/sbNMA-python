import numpy as np
from sympy import symbols, acos, diff, sin, cos, lambdify
from tqdm import tqdm
#import cupy as cp
import warnings 
import time
import jax

warnings.filterwarnings('ignore')

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



    def element_initialization(self):
        print("Element initialization!")

        xi,yi,zi,xj,yj,zj,xk,yk,zk,xl,yl,zl,k,b = symbols('xi yi zi xj yj zj xk yk zk xl yl zl k b', real=True)

        # bond potential #
        bond_rij = ((xj-xi)**2+(yj-yi)**2+(zj-zi)**2)**0.5
        bond_potential_form = k*(bond_rij-b)**2
        
        rij_xi_xi = lambdify((k,b,xi,yi,zi,xj,yj,zj),(bond_potential_form.diff(xi).diff(xi)),'numpy')
        rij_yi_yi = lambdify((k,b,xi,yi,zi,xj,yj,zj),(bond_potential_form.diff(yi).diff(yi)),'numpy')
        rij_zi_zi = lambdify((k,b,xi,yi,zi,xj,yj,zj),(bond_potential_form.diff(zi).diff(zi)),'numpy')
        rij_xi_yi = lambdify((k,b,xi,yi,zi,xj,yj,zj),(bond_potential_form.diff(xi).diff(yi)),'numpy')
        rij_xi_zi = lambdify((k,b,xi,yi,zi,xj,yj,zj),(bond_potential_form.diff(xi).diff(zi)),'numpy')
        rij_yi_zi = lambdify((k,b,xi,yi,zi,xj,yj,zj),(bond_potential_form.diff(yi).diff(zi)),'numpy')
        
        rij_xj_xj = lambdify((k,b,xi,yi,zi,xj,yj,zj),(bond_potential_form.diff(xj).diff(xj)),'numpy')
        rij_yj_yj = lambdify((k,b,xi,yi,zi,xj,yj,zj),(bond_potential_form.diff(yj).diff(yj)),'numpy')
        rij_zj_zj = lambdify((k,b,xi,yi,zi,xj,yj,zj),(bond_potential_form.diff(yj).diff(yj)),'numpy')
        rij_xj_yj = lambdify((k,b,xi,yi,zi,xj,yj,zj),(bond_potential_form.diff(xj).diff(yj)),'numpy')
        rij_xj_yj = lambdify((k,b,xi,yi,zi,xj,yj,zj),(bond_potential_form.diff(xj).diff(yj)),'numpy')
        rij_xj_zj = lambdify((k,b,xi,yi,zi,xj,yj,zj),(bond_potential_form.diff(xj).diff(zj)),'numpy')
        rij_yj_zj = lambdify((k,b,xi,yi,zi,xj,yj,zj),(bond_potential_form.diff(yj).diff(zj)),'numpy')

        rij_xi_xj = lambdify((k,b,xi,yi,zi,xj,yj,zj),(bond_potential_form.diff(xi).diff(xj)),'numpy')
        rij_xi_yj = lambdify((k,b,xi,yi,zi,xj,yj,zj),(bond_potential_form.diff(xi).diff(yj)),'numpy')
        rij_xi_zj = lambdify((k,b,xi,yi,zi,xj,yj,zj),(bond_potential_form.diff(xi).diff(zj)),'numpy')
        rij_yi_xj = lambdify((k,b,xi,yi,zi,xj,yj,zj),(bond_potential_form.diff(yi).diff(xj)),'numpy')
        rij_yi_yj = lambdify((k,b,xi,yi,zi,xj,yj,zj),(bond_potential_form.diff(yi).diff(yj)),'numpy')
        rij_yi_zj = lambdify((k,b,xi,yi,zi,xj,yj,zj),(bond_potential_form.diff(yi).diff(zj)),'numpy')
        rij_zi_xj = lambdify((k,b,xi,yi,zi,xj,yj,zj),(bond_potential_form.diff(zi).diff(xj)),'numpy')
        rij_zi_yj = lambdify((k,b,xi,yi,zi,xj,yj,zj),(bond_potential_form.diff(zi).diff(yj)),'numpy')
        rij_zi_zj = lambdify((k,b,xi,yi,zi,xj,yj,zj),(bond_potential_form.diff(zi).diff(zj)),'numpy')

        self.bond_element = np.array([rij_xi_xi,rij_xi_yi,rij_xi_zi,rij_xi_xj,rij_xi_yj,rij_xi_zj,
                                 rij_yi_yi,rij_yi_zi,rij_yi_xj,rij_yi_yj,rij_yi_zj,
                                 rij_zi_zi,rij_zi_xj,rij_zi_yj,rij_zi_zj,
                                 rij_xj_xj,rij_xj_yj,rij_xj_zj,
                                 rij_yj_yj,rij_yj_zj,
                                 rij_zj_zj])
        
        # angle potential #
        cos_theta = (((xj-xi)**2+(yj-yi)**2+(zj-zi)**2)+((xk-xj)**2+(yk-yj)**2+(zk-zj)**2)-((xk-xi)**2+(yk-yi)**2+(zk-zi)))/(2*((xj-xi)**2+(yj-yi)**2+(zj-zi)**2)**0.5+((xk-xj)**2+(yk-yj)**2+(zk-zj)**2))
        angle_potential_form = k*(cos_theta-b)**2

        cos_theta_xi_xi = lambdify((k,b,xi,yi,zi,xj,yj,zj,xk,yk,zk),(angle_potential_form.diff(xi).diff(xi)),'numpy')
        cos_theta_xi_yi = lambdify((k,b,xi,yi,zi,xj,yj,zj,xk,yk,zk),(angle_potential_form.diff(xi).diff(yi)),'numpy')
        cos_theta_xi_zi = lambdify((k,b,xi,yi,zi,xj,yj,zj,xk,yk,zk),(angle_potential_form.diff(xi).diff(zi)),'numpy')
        cos_theta_xi_xj = lambdify((k,b,xi,yi,zi,xj,yj,zj,xk,yk,zk),(angle_potential_form.diff(xi).diff(xj)),'numpy')
        cos_theta_xi_yj = lambdify((k,b,xi,yi,zi,xj,yj,zj,xk,yk,zk),(angle_potential_form.diff(xi).diff(yj)),'numpy')
        cos_theta_xi_zj = lambdify((k,b,xi,yi,zi,xj,yj,zj,xk,yk,zk),(angle_potential_form.diff(xi).diff(zj)),'numpy')
        cos_theta_xi_xk = lambdify((k,b,xi,yi,zi,xj,yj,zj,xk,yk,zk),(angle_potential_form.diff(xi).diff(xk)),'numpy')
        cos_theta_xi_yk = lambdify((k,b,xi,yi,zi,xj,yj,zj,xk,yk,zk),(angle_potential_form.diff(xi).diff(yk)),'numpy')
        cos_theta_xi_zk = lambdify((k,b,xi,yi,zi,xj,yj,zj,xk,yk,zk),(angle_potential_form.diff(xi).diff(zk)),'numpy')
        
        cos_theta_yi_yi = lambdify((k,b,xi,yi,zi,xj,yj,zj,xk,yk,zk),(angle_potential_form.diff(yi).diff(yi)),'numpy')
        cos_theta_yi_zi = lambdify((k,b,xi,yi,zi,xj,yj,zj,xk,yk,zk),(angle_potential_form.diff(yi).diff(zi)),'numpy')
        cos_theta_yi_xj = lambdify((k,b,xi,yi,zi,xj,yj,zj,xk,yk,zk),(angle_potential_form.diff(yi).diff(xj)),'numpy')
        cos_theta_yi_yj = lambdify((k,b,xi,yi,zi,xj,yj,zj,xk,yk,zk),(angle_potential_form.diff(yi).diff(yj)),'numpy')
        cos_theta_yi_zj = lambdify((k,b,xi,yi,zi,xj,yj,zj,xk,yk,zk),(angle_potential_form.diff(yi).diff(zj)),'numpy')
        cos_theta_yi_xk = lambdify((k,b,xi,yi,zi,xj,yj,zj,xk,yk,zk),(angle_potential_form.diff(yi).diff(xk)),'numpy')
        cos_theta_yi_yk = lambdify((k,b,xi,yi,zi,xj,yj,zj,xk,yk,zk),(angle_potential_form.diff(yi).diff(yk)),'numpy')
        cos_theta_yi_zk = lambdify((k,b,xi,yi,zi,xj,yj,zj,xk,yk,zk),(angle_potential_form.diff(yi).diff(zk)),'numpy')
        
        cos_theta_zi_zi = lambdify((k,b,xi,yi,zi,xj,yj,zj,xk,yk,zk),(angle_potential_form.diff(zi).diff(zi)),'numpy')
        cos_theta_zi_xj = lambdify((k,b,xi,yi,zi,xj,yj,zj,xk,yk,zk),(angle_potential_form.diff(zi).diff(xj)),'numpy')
        cos_theta_zi_yj = lambdify((k,b,xi,yi,zi,xj,yj,zj,xk,yk,zk),(angle_potential_form.diff(zi).diff(yj)),'numpy')
        cos_theta_zi_zj = lambdify((k,b,xi,yi,zi,xj,yj,zj,xk,yk,zk),(angle_potential_form.diff(zi).diff(zj)),'numpy')
        cos_theta_zi_xk = lambdify((k,b,xi,yi,zi,xj,yj,zj,xk,yk,zk),(angle_potential_form.diff(zi).diff(xk)),'numpy')
        cos_theta_zi_yk = lambdify((k,b,xi,yi,zi,xj,yj,zj,xk,yk,zk),(angle_potential_form.diff(zi).diff(yk)),'numpy')
        cos_theta_zi_zk = lambdify((k,b,xi,yi,zi,xj,yj,zj,xk,yk,zk),(angle_potential_form.diff(zi).diff(zk)),'numpy')

        cos_theta_xj_xj = lambdify((k,b,xi,yi,zi,xj,yj,zj,xk,yk,zk),(angle_potential_form.diff(xj).diff(xj)),'numpy')
        cos_theta_xj_yj = lambdify((k,b,xi,yi,zi,xj,yj,zj,xk,yk,zk),(angle_potential_form.diff(xj).diff(yj)),'numpy')
        cos_theta_xj_zj = lambdify((k,b,xi,yi,zi,xj,yj,zj,xk,yk,zk),(angle_potential_form.diff(xj).diff(zj)),'numpy')
        cos_theta_xj_xk = lambdify((k,b,xi,yi,zi,xj,yj,zj,xk,yk,zk),(angle_potential_form.diff(xj).diff(xk)),'numpy')
        cos_theta_xj_yk = lambdify((k,b,xi,yi,zi,xj,yj,zj,xk,yk,zk),(angle_potential_form.diff(xj).diff(yk)),'numpy')
        cos_theta_xj_zk = lambdify((k,b,xi,yi,zi,xj,yj,zj,xk,yk,zk),(angle_potential_form.diff(xj).diff(zk)),'numpy')

        cos_theta_yj_yj = lambdify((k,b,xi,yi,zi,xj,yj,zj,xk,yk,zk),(angle_potential_form.diff(yj).diff(yj)),'numpy')
        cos_theta_yj_zj = lambdify((k,b,xi,yi,zi,xj,yj,zj,xk,yk,zk),(angle_potential_form.diff(yj).diff(zj)),'numpy')
        cos_theta_yj_xk = lambdify((k,b,xi,yi,zi,xj,yj,zj,xk,yk,zk),(angle_potential_form.diff(yj).diff(xk)),'numpy')
        cos_theta_yj_yk = lambdify((k,b,xi,yi,zi,xj,yj,zj,xk,yk,zk),(angle_potential_form.diff(yj).diff(yk)),'numpy')
        cos_theta_yj_zk = lambdify((k,b,xi,yi,zi,xj,yj,zj,xk,yk,zk),(angle_potential_form.diff(yj).diff(zk)),'numpy')
        
        cos_theta_zj_zj = lambdify((k,b,xi,yi,zi,xj,yj,zj,xk,yk,zk),(angle_potential_form.diff(zj).diff(zj)),'numpy')
        cos_theta_zj_xk = lambdify((k,b,xi,yi,zi,xj,yj,zj,xk,yk,zk),(angle_potential_form.diff(zj).diff(xk)),'numpy')
        cos_theta_zj_yk = lambdify((k,b,xi,yi,zi,xj,yj,zj,xk,yk,zk),(angle_potential_form.diff(zj).diff(yk)),'numpy')
        cos_theta_zj_zk = lambdify((k,b,xi,yi,zi,xj,yj,zj,xk,yk,zk),(angle_potential_form.diff(zj).diff(zk)),'numpy')

        cos_theta_xk_xk = lambdify((k,b,xi,yi,zi,xj,yj,zj,xk,yk,zk),(angle_potential_form.diff(xk).diff(xk)),'numpy')
        cos_theta_xk_yk = lambdify((k,b,xi,yi,zi,xj,yj,zj,xk,yk,zk),(angle_potential_form.diff(xk).diff(yk)),'numpy')
        cos_theta_xk_zk = lambdify((k,b,xi,yi,zi,xj,yj,zj,xk,yk,zk),(angle_potential_form.diff(xk).diff(zk)),'numpy')

        cos_theta_yk_yk = lambdify((k,b,xi,yi,zi,xj,yj,zj,xk,yk,zk),(angle_potential_form.diff(yk).diff(yk)),'numpy')
        cos_theta_yk_zk = lambdify((k,b,xi,yi,zi,xj,yj,zj,xk,yk,zk),(angle_potential_form.diff(yk).diff(zk)),'numpy')

        cos_theta_zk_zk = lambdify((k,b,xi,yi,zi,xj,yj,zj,xk,yk,zk),(angle_potential_form.diff(zk).diff(zk)),'numpy')

        self.angle_element = np.array([cos_theta_xi_xi,cos_theta_xi_yi,cos_theta_xi_zi,cos_theta_xi_xj,cos_theta_xi_yj,cos_theta_xi_zj,cos_theta_xi_xk,cos_theta_xi_yk,cos_theta_xi_zk,
                                      cos_theta_yi_yi,cos_theta_yi_zi,cos_theta_yi_xj,cos_theta_yi_yj,cos_theta_yi_zj,cos_theta_yi_xk,cos_theta_yi_yk,cos_theta_yi_zk,
                                      cos_theta_zi_zi,cos_theta_zi_xj,cos_theta_zi_yj,cos_theta_zi_zj,cos_theta_zi_xk,cos_theta_zi_yk,cos_theta_zi_zk,
                                      cos_theta_xj_xj,cos_theta_xj_yj,cos_theta_xj_zj,cos_theta_xj_xk,cos_theta_xj_yk,cos_theta_xj_zk,
                                      cos_theta_yj_yj,cos_theta_yj_zj,cos_theta_yj_xk,cos_theta_yj_yk,cos_theta_yj_zk,
                                      cos_theta_zj_zj,cos_theta_zj_xk,cos_theta_zj_yk,cos_theta_zj_zk,
                                      cos_theta_xk_xk,cos_theta_xk_yk,cos_theta_xk_zk,
                                      cos_theta_yk_yk,cos_theta_yk_zk,
                                      cos_theta_zk_zk])

    '''
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
    '''

    def second_deriavete_element_two_body(self,kb,bb,m1,m2,x1,x2,mode='bond'):
        xi,yi,zi,xj,yj,zj,k,b = symbols('xi yi zi xj yj zj k b', real=True)
        if (mode=='bond'):
            substitute_num = []
            for i in range(0,self.bond_element.shape[0]):
                substitute_num.append(self.bond_element[i](kb,bb,x1[0],x1[1],x1[2],x2[0],x2[1],x2[2]))

            two_body_element = np.array([[substitute_num[0],substitute_num[1],substitute_num[2],substitute_num[3],substitute_num[4],substitute_num[5]],
                                        [substitute_num[1],substitute_num[6],substitute_num[7],substitute_num[8],substitute_num[9],substitute_num[10]],
                                        [substitute_num[2],substitute_num[7],substitute_num[11],substitute_num[12],substitute_num[13],substitute_num[14]],
                                        [substitute_num[3],substitute_num[8],substitute_num[12],substitute_num[15],substitute_num[16],substitute_num[17]],
                                        [substitute_num[4],substitute_num[9],substitute_num[13],substitute_num[16],substitute_num[18],substitute_num[19]],
                                        [substitute_num[5],substitute_num[10],substitute_num[14],substitute_num[17],substitute_num[19],substitute_num[20]]])
        
            mass_reduce = np.array([[(m1**0.5*m1**0.5),(m1**0.5*m1**0.5),(m1**0.5*m1**0.5),(m2**0.5*m1**0.5),(m2**0.5*m1**0.5),(m2**0.5*m1**0.5)],
                                    [(m1**0.5*m1**0.5),(m1**0.5*m1**0.5),(m1**0.5*m1**0.5),(m2**0.5*m1**0.5),(m2**0.5*m1**0.5),(m2**0.5*m1**0.5)],
                                    [(m1**0.5*m1**0.5),(m1**0.5*m1**0.5),(m1**0.5*m1**0.5),(m2**0.5*m1**0.5),(m2**0.5*m1**0.5),(m2**0.5*m1**0.5)],
                                    [(m2**0.5*m1**0.5),(m2**0.5*m1**0.5),(m2**0.5*m1**0.5),(m2**0.5*m2**0.5),(m2**0.5*m2**0.5),(m2**0.5*m2**0.5)],
                                    [(m2**0.5*m1**0.5),(m2**0.5*m1**0.5),(m2**0.5*m1**0.5),(m2**0.5*m2**0.5),(m2**0.5*m2**0.5),(m2**0.5*m2**0.5)],
                                    [(m2**0.5*m1**0.5),(m2**0.5*m1**0.5),(m2**0.5*m1**0.5),(m2**0.5*m2**0.5),(m2**0.5*m2**0.5),(m2**0.5*m2**0.5)]])

            return (two_body_element/mass_reduce).astype('float')

        else:
            print('error XD')

    '''
    def second_deriavete_element_three_body(self,k,t,m1,m2,m3,x1,x2,x3):
        xi,yi,zi,xj,yj,zj,xk,yk,zk = symbols('xi yi zi xj yj zj xk yk zk', real=True)

        # define potential #
        cos_theta = (((xj-xi)**2+(yj-yi)**2+(zj-zi)**2)+((xk-xj)**2+(yk-yj)**2+(zk-zj)**2)-((xk-xi)**2+(yk-yi)**2+(zk-zi)))/(2*((xj-xi)**2+(yj-yi)**2+(zj-zi)**2)**0.5+((xk-xj)**2+(yk-yj)**2+(zk-zj)**2))
        potential_form = k*(cos_theta-t)**2

        cos_theta_xi_xi = (potential_form.diff(xi).diff(xi)).subs(xi,x1[0]).subs(yi,x1[1]).subs(zi,x1[2]).subs(xj,x2[0]).subs(yj,x2[1]).subs(zj,x2[2]).subs(xk,x3[0]).subs(yk,x3[1]).subs(zk,x3[2])
        cos_theta_xi_yi = (potential_form.diff(xi).diff(yi)).subs(xi,x1[0]).subs(yi,x1[1]).subs(zi,x1[2]).subs(xj,x2[0]).subs(yj,x2[1]).subs(zj,x2[2]).subs(xk,x3[0]).subs(yk,x3[1]).subs(zk,x3[2])
        cos_theta_xi_zi = (potential_form.diff(xi).diff(zi)).subs(xi,x1[0]).subs(yi,x1[1]).subs(zi,x1[2]).subs(xj,x2[0]).subs(yj,x2[1]).subs(zj,x2[2]).subs(xk,x3[0]).subs(yk,x3[1]).subs(zk,x3[2])
        cos_theta_xi_xj = (potential_form.diff(xi).diff(xj)).subs(xi,x1[0]).subs(yi,x1[1]).subs(zi,x1[2]).subs(xj,x2[0]).subs(yj,x2[1]).subs(zj,x2[2]).subs(xk,x3[0]).subs(yk,x3[1]).subs(zk,x3[2])
        cos_theta_xi_yj = (potential_form.diff(xi).diff(yj)).subs(xi,x1[0]).subs(yi,x1[1]).subs(zi,x1[2]).subs(xj,x2[0]).subs(yj,x2[1]).subs(zj,x2[2]).subs(xk,x3[0]).subs(yk,x3[1]).subs(zk,x3[2])
        cos_theta_xi_zj = (potential_form.diff(xi).diff(zj)).subs(xi,x1[0]).subs(yi,x1[1]).subs(zi,x1[2]).subs(xj,x2[0]).subs(yj,x2[1]).subs(zj,x2[2]).subs(xk,x3[0]).subs(yk,x3[1]).subs(zk,x3[2])
        cos_theta_xi_xk = (potential_form.diff(xi).diff(xk)).subs(xi,x1[0]).subs(yi,x1[1]).subs(zi,x1[2]).subs(xj,x2[0]).subs(yj,x2[1]).subs(zj,x2[2]).subs(xk,x3[0]).subs(yk,x3[1]).subs(zk,x3[2])
        cos_theta_xi_yk = (potential_form.diff(xi).diff(yk)).subs(xi,x1[0]).subs(yi,x1[1]).subs(zi,x1[2]).subs(xj,x2[0]).subs(yj,x2[1]).subs(zj,x2[2]).subs(xk,x3[0]).subs(yk,x3[1]).subs(zk,x3[2])
        cos_theta_xi_zk = (potential_form.diff(xi).diff(zk)).subs(xi,x1[0]).subs(yi,x1[1]).subs(zi,x1[2]).subs(xj,x2[0]).subs(yj,x2[1]).subs(zj,x2[2]).subs(xk,x3[0]).subs(yk,x3[1]).subs(zk,x3[2])
        
        cos_theta_yi_yi = (potential_form.diff(yi).diff(yi)).subs(xi,x1[0]).subs(yi,x1[1]).subs(zi,x1[2]).subs(xj,x2[0]).subs(yj,x2[1]).subs(zj,x2[2]).subs(xk,x3[0]).subs(yk,x3[1]).subs(zk,x3[2])
        cos_theta_yi_zi = (potential_form.diff(yi).diff(zi)).subs(xi,x1[0]).subs(yi,x1[1]).subs(zi,x1[2]).subs(xj,x2[0]).subs(yj,x2[1]).subs(zj,x2[2]).subs(xk,x3[0]).subs(yk,x3[1]).subs(zk,x3[2])
        cos_theta_yi_xj = (potential_form.diff(yi).diff(xj)).subs(xi,x1[0]).subs(yi,x1[1]).subs(zi,x1[2]).subs(xj,x2[0]).subs(yj,x2[1]).subs(zj,x2[2]).subs(xk,x3[0]).subs(yk,x3[1]).subs(zk,x3[2])
        cos_theta_yi_yj = (potential_form.diff(yi).diff(yj)).subs(xi,x1[0]).subs(yi,x1[1]).subs(zi,x1[2]).subs(xj,x2[0]).subs(yj,x2[1]).subs(zj,x2[2]).subs(xk,x3[0]).subs(yk,x3[1]).subs(zk,x3[2])
        cos_theta_yi_zj = (potential_form.diff(yi).diff(zj)).subs(xi,x1[0]).subs(yi,x1[1]).subs(zi,x1[2]).subs(xj,x2[0]).subs(yj,x2[1]).subs(zj,x2[2]).subs(xk,x3[0]).subs(yk,x3[1]).subs(zk,x3[2])
        cos_theta_yi_xk = (potential_form.diff(yi).diff(xk)).subs(xi,x1[0]).subs(yi,x1[1]).subs(zi,x1[2]).subs(xj,x2[0]).subs(yj,x2[1]).subs(zj,x2[2]).subs(xk,x3[0]).subs(yk,x3[1]).subs(zk,x3[2])
        cos_theta_yi_yk = (potential_form.diff(yi).diff(yk)).subs(xi,x1[0]).subs(yi,x1[1]).subs(zi,x1[2]).subs(xj,x2[0]).subs(yj,x2[1]).subs(zj,x2[2]).subs(xk,x3[0]).subs(yk,x3[1]).subs(zk,x3[2])
        cos_theta_yi_zk = (potential_form.diff(yi).diff(zk)).subs(xi,x1[0]).subs(yi,x1[1]).subs(zi,x1[2]).subs(xj,x2[0]).subs(yj,x2[1]).subs(zj,x2[2]).subs(xk,x3[0]).subs(yk,x3[1]).subs(zk,x3[2])
        
        cos_theta_zi_zi = (potential_form.diff(zi).diff(zi)).subs(xi,x1[0]).subs(yi,x1[1]).subs(zi,x1[2]).subs(xj,x2[0]).subs(yj,x2[1]).subs(zj,x2[2]).subs(xk,x3[0]).subs(yk,x3[1]).subs(zk,x3[2])
        cos_theta_zi_xj = (potential_form.diff(zi).diff(xj)).subs(xi,x1[0]).subs(yi,x1[1]).subs(zi,x1[2]).subs(xj,x2[0]).subs(yj,x2[1]).subs(zj,x2[2]).subs(xk,x3[0]).subs(yk,x3[1]).subs(zk,x3[2])
        cos_theta_zi_yj = (potential_form.diff(zi).diff(yj)).subs(xi,x1[0]).subs(yi,x1[1]).subs(zi,x1[2]).subs(xj,x2[0]).subs(yj,x2[1]).subs(zj,x2[2]).subs(xk,x3[0]).subs(yk,x3[1]).subs(zk,x3[2])
        cos_theta_zi_zj = (potential_form.diff(zi).diff(zj)).subs(xi,x1[0]).subs(yi,x1[1]).subs(zi,x1[2]).subs(xj,x2[0]).subs(yj,x2[1]).subs(zj,x2[2]).subs(xk,x3[0]).subs(yk,x3[1]).subs(zk,x3[2])
        cos_theta_zi_xk = (potential_form.diff(zi).diff(xk)).subs(xi,x1[0]).subs(yi,x1[1]).subs(zi,x1[2]).subs(xj,x2[0]).subs(yj,x2[1]).subs(zj,x2[2]).subs(xk,x3[0]).subs(yk,x3[1]).subs(zk,x3[2])
        cos_theta_zi_yk = (potential_form.diff(zi).diff(yk)).subs(xi,x1[0]).subs(yi,x1[1]).subs(zi,x1[2]).subs(xj,x2[0]).subs(yj,x2[1]).subs(zj,x2[2]).subs(xk,x3[0]).subs(yk,x3[1]).subs(zk,x3[2])
        cos_theta_zi_zk = (potential_form.diff(zi).diff(zk)).subs(xi,x1[0]).subs(yi,x1[1]).subs(zi,x1[2]).subs(xj,x2[0]).subs(yj,x2[1]).subs(zj,x2[2]).subs(xk,x3[0]).subs(yk,x3[1]).subs(zk,x3[2])

        cos_theta_xj_xj = (potential_form.diff(xj).diff(xj)).subs(xi,x1[0]).subs(yi,x1[1]).subs(zi,x1[2]).subs(xj,x2[0]).subs(yj,x2[1]).subs(zj,x2[2]).subs(xk,x3[0]).subs(yk,x3[1]).subs(zk,x3[2])
        cos_theta_xj_yj = (potential_form.diff(xj).diff(yj)).subs(xi,x1[0]).subs(yi,x1[1]).subs(zi,x1[2]).subs(xj,x2[0]).subs(yj,x2[1]).subs(zj,x2[2]).subs(xk,x3[0]).subs(yk,x3[1]).subs(zk,x3[2])
        cos_theta_xj_zj = (potential_form.diff(xj).diff(zj)).subs(xi,x1[0]).subs(yi,x1[1]).subs(zi,x1[2]).subs(xj,x2[0]).subs(yj,x2[1]).subs(zj,x2[2]).subs(xk,x3[0]).subs(yk,x3[1]).subs(zk,x3[2])
        cos_theta_xj_xk = (potential_form.diff(xj).diff(xk)).subs(xi,x1[0]).subs(yi,x1[1]).subs(zi,x1[2]).subs(xj,x2[0]).subs(yj,x2[1]).subs(zj,x2[2]).subs(xk,x3[0]).subs(yk,x3[1]).subs(zk,x3[2])
        cos_theta_xj_yk = (potential_form.diff(xj).diff(yk)).subs(xi,x1[0]).subs(yi,x1[1]).subs(zi,x1[2]).subs(xj,x2[0]).subs(yj,x2[1]).subs(zj,x2[2]).subs(xk,x3[0]).subs(yk,x3[1]).subs(zk,x3[2])
        cos_theta_xj_zk = (potential_form.diff(xj).diff(zk)).subs(xi,x1[0]).subs(yi,x1[1]).subs(zi,x1[2]).subs(xj,x2[0]).subs(yj,x2[1]).subs(zj,x2[2]).subs(xk,x3[0]).subs(yk,x3[1]).subs(zk,x3[2])

        cos_theta_yj_yj = (potential_form.diff(yj).diff(yj)).subs(xi,x1[0]).subs(yi,x1[1]).subs(zi,x1[2]).subs(xj,x2[0]).subs(yj,x2[1]).subs(zj,x2[2]).subs(xk,x3[0]).subs(yk,x3[1]).subs(zk,x3[2])
        cos_theta_yj_zj = (potential_form.diff(yj).diff(zj)).subs(xi,x1[0]).subs(yi,x1[1]).subs(zi,x1[2]).subs(xj,x2[0]).subs(yj,x2[1]).subs(zj,x2[2]).subs(xk,x3[0]).subs(yk,x3[1]).subs(zk,x3[2])
        cos_theta_yj_xk = (potential_form.diff(yj).diff(xk)).subs(xi,x1[0]).subs(yi,x1[1]).subs(zi,x1[2]).subs(xj,x2[0]).subs(yj,x2[1]).subs(zj,x2[2]).subs(xk,x3[0]).subs(yk,x3[1]).subs(zk,x3[2])
        cos_theta_yj_yk = (potential_form.diff(yj).diff(yk)).subs(xi,x1[0]).subs(yi,x1[1]).subs(zi,x1[2]).subs(xj,x2[0]).subs(yj,x2[1]).subs(zj,x2[2]).subs(xk,x3[0]).subs(yk,x3[1]).subs(zk,x3[2])
        cos_theta_yj_zk = (potential_form.diff(yj).diff(zk)).subs(xi,x1[0]).subs(yi,x1[1]).subs(zi,x1[2]).subs(xj,x2[0]).subs(yj,x2[1]).subs(zj,x2[2]).subs(xk,x3[0]).subs(yk,x3[1]).subs(zk,x3[2])
        
        cos_theta_zj_zj = (potential_form.diff(zj).diff(zj)).subs(xi,x1[0]).subs(yi,x1[1]).subs(zi,x1[2]).subs(xj,x2[0]).subs(yj,x2[1]).subs(zj,x2[2]).subs(xk,x3[0]).subs(yk,x3[1]).subs(zk,x3[2])
        cos_theta_zj_xk = (potential_form.diff(zj).diff(xk)).subs(xi,x1[0]).subs(yi,x1[1]).subs(zi,x1[2]).subs(xj,x2[0]).subs(yj,x2[1]).subs(zj,x2[2]).subs(xk,x3[0]).subs(yk,x3[1]).subs(zk,x3[2])
        cos_theta_zj_yk = (potential_form.diff(zj).diff(yk)).subs(xi,x1[0]).subs(yi,x1[1]).subs(zi,x1[2]).subs(xj,x2[0]).subs(yj,x2[1]).subs(zj,x2[2]).subs(xk,x3[0]).subs(yk,x3[1]).subs(zk,x3[2])
        cos_theta_zj_zk = (potential_form.diff(zj).diff(zk)).subs(xi,x1[0]).subs(yi,x1[1]).subs(zi,x1[2]).subs(xj,x2[0]).subs(yj,x2[1]).subs(zj,x2[2]).subs(xk,x3[0]).subs(yk,x3[1]).subs(zk,x3[2])

        cos_theta_xk_xk = (potential_form.diff(xk).diff(xk)).subs(xi,x1[0]).subs(yi,x1[1]).subs(zi,x1[2]).subs(xj,x2[0]).subs(yj,x2[1]).subs(zj,x2[2]).subs(xk,x3[0]).subs(yk,x3[1]).subs(zk,x3[2])
        cos_theta_xk_yk = (potential_form.diff(xk).diff(yk)).subs(xi,x1[0]).subs(yi,x1[1]).subs(zi,x1[2]).subs(xj,x2[0]).subs(yj,x2[1]).subs(zj,x2[2]).subs(xk,x3[0]).subs(yk,x3[1]).subs(zk,x3[2])
        cos_theta_xk_zk = (potential_form.diff(xk).diff(zk)).subs(xi,x1[0]).subs(yi,x1[1]).subs(zi,x1[2]).subs(xj,x2[0]).subs(yj,x2[1]).subs(zj,x2[2]).subs(xk,x3[0]).subs(yk,x3[1]).subs(zk,x3[2])

        cos_theta_yk_yk = (potential_form.diff(yk).diff(yk)).subs(xi,x1[0]).subs(yi,x1[1]).subs(zi,x1[2]).subs(xj,x2[0]).subs(yj,x2[1]).subs(zj,x2[2]).subs(xk,x3[0]).subs(yk,x3[1]).subs(zk,x3[2])
        cos_theta_yk_zk = (potential_form.diff(yk).diff(zk)).subs(xi,x1[0]).subs(yi,x1[1]).subs(zi,x1[2]).subs(xj,x2[0]).subs(yj,x2[1]).subs(zj,x2[2]).subs(xk,x3[0]).subs(yk,x3[1]).subs(zk,x3[2])

        cos_theta_zk_zk = (potential_form.diff(zk).diff(zk)).subs(xi,x1[0]).subs(yi,x1[1]).subs(zi,x1[2]).subs(xj,x2[0]).subs(yj,x2[1]).subs(zj,x2[2]).subs(xk,x3[0]).subs(yk,x3[1]).subs(zk,x3[2])

        three_body_element = np.array([[cos_theta_xi_xi,cos_theta_xi_yi,cos_theta_xi_zi,cos_theta_xi_xj,cos_theta_xi_yj,cos_theta_xi_zj,cos_theta_xi_xk,cos_theta_xi_yk,cos_theta_xi_zk],
                                       [cos_theta_xi_yi,cos_theta_yi_yi,cos_theta_yi_zi,cos_theta_yi_xj,cos_theta_yi_yj,cos_theta_yi_zj,cos_theta_yi_xk,cos_theta_yi_yk,cos_theta_yi_zk],
                                       [cos_theta_xi_zi,cos_theta_yi_zi,cos_theta_zi_zi,cos_theta_zi_xj,cos_theta_zi_yj,cos_theta_zi_zj,cos_theta_zi_xk,cos_theta_zi_yk,cos_theta_zi_zk],
                                       [cos_theta_xi_xj,cos_theta_yi_xj,cos_theta_zi_xj,cos_theta_xj_xj,cos_theta_xj_yj,cos_theta_xj_zj,cos_theta_xj_xk,cos_theta_xj_yk,cos_theta_xj_zk],
                                       [cos_theta_xi_yj,cos_theta_yi_yj,cos_theta_zi_yj,cos_theta_xj_yj,cos_theta_yj_yj,cos_theta_yj_zj,cos_theta_yj_xk,cos_theta_yj_yk,cos_theta_yj_zk],
                                       [cos_theta_xi_zj,cos_theta_yi_zj,cos_theta_zi_zj,cos_theta_xj_zj,cos_theta_yj_zj,cos_theta_zj_zj,cos_theta_zj_xk,cos_theta_zj_yk,cos_theta_zj_zk],
                                       [cos_theta_xi_xk,cos_theta_yi_xk,cos_theta_zi_xk,cos_theta_xj_xk,cos_theta_yj_xk,cos_theta_zj_xk,cos_theta_xk_xk,cos_theta_xk_yk,cos_theta_xk_zk],
                                       [cos_theta_xi_yk,cos_theta_yi_yk,cos_theta_zi_yk,cos_theta_xj_yk,cos_theta_yj_yk,cos_theta_zj_yk,cos_theta_xk_yk,cos_theta_yk_yk,cos_theta_yk_zk],
                                       [cos_theta_xi_zk,cos_theta_yi_zk,cos_theta_zi_zk,cos_theta_xj_zk,cos_theta_yj_zk,cos_theta_zj_zk,cos_theta_xk_zk,cos_theta_yk_zk,cos_theta_zk_zk]])

        mass_reduce = np.array([[(m1**0.5*m1**0.5),(m1**0.5*m1**0.5),(m1**0.5*m1**0.5),(m1**0.5*m2**0.5),(m1**0.5*m2**0.5),(m1**0.5*m2**0.5),(m1**0.5*m3**0.5),(m1**0.5*m3**0.5),(m1**0.5*m3**0.5)],
                                [(m1**0.5*m1**0.5),(m1**0.5*m1**0.5),(m1**0.5*m1**0.5),(m1**0.5*m2**0.5),(m1**0.5*m2**0.5),(m1**0.5*m2**0.5),(m1**0.5*m3**0.5),(m1**0.5*m3**0.5),(m1**0.5*m3**0.5)],
                                [(m1**0.5*m1**0.5),(m1**0.5*m1**0.5),(m1**0.5*m1**0.5),(m1**0.5*m2**0.5),(m1**0.5*m2**0.5),(m1**0.5*m2**0.5),(m1**0.5*m3**0.5),(m1**0.5*m3**0.5),(m1**0.5*m3**0.5)],
                                [(m1**0.5*m2**0.5),(m1**0.5*m2**0.5),(m1**0.5*m2**0.5),(m2**0.5*m2**0.5),(m2**0.5*m2**0.5),(m2**0.5*m2**0.5),(m2**0.5*m3**0.5),(m2**0.5*m3**0.5),(m2**0.5*m3**0.5)],
                                [(m1**0.5*m2**0.5),(m1**0.5*m2**0.5),(m1**0.5*m2**0.5),(m2**0.5*m2**0.5),(m2**0.5*m2**0.5),(m2**0.5*m2**0.5),(m2**0.5*m3**0.5),(m2**0.5*m3**0.5),(m2**0.5*m3**0.5)],
                                [(m1**0.5*m2**0.5),(m1**0.5*m2**0.5),(m1**0.5*m2**0.5),(m2**0.5*m2**0.5),(m2**0.5*m2**0.5),(m2**0.5*m2**0.5),(m2**0.5*m3**0.5),(m2**0.5*m3**0.5),(m2**0.5*m3**0.5)],
                                [(m1**0.5*m3**0.5),(m1**0.5*m3**0.5),(m1**0.5*m3**0.5),(m2**0.5*m3**0.5),(m2**0.5*m3**0.5),(m2**0.5*m3**0.5),(m3**0.5*m3**0.5),(m3**0.5*m3**0.5),(m3**0.5*m3**0.5)],
                                [(m1**0.5*m3**0.5),(m1**0.5*m3**0.5),(m1**0.5*m3**0.5),(m2**0.5*m3**0.5),(m2**0.5*m3**0.5),(m2**0.5*m3**0.5),(m3**0.5*m3**0.5),(m3**0.5*m3**0.5),(m3**0.5*m3**0.5)],
                                [(m1**0.5*m3**0.5),(m1**0.5*m3**0.5),(m1**0.5*m3**0.5),(m2**0.5*m3**0.5),(m2**0.5*m3**0.5),(m2**0.5*m3**0.5),(m3**0.5*m3**0.5),(m3**0.5*m3**0.5),(m3**0.5*m3**0.5)]])

        return (three_body_element/mass_reduce).astype('float')
    '''

    def second_deriavete_element_three_body(self,kb,bb,m1,m2,m3,x1,x2,x3):
        xi,yi,zi,xj,yj,zj,xk,yk,zk,k,b = symbols('xi yi zi xj yj zj xk yk zk k b', real=True)

        substitute_num = []
        
        for i in range(0,self.angle_element.shape[0]):
            substitute_num.append(self.angle_element[i](kb,bb,x1[0],x1[1],x1[2],x2[0],x2[1],x2[2],x3[0],x3[1],x3[2]))
        
        three_body_element = np.array([[substitute_num[0],substitute_num[1],substitute_num[2],substitute_num[3],substitute_num[4],substitute_num[5],substitute_num[6],substitute_num[7],substitute_num[8]],
                                       [substitute_num[1],substitute_num[9],substitute_num[10],substitute_num[11],substitute_num[12],substitute_num[13],substitute_num[14],substitute_num[15],substitute_num[16]],
                                       [substitute_num[2],substitute_num[10],substitute_num[17],substitute_num[18],substitute_num[19],substitute_num[20],substitute_num[21],substitute_num[22],substitute_num[23]],
                                       [substitute_num[3],substitute_num[11],substitute_num[18],substitute_num[24],substitute_num[25],substitute_num[26],substitute_num[27],substitute_num[28],substitute_num[29]],
                                       [substitute_num[4],substitute_num[12],substitute_num[19],substitute_num[25],substitute_num[30],substitute_num[31],substitute_num[32],substitute_num[33],substitute_num[34]],
                                       [substitute_num[5],substitute_num[13],substitute_num[20],substitute_num[26],substitute_num[31],substitute_num[35],substitute_num[36],substitute_num[37],substitute_num[38]],
                                       [substitute_num[6],substitute_num[14],substitute_num[21],substitute_num[27],substitute_num[32],substitute_num[36],substitute_num[39],substitute_num[40],substitute_num[41]],
                                       [substitute_num[7],substitute_num[15],substitute_num[22],substitute_num[28],substitute_num[33],substitute_num[37],substitute_num[40],substitute_num[42],substitute_num[43]],
                                       [substitute_num[8],substitute_num[16],substitute_num[23],substitute_num[29],substitute_num[34],substitute_num[38],substitute_num[41],substitute_num[43],substitute_num[44]]])
        
        mass_reduce = np.array([[(m1**0.5*m1**0.5),(m1**0.5*m1**0.5),(m1**0.5*m1**0.5),(m1**0.5*m2**0.5),(m1**0.5*m2**0.5),(m1**0.5*m2**0.5),(m1**0.5*m3**0.5),(m1**0.5*m3**0.5),(m1**0.5*m3**0.5)],
                                [(m1**0.5*m1**0.5),(m1**0.5*m1**0.5),(m1**0.5*m1**0.5),(m1**0.5*m2**0.5),(m1**0.5*m2**0.5),(m1**0.5*m2**0.5),(m1**0.5*m3**0.5),(m1**0.5*m3**0.5),(m1**0.5*m3**0.5)],
                                [(m1**0.5*m1**0.5),(m1**0.5*m1**0.5),(m1**0.5*m1**0.5),(m1**0.5*m2**0.5),(m1**0.5*m2**0.5),(m1**0.5*m2**0.5),(m1**0.5*m3**0.5),(m1**0.5*m3**0.5),(m1**0.5*m3**0.5)],
                                [(m1**0.5*m2**0.5),(m1**0.5*m2**0.5),(m1**0.5*m2**0.5),(m2**0.5*m2**0.5),(m2**0.5*m2**0.5),(m2**0.5*m2**0.5),(m2**0.5*m3**0.5),(m2**0.5*m3**0.5),(m2**0.5*m3**0.5)],
                                [(m1**0.5*m2**0.5),(m1**0.5*m2**0.5),(m1**0.5*m2**0.5),(m2**0.5*m2**0.5),(m2**0.5*m2**0.5),(m2**0.5*m2**0.5),(m2**0.5*m3**0.5),(m2**0.5*m3**0.5),(m2**0.5*m3**0.5)],
                                [(m1**0.5*m2**0.5),(m1**0.5*m2**0.5),(m1**0.5*m2**0.5),(m2**0.5*m2**0.5),(m2**0.5*m2**0.5),(m2**0.5*m2**0.5),(m2**0.5*m3**0.5),(m2**0.5*m3**0.5),(m2**0.5*m3**0.5)],
                                [(m1**0.5*m3**0.5),(m1**0.5*m3**0.5),(m1**0.5*m3**0.5),(m2**0.5*m3**0.5),(m2**0.5*m3**0.5),(m2**0.5*m3**0.5),(m3**0.5*m3**0.5),(m3**0.5*m3**0.5),(m3**0.5*m3**0.5)],
                                [(m1**0.5*m3**0.5),(m1**0.5*m3**0.5),(m1**0.5*m3**0.5),(m2**0.5*m3**0.5),(m2**0.5*m3**0.5),(m2**0.5*m3**0.5),(m3**0.5*m3**0.5),(m3**0.5*m3**0.5),(m3**0.5*m3**0.5)],
                                [(m1**0.5*m3**0.5),(m1**0.5*m3**0.5),(m1**0.5*m3**0.5),(m2**0.5*m3**0.5),(m2**0.5*m3**0.5),(m2**0.5*m3**0.5),(m3**0.5*m3**0.5),(m3**0.5*m3**0.5),(m3**0.5*m3**0.5)]])

        return (three_body_element/mass_reduce).astype('float')

    #def dihedral_K(self,x1,x2,x3,x4):
    
    #def nonbonded_K(self,x1,x2)

    def build_matrix(self):
        self.element_initialization()

        # initial hessian matrix #
        hm = np.zeros((self.mass_type.shape[0]*3,self.mass_type.shape[0]*3))
  
        num_cout = 0
        print("Build bond potential")
        bond_times = tqdm(total=self.bond_index.shape[0])
        for i,j in self.bond_index*3:

            k_n = self.second_deriavete_element_two_body(self.bond_par[num_cout,0],self.bond_par[num_cout,1],self.mass_type[int(i/3)],self.mass_type[int(j/3)],self.position[int(i/3)],self.position[int(j/3)])
            
            hm[np.ix_([i,i+1,i+2,j,j+1,j+2],[i,i+1,i+2,j,j+1,j+2])] += k_n
            
            bond_times.update(1)
            num_cout += 1

        print("Build angle potential")
        num_cout = 0
        angle_times = tqdm(total=self.angle_index.shape[0])
        for i,j,k in self.angle_index*3:
            
            ka_n = self.second_deriavete_element_three_body(self.angle_par[num_cout,0],np.deg2rad(self.angle_par[num_cout,1]),self.mass_type[int(i/3)],self.mass_type[int(j/3)],self.mass_type[int(k/3)],self.position[int(i/3)],self.position[int(j/3)],self.position[int(k/3)])
            
            hm[np.ix_([i,i+1,i+2,j,j+1,j+2,k,k+1,k+2],[i,i+1,i+2,j,j+1,j+2,k,k+1,k+2])] += ka_n

            angle_times.update(1)
            num_cout += 1

        print("hessian mattrix builded")
        return hm

    #def energy_minimize(self):

    def solve_Hessian(self,h):
       h_val, h_vec = jax.numpy.linalg.eig(h)
       return h_val, h_vec
    
    #def PCA_frequence(self,n=20):



