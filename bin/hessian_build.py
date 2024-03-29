import numpy as np
from sympy import symbols, acos, diff, sin, cos, lambdify
from tqdm.auto  import tqdm
#import cupy as cp
import warnings 
import time
import jax
from scipy.spatial import distance_matrix
import numba as nb

warnings.filterwarnings('ignore')

class Hessian:

    def __init__(self,index,potential,pos,mode='bad'):
        if (mode == 'bad'):
            self.bond_index = np.sort(index[0],axis=1)
            self.bond_index = index[0]
            self.angle_index = index[1]
            self.dihedral_index = index[2]
            self.improper_index = index[3]
            self.nonbonded_index = index[4]
            self.nonbond_par = index[5]
            self.position = pos
            self.real_type = index[6]
            self.mass_type = index[7]
            
            self.bond_par = potential[0]
            self.angle_par = potential[1][0]
            self.UB_par = potential[1][1]
            self.dihedral_par = potential[2]
            self.improper_par = potential[3]

        else:
            print('Error XD')

    def check_symmetric(self,a, tol=1e-8):
        return np.all(np.abs(a-a.T) < tol)

    def check_distance2vdw(self,cutoff = 12):
        print("check pair distance")
        # without 1-2, 1-3, 1-4 (Need careful) #
        pos_table = distance_matrix(self.position,self.position)
        cutoff_dis_list = np.where((np.triu(pos_table,1) <= cutoff) & (np.triu(pos_table,1)!=0))
        cutoff_dis_list_ = np.sort(np.vstack((cutoff_dis_list[0][:],cutoff_dis_list[1][:])).T,axis=1)
        '''
        no_12 = np.array(list(set(map(tuple, cutoff_dis_list_)) - set(map(tuple, self.bond_index))))
        no_12_ = np.array(list(set(map(tuple, no_12)) - set(map(tuple, self.bond_index[:,[1,0]]))))
        no_123 = np.array(list(set(map(tuple, no_12_)) - set(map(tuple, self.angle_index[:,[0,2]]))))
        no_123_ = np.array(list(set(map(tuple, no_123)) - set(map(tuple, self.angle_index[:,[2,0]]))))
        no_1234 = np.array(list(set(map(tuple, no_123_)) - set(map(tuple, self.dihedral_index[:,[0,3]]))))
        no_1234_ = np.array(list(set(map(tuple, no_1234)) - set(map(tuple, self.dihedral_index[:,[3,0]]))))
        no_1234_1 = np.array(list(set(map(tuple, no_1234_)) - set(map(tuple, self.improper_index[:,[0,1]]))))
        no_1234_2 = np.array(list(set(map(tuple, no_1234_1)) - set(map(tuple, self.improper_index[:,[1,0]]))))
        '''
        no_neighber = np.array(list(set(map(tuple, cutoff_dis_list_))- 
                                    set(map(tuple, self.bond_index))-
                                    set(map(tuple, self.bond_index[:,[1,0]]))-
                                    set(map(tuple, self.angle_index[:,[0,2]]))-
                                    set(map(tuple, self.angle_index[:,[2,0]]))
                                    #set(map(tuple, self.dihedral_index[:,[0,3]]))-
                                    #set(map(tuple, self.dihedral_index[:,[3,0]]))-
                                    #set(map(tuple, self.improper_index[:,[0,2]]))-
                                    #set(map(tuple, self.improper_index[:,[2,0]]))-
                                    #set(map(tuple, self.improper_index[:,[0,3]]))-
                                    #set(map(tuple, self.improper_index[:,[3,0]]))
                                    ))
        self.vdw_index = no_neighber.reshape((-1,2))
        #print(no_neighber.shape)
        
        # mixture potential (mix arithmetic)#
        print("Build mix arithmetic table")
        vdw_times_2 = tqdm(total=self.vdw_index.shape[0],ncols=100)
        vdw_pot = []
        for i, j in self.vdw_index:
            eps = (self.nonbond_par[np.where(self.nonbonded_index==self.real_type[i])[0],0]*self.nonbond_par[np.where(self.nonbonded_index==self.real_type[j])[0],0])**0.5
            sig = 0.5*(self.nonbond_par[np.where(self.nonbonded_index==self.real_type[i])[0],1]+self.nonbond_par[np.where(self.nonbonded_index==self.real_type[j])[0],1])
            vdw_pot.append([eps,sig])
            vdw_times_2.update()
        self.pair_par = np.array(vdw_pot).reshape(-1,2)
        vdw_times_2.close()
        print("Done") 
        
    def element_initialization(self):
        print("Element initialization!")

        xi,yi,zi,xj,yj,zj,xk,yk,zk,xl,yl,zl = symbols('xi yi zi xj yj zj xk yk zk xl yl zl', real=True)

        # bond potential #
        bond_rij = ((xj-xi)**2+(yj-yi)**2+(zj-zi)**2)**0.5
        bond_table = np.array([[xi,xi],[xi,yi],[xi,zi],[xi,xj],[xi,yj],[xi,zj],
                               [yi,yi],[yi,zi],[yi,xj],[yi,yj],[yi,zj],
                               [zi,zi],[zi,xj],[zi,yj],[zi,zj],
                               [xj,xj],[xj,yj],[xj,zj],
                               [yj,yj],[yj,zj],
                               [zj,zj]])
        b_t = []
        print("Two body element loading ")
        bond_times = tqdm(total=bond_table.shape[0],ncols=100)
        for i,j in bond_table:
            #b_t.append(lambdify((k,b,xi,yi,zi,xj,yj,zj),(bond_potential_form.diff(i).diff(j)),'numpy'))       
            b_t.append(lambdify((xi,yi,zi,xj,yj,zj),(bond_rij.diff(i)*bond_rij.diff(j)),'numpy'))
            bond_times.update()
        self.bond_element = np.array(b_t)
        bond_times.close()
        
        # angle potential #
        #cos_theta = (((xj-xi)**2+(yj-yi)**2+(zj-zi)**2)+((xk-xj)**2+(yk-yj)**2+(zk-zj)**2)-((xk-xi)**2+(yk-yi)**2+(zk-zi)))/(2*((xj-xi)**2+(yj-yi)**2+(zj-zi)**2)**0.5+((xk-xj)**2+(yk-yj)**2+(zk-zj)**2))
        #angle_potential_form = k*(cos_theta-b)**2
        theta = acos((((xj-xi)**2+(yj-yi)**2+(zj-zi)**2)+((xk-xj)**2+(yk-yj)**2+(zk-zj)**2)-((xk-xi)**2+(yk-yi)**2+(zk-zi)**2))/(2*((xj-xi)**2+(yj-yi)**2+(zj-zi)**2)**0.5*((xk-xj)**2+(yk-yj)**2+(zk-zj)**2)**0.5))
        angle_table = np.array([[xi,xi],[xi,yi],[xi,zi],[xi,xj],[xi,yj],[xi,zj],[xi,xk],[xi,yk],[xi,zk],
                                [yi,yi],[yi,zi],[yi,xj],[yi,yj],[yi,zj],[yi,xk],[yi,yk],[yi,zk],
                                [zi,zi],[zi,xj],[zi,yj],[zi,zj],[zi,xk],[zi,yk],[zi,zk],
                                [xj,xj],[xj,yj],[xj,zj],[xj,xk],[xj,yk],[xj,zk],
                                [yj,yj],[yj,zj],[yj,xk],[yj,yk],[yj,zk],
                                [zj,zj],[zj,xk],[zj,yk],[zj,zk],
                                [xk,xk],[xk,yk],[xk,zk],
                                [yk,yk],[yk,zk],
                                [zk,zk]])  
        a_t = []
        print("Three body element loading ")
        angle_times = tqdm(total=angle_table.shape[0],ncols=100)
        for i,j in angle_table: 
            a_t.append(lambdify((xi,yi,zi,xj,yj,zj,xk,yk,zk),(theta.diff(i)*theta.diff(j)),'numpy'))
            angle_times.update()
        self.angle_element = np.array(a_t)
        angle_times.close()
        
        # dihedral potential #
        #v1 = (xj-xi,yj-yi,zj-zi)
        #v2 = (xk-xj,yk-yj,zk-zj)
        #v3 = (xl-xk,yl-yk,zl-zk)
        w1n=(((yj-yi)*(zk-zj)-(zj-zi)*(yk-yj))**2+((zj-zi)*(xk-xj)-(xj-xi)*(zk-zj))**2+((xj-xi)*(yk-yj)-(yj-yi)*(xk-xj))**2)**0.5
        w2n=(((yk-yj)*(zl-zk)-(zk-zj)*(yl-yk))**2+((zk-zj)*(xl-xk)-(xk-xj)*(zl-zk))**2+((xk-xj)*(yl-yk)-(yk-yj)*(xl-xk))**2)**0.5
        w1 = [((yj-yi)*(zk-zj)-(zj-zi)*(yk-yj))/w1n,((zj-zi)*(xk-xj)-(xj-xi)*(zk-zj))/w1n,((xj-xi)*(yk-yj)-(yj-yi)*(xk-xj))/w1n]
        w2 = [((yk-yj)*(zl-zk)-(zk-zj)*(yl-yk))/w2n,((zk-zj)*(xl-xk)-(xk-xj)*(zl-zk))/w2n,((xk-xj)*(yl-yk)-(yk-yj)*(xl-xk))/w2n]
        

        #phi = acos(((((yj-yi)*(zk-zj)-(zj-zi)*(yk-yj))*((yk-yj)*(zl-zk)-(zk-zj)*(yl-yk)))+(((zj-zi)*(xk-xj)-(xj-xi)*(zk-zj))*((zk-zj)*(xl-xk)-(xk-xj)*(zl-zk)))
        #       +(((xj-xi)*(yk-yj)-(yj-yi)*(xk-xj))*((xk-xj)*(yl-yk)-(yk-yj)*(xl-xk))))/((((yj-yi)*(zk-zj)-(zj-zi)*(yk-yj))**2+((zj-zi)*(xk-xj)-(xj-xi)*(zk-zj))**2+((xj-xi)*(yk-yj)-(yj-yi)*(xk-xj))**2)**0.5*
        #                                                                                (((yk-yj)*(zl-zk)-(zk-zj)*(yl-yk))**2+((zk-zj)*(xl-xk)-(xk-xj)*(zl-zk))**2+((yl-yk)-(yk-yj)*(xl-xk))**2)**0.5))
        phi = acos((w1[0]*w2[0]+w1[1]*w2[1]+w1[2]*w2[2])/((w1[0]**2+w1[1]**2+w1[2]**2)**0.5*(w2[0]**2+w2[1]**2+w2[2]**2)**0.5))

        #dihedral_potential_form = k*(1-cos(phi*n-b))
        dihedral_table = np.array([[xi,xi],[xi,yi],[xi,zi],[xi,xj],[xi,yj],[xi,zj],[xi,xk],[xi,yk],[xi,zk],[xi,xl],[xi,yl],[xi,zl],
                                   [yi,yi],[yi,zi],[yi,xj],[yi,yj],[yi,zj],[yi,xk],[yi,yk],[yi,zk],[yi,xl],[yi,yl],[yi,zl],
                                   [zi,zi],[zi,xj],[zi,yj],[zi,zj],[zi,xk],[zi,yk],[zi,zk],[zi,xl],[zi,yl],[zi,zl],
                                   [xj,xj],[xj,yj],[xj,zj],[xj,xk],[xj,yk],[xj,zk],[xj,xl],[xj,yl],[xj,zl],
                                   [yj,yj],[yj,zj],[yj,xk],[yj,yk],[yj,zk],[yj,xl],[yj,yl],[yj,zl],
                                   [zj,zj],[zj,xk],[zj,yk],[zj,zk],[zj,xl],[zj,yl],[zj,zl],
                                   [xk,xk],[xk,yk],[xk,zk],[xk,xl],[xk,yl],[xk,zl],
                                   [yk,yk],[yk,zk],[yk,xl],[yk,yl],[yk,zl],
                                   [zk,zk],[zk,xl],[zk,yl],[zk,zl],
                                   [xl,xl],[xl,yl],[xl,zl],
                                   [yl,yl],[yl,zl],
                                   [zl,zl]])
        di_t = []
        print("Four body element loading ")
        dihedral_times = tqdm(total=dihedral_table.shape[0],ncols=100)
        for i,j in dihedral_table:
            di_t.append(lambdify((xi,yi,zi,xj,yj,zj,xk,yk,zk,xl,yl,zl),(phi.diff(i)*phi.diff(j)),'numpy'))
            dihedral_times.update()
        self.dihedral_element  = np.array(di_t)
        dihedral_times.close()
        
        print("Done")
    
    #@nb.jit
    def second_deriavete_element_two_body(self,kb,bb,m1,m2,x1,x2,mode='bond'):
        substitute_num = []
        if (mode=='bond'):
            for i in range(0,self.bond_element.shape[0]):
                substitute_num.append(self.bond_element[i](x1[0],x1[1],x1[2],x2[0],x2[1],x2[2])*kb*2)
        elif(mode=='vdw'):
            k_vdw = ((12*kb)/np.linalg.norm(x2-x1)**2)*(13*(bb/np.linalg.norm(x2-x1))**12-7*(bb/np.linalg.norm(x2-x1))**6)
            if(k_vdw<0):
                k_vdw=0
            elif(k_vdw>10):
                k_vdw=10
            for i in range(0,self.bond_element.shape[0]):
                substitute_num.append(self.bond_element[i](x1[0],x1[1],x1[2],x2[0],x2[1],x2[2])*k_vdw)
        else:
            print("error XD")
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

    #@nb.jit
    def second_deriavete_element_three_body(self,kb,bb,m1,m2,m3,x1,x2,x3):
        substitute_num = []
        
        for i in range(0,self.angle_element.shape[0]):
            substitute_num.append(self.angle_element[i](x1[0],x1[1],x1[2],x2[0],x2[1],x2[2],x3[0],x3[1],x3[2])*kb*2)
        
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

    #@nb.jit
    def second_deriavete_element_four_body(self,kb,bb,nn,m1,m2,m3,m4,x1,x2,x3,x4,mode='dihedral'): 
        #xi,yi,zi,xj,yj,zj,xk,yk,zk,xl,yl,zl,k,b,n = symbols('xi yi zi xj yj zj xk yk zk xl yl zl k b n', real=True)
        substitute_num = []
        if(mode=='dihedral'):   
            for i in range(0,self.dihedral_element.shape[0]):
                substitute_num.append(self.dihedral_element[i](x1[0],x1[1],x1[2],x2[0],x2[1],x2[2],x3[0],x3[1],x3[2],x4[0],x4[1],x4[2])*abs(nn**2*kb))  
        elif(mode=='improper'):
            for i in range(0,self.dihedral_element.shape[0]):
                substitute_num.append(self.dihedral_element[i](x1[0],x1[1],x1[2],x2[0],x2[1],x2[2],x3[0],x3[1],x3[2],x4[0],x4[1],x4[2])*2*kb)  
        else:
            print('error XD')

        four_body_element = np.array([[substitute_num[0],substitute_num[1],substitute_num[2],substitute_num[3],substitute_num[4],substitute_num[5],substitute_num[6],substitute_num[7],substitute_num[8],substitute_num[9],substitute_num[10],substitute_num[11]],
                                    [substitute_num[1],substitute_num[12],substitute_num[13],substitute_num[14],substitute_num[15],substitute_num[16],substitute_num[17],substitute_num[18],substitute_num[19],substitute_num[20],substitute_num[21],substitute_num[22]],
                                    [substitute_num[2],substitute_num[13],substitute_num[23],substitute_num[24],substitute_num[25],substitute_num[26],substitute_num[27],substitute_num[28],substitute_num[29],substitute_num[30],substitute_num[31],substitute_num[32]],
                                    [substitute_num[3],substitute_num[14],substitute_num[24],substitute_num[33],substitute_num[34],substitute_num[35],substitute_num[36],substitute_num[37],substitute_num[38],substitute_num[39],substitute_num[40],substitute_num[41]],
                                    [substitute_num[4],substitute_num[15],substitute_num[25],substitute_num[34],substitute_num[42],substitute_num[43],substitute_num[44],substitute_num[45],substitute_num[46],substitute_num[47],substitute_num[48],substitute_num[49]],
                                    [substitute_num[5],substitute_num[16],substitute_num[26],substitute_num[35],substitute_num[43],substitute_num[50],substitute_num[51],substitute_num[52],substitute_num[53],substitute_num[54],substitute_num[55],substitute_num[56]],
                                    [substitute_num[6],substitute_num[17],substitute_num[27],substitute_num[36],substitute_num[44],substitute_num[51],substitute_num[57],substitute_num[58],substitute_num[59],substitute_num[60],substitute_num[61],substitute_num[62]],
                                    [substitute_num[7],substitute_num[18],substitute_num[28],substitute_num[37],substitute_num[45],substitute_num[52],substitute_num[58],substitute_num[63],substitute_num[64],substitute_num[65],substitute_num[66],substitute_num[67]],
                                    [substitute_num[8],substitute_num[19],substitute_num[29],substitute_num[38],substitute_num[46],substitute_num[53],substitute_num[59],substitute_num[64],substitute_num[68],substitute_num[69],substitute_num[70],substitute_num[71]],
                                    [substitute_num[9],substitute_num[20],substitute_num[30],substitute_num[39],substitute_num[47],substitute_num[54],substitute_num[60],substitute_num[65],substitute_num[69],substitute_num[72],substitute_num[73],substitute_num[74]],
                                    [substitute_num[10],substitute_num[21],substitute_num[31],substitute_num[40],substitute_num[48],substitute_num[55],substitute_num[61],substitute_num[66],substitute_num[70],substitute_num[73],substitute_num[75],substitute_num[76]],
                                    [substitute_num[11],substitute_num[22],substitute_num[32],substitute_num[41],substitute_num[49],substitute_num[56],substitute_num[62],substitute_num[67],substitute_num[71],substitute_num[74],substitute_num[76],substitute_num[77]]])
        
        mass_reduce = np.array([[(m1**0.5*m1**0.5),(m1**0.5*m1**0.5),(m1**0.5*m1**0.5),(m2**0.5*m1**0.5),(m2**0.5*m1**0.5),(m2**0.5*m1**0.5),(m3**0.5*m1**0.5),(m3**0.5*m1**0.5),(m3**0.5*m1**0.5),(m4**0.5*m1**0.5),(m4**0.5*m1**0.5),(m4**0.5*m1**0.5)],
                                [(m1**0.5*m1**0.5),(m1**0.5*m1**0.5),(m1**0.5*m1**0.5),(m2**0.5*m1**0.5),(m2**0.5*m1**0.5),(m2**0.5*m1**0.5),(m3**0.5*m1**0.5),(m3**0.5*m1**0.5),(m3**0.5*m1**0.5),(m4**0.5*m1**0.5),(m4**0.5*m1**0.5),(m4**0.5*m1**0.5)],
                                [(m1**0.5*m1**0.5),(m1**0.5*m1**0.5),(m1**0.5*m1**0.5),(m2**0.5*m1**0.5),(m2**0.5*m1**0.5),(m2**0.5*m1**0.5),(m3**0.5*m1**0.5),(m3**0.5*m1**0.5),(m3**0.5*m1**0.5),(m4**0.5*m1**0.5),(m4**0.5*m1**0.5),(m4**0.5*m1**0.5)],
                                [(m1**0.5*m2**0.5),(m1**0.5*m2**0.5),(m1**0.5*m2**0.5),(m2**0.5*m2**0.5),(m2**0.5*m2**0.5),(m2**0.5*m2**0.5),(m3**0.5*m2**0.5),(m3**0.5*m2**0.5),(m3**0.5*m2**0.5),(m4**0.5*m2**0.5),(m4**0.5*m2**0.5),(m4**0.5*m2**0.5)],
                                [(m1**0.5*m2**0.5),(m1**0.5*m2**0.5),(m1**0.5*m2**0.5),(m2**0.5*m2**0.5),(m2**0.5*m2**0.5),(m2**0.5*m2**0.5),(m3**0.5*m2**0.5),(m3**0.5*m2**0.5),(m3**0.5*m2**0.5),(m4**0.5*m2**0.5),(m4**0.5*m2**0.5),(m4**0.5*m2**0.5)],
                                [(m1**0.5*m2**0.5),(m1**0.5*m2**0.5),(m1**0.5*m2**0.5),(m2**0.5*m2**0.5),(m2**0.5*m2**0.5),(m2**0.5*m2**0.5),(m3**0.5*m2**0.5),(m3**0.5*m2**0.5),(m3**0.5*m2**0.5),(m4**0.5*m2**0.5),(m4**0.5*m2**0.5),(m4**0.5*m2**0.5)],
                                [(m1**0.5*m3**0.5),(m1**0.5*m3**0.5),(m1**0.5*m3**0.5),(m2**0.5*m3**0.5),(m2**0.5*m3**0.5),(m2**0.5*m3**0.5),(m3**0.5*m3**0.5),(m3**0.5*m3**0.5),(m3**0.5*m3**0.5),(m4**0.5*m3**0.5),(m4**0.5*m3**0.5),(m4**0.5*m3**0.5)],
                                [(m1**0.5*m3**0.5),(m1**0.5*m3**0.5),(m1**0.5*m3**0.5),(m2**0.5*m3**0.5),(m2**0.5*m3**0.5),(m2**0.5*m3**0.5),(m3**0.5*m3**0.5),(m3**0.5*m3**0.5),(m3**0.5*m3**0.5),(m4**0.5*m3**0.5),(m4**0.5*m3**0.5),(m4**0.5*m3**0.5)],
                                [(m1**0.5*m3**0.5),(m1**0.5*m3**0.5),(m1**0.5*m3**0.5),(m2**0.5*m3**0.5),(m2**0.5*m3**0.5),(m2**0.5*m3**0.5),(m3**0.5*m3**0.5),(m3**0.5*m3**0.5),(m3**0.5*m3**0.5),(m4**0.5*m3**0.5),(m4**0.5*m3**0.5),(m4**0.5*m3**0.5)],
                                [(m1**0.5*m4**0.5),(m1**0.5*m4**0.5),(m1**0.5*m4**0.5),(m2**0.5*m4**0.5),(m2**0.5*m4**0.5),(m2**0.5*m4**0.5),(m3**0.5*m4**0.5),(m3**0.5*m4**0.5),(m3**0.5*m4**0.5),(m4**0.5*m4**0.5),(m4**0.5*m4**0.5),(m4**0.5*m4**0.5)],
                                [(m1**0.5*m4**0.5),(m1**0.5*m4**0.5),(m1**0.5*m4**0.5),(m2**0.5*m4**0.5),(m2**0.5*m4**0.5),(m2**0.5*m4**0.5),(m3**0.5*m4**0.5),(m3**0.5*m4**0.5),(m3**0.5*m4**0.5),(m4**0.5*m4**0.5),(m4**0.5*m4**0.5),(m4**0.5*m4**0.5)],
                                [(m1**0.5*m4**0.5),(m1**0.5*m4**0.5),(m1**0.5*m4**0.5),(m2**0.5*m4**0.5),(m2**0.5*m4**0.5),(m2**0.5*m4**0.5),(m3**0.5*m4**0.5),(m3**0.5*m4**0.5),(m3**0.5*m4**0.5),(m4**0.5*m4**0.5),(m4**0.5*m4**0.5),(m4**0.5*m4**0.5)]])

        return (four_body_element/mass_reduce).astype('float')
            
    #@nb.jit
    def build_matrix(self):
        # second deriavete element #
        self.element_initialization()
        
        # Found Pairwise #
        self.check_distance2vdw(cutoff=12)

        # initial hessian matrix #
        hm = np.zeros((self.mass_type.shape[0]*3,self.mass_type.shape[0]*3))

        print("Build bonded two-body potential")
        # bond #
        num_cout = 0
        tb_times = tqdm(total=self.bond_index.shape[0]+self.angle_index.shape[0],ncols=100)
        for i,j in self.bond_index*3:
            k_n = self.second_deriavete_element_two_body(self.bond_par[num_cout,0],self.bond_par[num_cout,1],
                                                         self.mass_type[int(i/3)],self.mass_type[int(j/3)],
                                                         self.position[int(i/3)],self.position[int(j/3)])
            hm[np.ix_([i,i+1,i+2,j,j+1,j+2],[i,i+1,i+2,j,j+1,j+2])] += k_n
            tb_times.update(1)
            num_cout += 1
        # UB #
        num_cout = 0
        for i,x_,j in self.angle_index*3:
            k_ub = self.second_deriavete_element_two_body(self.UB_par[num_cout,0],self.UB_par[num_cout,1],
                                                         self.mass_type[int(i/3)],self.mass_type[int(j/3)],
                                                         self.position[int(i/3)],self.position[int(j/3)])
            
            hm[np.ix_([i,i+1,i+2,j,j+1,j+2],[i,i+1,i+2,j,j+1,j+2])] += k_ub
            tb_times.update(1)
        tb_times.close()
        
        print("Build bonded three-body potential")
        # angle #
        num_cout = 0
        trb_times = tqdm(total=self.angle_index.shape[0],ncols=100)
        for i,j,k in self.angle_index*3:
            
            ka_n = self.second_deriavete_element_three_body(self.angle_par[num_cout,0],np.deg2rad(self.angle_par[num_cout,1]),
                                                            self.mass_type[int(i/3)],self.mass_type[int(j/3)],self.mass_type[int(k/3)],
                                                            self.position[int(i/3)],self.position[int(j/3)],self.position[int(k/3)])
            
            hm[np.ix_([i,i+1,i+2,j,j+1,j+2,k,k+1,k+2],[i,i+1,i+2,j,j+1,j+2,k,k+1,k+2])] += ka_n

            trb_times.update(1)
            num_cout += 1
        trb_times.close()

        
        print("Build bonded four-body potential")
        # dihedral #
        num_cout = 0
        aa = 0
        fb_times = tqdm(total=self.dihedral_index.shape[0]+self.improper_index.shape[0],ncols=100)
        for i,j,k,l in self.dihedral_index*3: 
            kd_n = self.second_deriavete_element_four_body(self.dihedral_par[num_cout,0],np.deg2rad(self.dihedral_par[num_cout,2]),self.dihedral_par[num_cout,1],
                                                           self.mass_type[int(i/3)],self.mass_type[int(j/3)],self.mass_type[int(k/3)],self.mass_type[int(l/3)],
                                                           self.position[int(i/3)],self.position[int(j/3)],self.position[int(k/3)],self.position[int(l/3)])
            
            hm[np.ix_([i,i+1,i+2,j,j+1,j+2,k,k+1,k+2,l,l+1,l+2],[i,i+1,i+2,j,j+1,j+2,k,k+1,k+2,l,l+1,l+2])] += kd_n
            
            fb_times.update(1)
            num_cout += 1
        # improper #
        num_cout = 0
        for i,j,k,l in self.improper_index*3: 
            ki_n = self.second_deriavete_element_four_body(self.improper_par[num_cout,0],np.deg2rad(self.improper_par[num_cout,1]),None,
                                                           self.mass_type[int(i/3)],self.mass_type[int(j/3)],self.mass_type[int(k/3)],self.mass_type[int(l/3)],
                                                           self.position[int(i/3)],self.position[int(j/3)],self.position[int(k/3)],self.position[int(l/3)],
                                                           mode='improper')
            
            hm[np.ix_([i,i+1,i+2,j,j+1,j+2,k,k+1,k+2,l,l+1,l+2],[i,i+1,i+2,j,j+1,j+2,k,k+1,k+2,l,l+1,l+2])] += ki_n

            fb_times.update(1)
            num_cout += 1
        fb_times.close()
        
        print("Build nonbonded pairwise potential")
        num_cout = 0
        vdw_times = tqdm(total=self.vdw_index.shape[0],ncols=100)
        for i,j in self.vdw_index*3:
            k_vn = self.second_deriavete_element_two_body(self.pair_par[num_cout,0],self.pair_par[num_cout,1],
                                                         self.mass_type[int(i/3)],self.mass_type[int(j/3)],
                                                         self.position[int(i/3)],self.position[int(j/3)],
                                                         mode='vdw')
            hm[np.ix_([i,i+1,i+2,j,j+1,j+2],[i,i+1,i+2,j,j+1,j+2])] += k_vn
            
            vdw_times.update(1)
            num_cout += 1
        vdw_times.close()

        print("Symmetric matrix: {}".format(self.check_symmetric(hm)))
        print("Determinant: {}".format(np.linalg.det(hm)>0))
        print("Hessian mattrix builded")        
        return np.round(hm,6)

    def solve_Hessian(self,h):
       print("Solve eignvalue and eignvector")
       h_val, h_vec = jax.numpy.linalg.eigh(h)
       print("Positive matrix: {}".format(len(np.where(h_val<-1)[0])==0))
       print("Done")
       return h_val, h_vec



