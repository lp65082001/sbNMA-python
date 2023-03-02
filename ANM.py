from prody import *
import matplotlib.pyplot as plt
import numpy as np
import parmed as pmd
from bin import v3d_plot
plt.rcParams['savefig.dpi'] = 250
plt.rcParams['figure.dpi'] = 250

def get_reduce_index(structure,name="CA"):
    red_index = []
    for i in structure.atoms:
        red_index.append(i.name)         
    red_index = np.array(red_index)
    return np.where(red_index==name)[0]

pro = parsePDB('1UBQ')
calphas = pro.select('protein and name CA')
anm = ANM('pro ANM analysis')
anm.buildHessian(calphas)
anm.calcModes(n_modes=20)
corr = calcCrossCorr(anm)


fig, ax = plt.subplots()
im = ax.imshow(corr,vmin=-1, vmax=1, cmap='cool')
fig.colorbar(im, ax=ax, label='Cross correlation')
ax.invert_yaxis()
plt.savefig('./result/anm.png',bbox_inches='tight')

np.save("./anm.npy",corr)