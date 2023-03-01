from prody import *
import matplotlib.pyplot as plt

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