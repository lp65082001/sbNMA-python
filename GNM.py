from prody import *
import matplotlib.pyplot as plt
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['figure.dpi'] = 300

ubi = parsePDB('1UBQ')
calphas = ubi.select('protein and name CA')
gnm = GNM('Ubiquitin')
gnm.buildKirchhoff(calphas)
gnm.calcModes(n_modes=20)
corr = calcCrossCorr(gnm)

fig, ax = plt.subplots()
im = ax.imshow(corr,vmin=-1, vmax=1, cmap='cool')
fig.colorbar(im, ax=ax, label='Cross correlation')
ax.invert_yaxis()
plt.savefig('./result/gnm.png',bbox_inches='tight')