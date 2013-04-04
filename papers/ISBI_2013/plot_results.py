from matplotlib import rc
import matplotlib.pyplot as plt
import numpy as np


rc('text', usetex=True)
rc('font', family='serif')

sk    = np.loadtxt('_recovered_angle_sk_0drop.npy')
sk5d  = np.loadtxt('_recovered_angle_sk_5drop.npy')
sk15d = np.loadtxt('_recovered_angle_sk_15drop.npy')
ag    = np.loadtxt('_recovered_angle_aganj.npy')
tc    = np.loadtxt('_recovered_angle_tuch.npy')

plt.xlabel(r'Exact crossing angle (degrees)',fontsize=16)
plt.ylabel(r'Average error in estimated crossing angle (degrees)',fontsize=16)
plt.ylim([-10,70])
plt.xlim([25,95])
plt.grid(True)

plt.errorbar(sk[:,0],sk[:,3],yerr=sk[:,3].std(),fmt='-x',label='SK')
#plt.errorbar(sk5d[:,0],sk5d[:,3],yerr=sk5d[:,3].std(),fmt='-*',label='SK 5 dropped')
plt.errorbar(sk15d[:,0],sk15d[:,3],yerr=sk15d[:,3].std(),fmt='-*',label='SK 15 dropped')
plt.errorbar(ag[:,0],ag[:,3],yerr=ag[:,3].std(),fmt='--o',label='AG-SH')
#plt.errorbar(tc[4:,0],tc[4:,3],yerr=tc[4:,3].std(),fmt='-.+',label='QB-SH')
plt.legend()


#plt.savefig('test')

plt.show()
