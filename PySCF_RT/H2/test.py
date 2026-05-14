#!/usr/bin/env python
import sys
sys.path += ["../src"]
print('python3 is looking into:',sys.path)
import rtutil
import pyscf_module

import numpy 
import sys
from pyscf import gto, scf
import scipy
from scipy.fft import fft, fftfreq, rfft
import math
import rtutil
import pyscf_module
import matplotlib.pyplot as plt 

#=====================================
def fieldinfo(e0):
    """
    fieldinfo(e0):
    e0 : input  value of Fmax
    intensityau: (output) energy flux in a.u.
    intensity: (output) energy flux in W/cm^2 
    """
    c=137.0359
    intensityau = c*1/(8.*numpy.pi)*e0**2
    # intensity on W/cm^2
    conv= 6.436*10**15
    intensity=conv*intensityau

    return intensityau, intensity
#=====================================



mol = gto.Mole()
mol.verbose = 4
#mol.atom = '''
#O        0.000000    0.000000    0.117790
#H        0.000000    0.755453   -0.471161
#H        0.000000   -0.755453   -0.471161'''
mol.atom = '''
H        0.000000    0.000000   0.37 
H        0.000000    0.0        -0.37 '''

#mol.basis = '631g'
mol.basis = 'sto3g'
#mol.symmetry =  
mol.build()




mf = scf.RHF(mol)
mf.conv_tol=1.0e-24
mf.kernel()
print(mf.e_tot)
print(mf.dip_moment())
print(mf.dip_moment(unit=''))


#print(mf.mo_coeff)
mo_occ = mf.get_occ()
print(mo_occ)
nocc = int(numpy.sum(mo_occ))
nocc_orb = int(nocc/2)

h1e = mf.get_hcore(mol)
dm = mf.make_rdm1()

#mydm = numpy.dot(mf.mo_coeff[:,mo_occ>0]*mo_occ[mo_occ>0], mf.mo_coeff[:,mo_occ>0].T)

#vhf = mf.get_veff(mol, dm)
vj, vk = mf.get_jk(mol, dm)
myvhf = vj - vk * .5
myF = h1e + myvhf
print(numpy.trace(numpy.dot((myF+h1e),dm))*0.5+mf.mol.energy_nuc())

fockm = myF
ovap = mf.get_ovlp()
w, vec = scipy.linalg.eigh(fockm,ovap)
print(w[0:nocc_orb])
#print(dir(mf))
print(numpy.shape(ovap))
print(ovap)
print(mf._eri)
print(dir(mf))

ao_dip = -mol.intor_symmetric('int1e_r', comp=3)
el_dip = numpy.einsum('aij,ji->a', ao_dip, dm).real
charges = mol.atom_charges()
coords  = mol.atom_coords()
nucl_dip = numpy.einsum('i,ix->x', charges, coords)

mol_dip = nucl_dip + el_dip
print(mol_dip)


ke_data=mol.intor("int1e_kin")
en_data=-mol.intor("int1e_nuc")
ee_twoe_data=mol.intor("int2e")

print('')
print('')
print(ke_data)
print(en_data)
print(ovap)
print(ee_twoe_data)
print(ao_dip)



