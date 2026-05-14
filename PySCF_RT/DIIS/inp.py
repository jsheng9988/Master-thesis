#!/usr/bin/env python
import sys
import numpy 
import sys
from pyscf import gto, scf
import scipy
from scipy.fft import fft, fftfreq, rfft
import math
import matplotlib.pyplot as plt 
from scipy.linalg import sqrtm, inv



mol = gto.Mole()
mol.verbose = 4
#mol.atom = '''
#O        0.000000    0.000000    0.117790
#H        0.000000    0.755453   -0.471161
#H        0.000000   -0.755453   -0.471161'''
mol.atom = '''
O
H 1 1.1
H 1 1.1 2 104
'''

mol.basis = '631g'
#mol.symmetry = 1
mol.build()

mf = scf.RHF(mol)
mf.conv_tol=1.0e-24
mf.kernel()

print(mf.e_tot)
print(mf.dip_moment())
print(mf.dip_moment(unit=''))

dm0=mf.get_init_guess(mol)
dm=dm0
h1e = mf.get_hcore(mol)
vj, vk = mf.get_jk(mol, dm)
myvhf = vj - vk * .5
myF = h1e + myvhf
print(numpy.trace(numpy.dot((myF+h1e),dm))*0.5+mf.mol.energy_nuc())
H=myF 
ovap = mf.get_ovlp()

# ovap^-1/2
A = sqrtm(inv(ovap)) 

w, vec = scipy.linalg.eigh(H,ovap)

#print(mf.mo_coeff)
mo_occ = mf.get_occ()
print(mo_occ)
nocc = int(numpy.sum(mo_occ))
nocc_orb = int(nocc/2)
print('shape',vec.shape)
mo_occ = vec[0:nocc_orb,:]
print(w[0:nocc_orb])

md = numpy.einsum("pi,qi->pq", mo_occ, mo_occ, optimize=True)
# Nuclear Repulsion Energy
E_nuc = mf.mol.energy_nuc()

# ==> Pre-Iteration Setup <==
# SCF & Previous Energy
SCF_E = 0.0
E_old = 0.0

# Start from fresh orbitals
F_p =  numpy.conjugate(A.T).dot(H).dot(A)
e, C_p = numpy.linalg.eigh(F_p)
print(e)
C = A.dot(C_p)
ndocc = nocc_orb
C_occ = C[:, :ndocc]
D = numpy.einsum('pi,qi->pq', C_occ, C_occ, optimize=True)

E_nuc = mf.mol.energy_nuc()

# ==> SCF Iterations w/ DIIS <==
print('==> Starting SCF Iterations <==\n')
MAXITER =100
# Begin Iterations
dm0=mf.get_init_guess(mol)
dm=dm0
S=ovap
# ovap^-1/2
A = sqrtm(inv(ovap))
# Trial & Residual Vector Lists
F_list = []
DIIS_RESID = []
E_conv = 1.0e-13
D_conv = 1.0e-10
#diis = False
diis = True 

# ==> SCF Iterations w/ DIIS <==
print('==> Starting SCF Iterations <==\n')
for scf_iter in range(1, MAXITER + 1):
    vj, vk = mf.get_jk(mol, dm)
    myvhf = vj - vk * .5
    myF = h1e + myvhf
    F = myF

    # Build DIIS Residual (A+ (FDS - SDF) A)
#    diis_r = numpy.conjugate(A.T).dot(F.dot(dm).dot(S) - S.dot(dm).dot(F)).dot(A)
    diis_r = F.dot(dm).dot(S) - S.dot(dm).dot(F)
    # Append trial & residual vectors to lists
    F_list.append(F)
    DIIS_RESID.append(diis_r)
    # Compute RHF energy
    SCF_E = numpy.einsum('pq,qp->', (myF+h1e)*0.5, dm, optimize=True) + E_nuc
    dE = SCF_E - E_old

    dRMS = numpy.mean(diis_r**2)**0.5

    print('SCF Iteration %3d: Energy = %4.16f dE = % 1.5E dRMS = %1.5E' % (scf_iter, SCF_E, dE, dRMS))
 
    # SCF Converged?
    if (abs(dE) < E_conv) and (dRMS < D_conv):
        break
    if (scf_iter >= 2) and (diis == True):
        # Build B matrix
        B_dim = len(F_list) + 1
        B = numpy.empty((B_dim, B_dim))
        B[-1, :] = -1
        B[:, -1] = -1
        B[-1, -1] = 0
        for i in range(len(F_list)):
            for j in range(len(F_list)):
                B[i, j] = numpy.einsum('ij,ij->', DIIS_RESID[i], DIIS_RESID[j], optimize=True)


        # Build RHS of Pulay equation 
        rhs = numpy.zeros((B_dim))
        rhs[-1] = -1
        
        # Solve Pulay equation for c_i's with NumPy
        coeff = numpy.linalg.solve(B, rhs)
#        print(coeff)
        
        # Build DIIS Fock matrix
        F_diis = numpy.zeros_like(F)
        for x in range(coeff.shape[0] - 1):
            F_diis += coeff[x] * F_list[x]
            F = F_diis

    w, vec = scipy.linalg.eigh(F,ovap)
    mo_occ = vec[:,:nocc_orb]
    dm = numpy.einsum("pi,qi->pq", mo_occ, mo_occ, optimize=True)*2.0
    dE = SCF_E - E_old
    E_old = SCF_E


exit()


ao_dip = -mol.intor_symmetric('int1e_r', comp=3)
el_dip = numpy.einsum('aij,ji->a', ao_dip, dm).real
charges = mol.atom_charges()
coords  = mol.atom_coords()
nucl_dip = numpy.einsum('i,ix->x', charges, coords)

mol_dip = nucl_dip + el_dip

ke_data=mol.intor("int1e_kin")
en_data=-mol.intor("int1e_nuc")
ee_twoe_data=mol.intor("int2e")

print('')
print('')
print(ke_data)
print(en_data)
print(ovap)
print(ee_twoe_data)

