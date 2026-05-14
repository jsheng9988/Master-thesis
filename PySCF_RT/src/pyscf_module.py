import numpy
import rtutil
def get_fock(mol,mf,h1e,Density_mat):
    vj, vk = mf.get_jk(mol, Density_mat)
    myvhf = vj - vk * .5
#    myvhf = 0.0
    myF = h1e + myvhf
    return myF

#######################################################

def mo_fock_mid_forwd_eval(mol,mf,h1e,D_ti,fock_mid_ti_backwd,delta_t,dipole_z,C,C_inv,S,ndim,propthresh,pulse):

    fock_inter = numpy.zeros((ndim,ndim),dtype=numpy.complex128)
# D_ti is in AO basis 
# transform in the MO ref basis
    
    Dp_ti = numpy.matmul(C_inv,numpy.matmul(D_ti,numpy.conjugate(C_inv.T)))
    
    k = 1
    fockmtx = get_fock(mol,mf,h1e,D_ti)
    fock_ti_ao = fockmtx - (dipole_z * pulse)
    fock_guess = 2.00*fock_ti_ao - fock_mid_ti_backwd
    while True:
         fockp_guess = numpy.matmul(numpy.conjugate(C.T), \
                     numpy.matmul(fock_guess,C))
     
#         u = rtutil.exp_opmat(fockp_guess,delta_t,debug,odbg)
         u = rtutil.exp_opmat(fockp_guess,delta_t)
         tmpd = numpy.matmul(Dp_ti,numpy.conjugate(u.T))
         Dp_ti_dt = numpy.matmul(u,tmpd)
         #backtrasform Dp_ti_dt
         D_ti_dt = numpy.matmul(C,numpy.matmul(Dp_ti_dt,numpy.conjugate(C.T)))
         #build the correspondig Fock , fock_ti+dt
         fock_ti_dt_ao=get_fock(mol,mf,h1e,D_ti_dt)-(dipole_z*pulse)
         fock_inter = 0.5*fock_ti_ao + 0.5*fock_ti_dt_ao
         fock_guess = numpy.copy(fock_inter)
     
         if k > 1:
              # test on the norm: compare the density at current step and previous step
              # calc frobenius of the difference D_ti_dt_mo_new-D_ti_dt_mo
              diff = D_ti_dt-dens_test
              norm_f = numpy.linalg.norm(diff,'fro')
              if norm_f < (propthresh):
                 break
     
         dens_test = numpy.copy(D_ti_dt)
         k += 1

    return fock_inter

 




