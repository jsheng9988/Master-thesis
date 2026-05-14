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

def readnpz(filename):
    data =numpy.load(filename)
    lst = data.files
    print('')
    print('')
    for item in lst:
        print(item)
        print(data[item])
    print('')
    print('')
    return

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
mol.atom = '''
O        0.000000    0.000000    0.117790
H        0.000000    0.755453   -0.471161
H        0.000000   -0.755453   -0.471161'''
#mol.atom = '''
#H        0.000000    0.000000   0.37 
#H        0.000000    0.0        -0.37 '''

#mol.basis = '631g'
mol.basis = 'sto3g'
#mol.symmetry = 1
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

ao_dip = -mol.intor_symmetric('int1e_r', comp=3)
el_dip = numpy.einsum('aij,ji->a', ao_dip, dm).real
charges = mol.atom_charges()
coords  = mol.atom_coords()
nucl_dip = numpy.einsum('i,ix->x', charges, coords)

mol_dip = nucl_dip + el_dip
print(mol_dip)

funcswitcher = {
    "kick": rtutil.kick,
    "gauss_env": rtutil.gauss_env,
    "envelope":  rtutil.envelope,
    "sin_oc":    rtutil.sin_oc,
    "cos_env":   rtutil.cos_env,
    "lin_cos":   rtutil.lin_cos,
    "analytic":  rtutil.analytic
     }


# Choose the impulse
#impulsefunc="analytic"
#impulsefunc="cos_env"
#impulsefunc="lin_cos"
impulsefunc="analytic"
# Choose E0 of the field
fmax=0.05
# Choose dierction ('x','y','z')
direction = 'z'
# Choose w of the field in a.u. 
# Case H2 sto3g
w0=0.6864267
#w0=0.004702
t0=0.0
sigma=200000.0
# Choose dt 
dt=0.08268
# Choose total_time_steps
total_time_steps = 10000
propthresh=1.0e-6
debug=False
odbg=sys.stderr


intensityau, intensity = fieldinfo(fmax)

print('')
print('Details of the simulation:')
txt1 = "Inpulse: {}, Fmax (a.u.): {}, Direction:{} ".format(impulsefunc, fmax, direction)
txt2 = "dt: {} a.u., Total num. steps: {} ".format(dt, total_time_steps)
txt3 = "E0 in a.u.  {:1e}".format(fmax)
txt4 = "Energy/(area*time) in a.u.  {:1e}".format(intensityau)
txt5 = "Energy/(area*time) in W/cm^2 {:1e}".format(intensity)
print(txt1)
print(txt2)
print(txt3)
print(txt4)
print(txt5)

print('')

dipx_mat, dipy_mat, dipz_mat = -mol.intor_symmetric('int1e_r', comp=3)
if direction == 'x':
   dip_mat=dipx_mat

if direction == 'y':
   dip_mat=dipy_mat

if direction == 'z':
   dip_mat=dipz_mat

#print(numpy.trace(numpy.matmul(dip_mat,dm)))
fockm_at = pyscf_module.get_fock(mol,mf,h1e,dm)
w, eigem = scipy.linalg.eigh(fockm,ovap)
ndim=len(w)
C = eigem
C_inv = numpy.linalg.inv(C)
dipz_mo=numpy.matmul(numpy.conjugate(C.T),numpy.matmul(dip_mat,C))

func = funcswitcher.get(impulsefunc, lambda: rtutil.kick)

print("Start first mo_fock_mid_forwd_eval ")
#t=0
D_0 = numpy.zeros((ndim,ndim), dtype=numpy.complex128)
print(nocc_orb)
for num in range(nocc_orb):
        D_0[num,num] = 2.+0.0j

Dp_ti= D_0 
#backtrasform Dp_t1
D_ti=numpy.matmul(C,numpy.matmul(Dp_ti,numpy.conjugate(C.T)))

print(numpy.trace(numpy.matmul(dip_mat,D_ti)))
print(numpy.trace(numpy.matmul(dip_mat,dm)))

fockm_mid_at=fockm_at

if func.__name__ == 'analytic':
    dipz_mo=numpy.matmul(numpy.conjugate(C.T),numpy.matmul(dip_mat,C))
    print(" Perturb with analytic kick ")
    u0=rtutil.exp_opmat(dipz_mo,numpy.float_(-fmax),debug)
    Dp_init=numpy.matmul(u0,numpy.matmul(Dp_ti,numpy.conjugate(u0.T)))
    #transform back Dp_int
    D_ti=numpy.matmul(C,numpy.matmul(Dp_init,numpy.conjugate(C.T)))
    print('dip with pert.', numpy.trace(numpy.matmul(dip_mat,D_ti)))

dip_list=[]
t_list=[]
pulse_list=[]
energy_list=[]

data_D_re  =numpy.empty((total_time_steps,D_ti.shape[0],D_ti.shape[0]),dtype=float)
data_D_imag=numpy.empty((total_time_steps,D_ti.shape[0],D_ti.shape[0]),dtype=float)
data_Fat_re  =numpy.empty((total_time_steps,D_ti.shape[0],D_ti.shape[0]),dtype=float)
data_Fat_imag=numpy.empty((total_time_steps,D_ti.shape[0],D_ti.shape[0]),dtype=float)

#myF = h1e + myvhf
#print(np.trace(np.dot((myF+h1e),dm))*0.5+mf.mol.energy_nuc())
for j in range(total_time_steps):
    t_arg = float(j*dt)
    pulse = func(fmax, w0, t_arg, t0, sigma)


    
    dip_list.append(numpy.trace(numpy.matmul(dip_mat,D_ti)))
    t_list.append(t_arg)

    myfock = pyscf_module.get_fock(mol,mf,h1e,D_ti)


    data_D_re[j,:,:]  =D_ti.real
    data_D_imag[j,:,:]=D_ti.imag
    data_Fat_re[j,:,:]  =myfock.real
    data_Fat_imag[j,:,:]=myfock.imag


    energy_list.append(numpy.trace(numpy.dot((myfock+h1e),D_ti))*0.5+mf.mol.energy_nuc())
    pulse_list.append(pulse)
    fock_mid_at = pyscf_module.mo_fock_mid_forwd_eval(mol,mf,h1e,D_ti,numpy.copy(fockm_mid_at),float(dt),\
            dip_mat,C,C_inv,ovap,ndim,propthresh,pulse)

    fockp_mid_tmp = numpy.matmul(numpy.conjugate(C.T),numpy.matmul(fock_mid_at, C))
    u = rtutil.exp_opmat(numpy.copy(fockp_mid_tmp),numpy.float_(dt),debug)
    Dp_ti = numpy.matmul(C_inv,numpy.matmul(D_ti,numpy.conjugate(C_inv.T)))
    temp=numpy.matmul(Dp_ti,numpy.conjugate(u.T))
    Dp_ti_dt = numpy.matmul(u,temp)
    D_ti_dt = numpy.matmul(C,numpy.matmul(Dp_ti_dt,numpy.conjugate(C.T))) 
    D_ti=D_ti_dt

    rtutil.progress_bar(j, total_time_steps-1)

ke_data=mol.intor("int1e_kin")
en_data=-mol.intor("int1e_nuc")
ee_twoe_data=mol.intor("int2e")

print('')
print('')
print(ke_data)
print(en_data)
print(ovap)
print(ee_twoe_data)

print()
print("ao_dip:")
print(ao_dip)
print(ao_dip[0])
print(ao_dip[1])
print(ao_dip[2])

# save  ke+en+overlap+ee_twoe+dip_hf_delta....npz file
numpy.savez('ke+en+overlap+ee_twoe+dip_hf_delta_s0_h2_sto-3g_PYSCF.npz', ke_data=ke_data, en_data=en_data, overlap_data=ovap, ee_twoe_data=ee_twoe_data, dipx_data=ao_dip[0],dipy_data=ao_dip[1], dipz_data=ao_dip[2])

readnpz('ke+en+overlap+ee_twoe+dip_hf_delta_s0_h2_sto-3g_PYSCF.npz')

# save td_dens_re+im_rt-tdexx_delta_s0_ .....npz file
numpy.savez('td_dens_re+im_rt-tdexx_delta_s0_h2_sto-3g_PYSCF.npz', td_dens_re_data=data_D_re[:,:,:], td_dens_im_data=data_D_imag[:,:,:])

# save td_dens_re+im_rt-tdexx_delta_s0_ .....npz file
numpy.savez('td_Fock_at_re+im_rt-tdexx_delta_s0_h2_sto-3g_PYSCF.npz', td_fock_at_re_data=data_Fat_re[:,:,:], td_fock_at__im_data=data_Fat_imag[:,:,:])






#print(numpy.array(dip_list).real)
dip_data=numpy.array(dip_list).real
t_data=numpy.array(t_list).real
energy_data=numpy.array(energy_list).real
pulse_data=numpy.array(pulse_list).real

#plt.plot(energy_list)
data_dip=numpy.stack((t_data,dip_data.real-dip_data[0].real)).T
data_pulse=numpy.stack((t_data,pulse_data.real)).T
data_energy=numpy.stack((t_data,energy_data.real)).T
#fig, (ax1,ax2,ax3,ax4,ax5) = plt.subplots(5)
fig, (ax2,ax4) = plt.subplots(2)
#ax1.plot(t_data,pulse_data.real, label='Field')
ax2.plot(t_data,dip_data.real, label='dipole')
#ax1.legend()
ax2.legend()
fig.suptitle('A single plot')



col=data_dip.T
damp=numpy.exp(-0.0001*col[0])
# Number of sample points
#N = 10000
N=len(col[0])
# sample spacing
#T = 1/10
T=col[0][1]-col[0][0]

x = numpy.linspace(0.0, N*T, N, endpoint=False)
#print(x)
#damp=numpy.exp(-0.00000001*x**2)
#damp=1
#y = (np.sin(0.5 * 2.0*np.pi*x) + 0.5*np.sin(85.0 * 2.0*np.pi*x)+0.1*np.sin(51.0 * 2.0*np.pi*x)+0.1*np.sin(1.0 * 2.0*np.pi*x)+0.1*np.sin(1.1 * 2.0*np.pi*x))*damp
y = (col[1]-col[1][0])*damp
#ax3.plot(t_data,y , label='dipole damp.')
#ax3.legend()

data_damped_dipole=numpy.stack((t_data,y)).T
#yf = fft(y)

print(dir(scipy.fft))
yf = scipy.fft.rfft(y)
xf = scipy.fft.fftfreq(N, T)[:N//2]
print(xf)

import matplotlib.pyplot as plt
#plt.plot(xf, 2.0/N * np.abs(yf[0:N//2]))
#plt.plot(xf, 2.0/N * yf[0:N//2].real)
#plt.plot(xf, 2.0/N * yf[0:N//2].imag)
plt.plot(xf*27.211*2.0*numpy.pi, 2.0/N * numpy.abs(yf[0:N//2])*xf)
plt.plot(xf*27.211*2.0*numpy.pi, 2.0/N * numpy.abs(yf[0:N//2])*xf)
ax4.plot(xf*27.211*2.0*numpy.pi,2.0/N * numpy.abs(yf[0:N//2])*xf, label='Spectra')
ax4.legend()
#ax5.plot(xf*27.211*2.0*numpy.pi,2.0/N * yf[0:N//2].imag*xf, label='Im')
#ax5.legend()

data_omega = xf*27.211*2.0*numpy.pi
data_spectra = 2.0/N * numpy.abs(yf[0:N//2])*xf
data_spectra_real = 2.0/N * numpy.real(yf[0:N//2])*xf
data_spectra_imag= 2.0/N * numpy.imag(yf[0:N//2])*xf

data_plot_spectra=numpy.stack((data_omega,data_spectra)).T
data_plot_spectra_real=numpy.stack((data_omega,data_spectra_real)).T
data_plot_spectra_imag=numpy.stack((data_omega,data_spectra_imag)).T



#dap=numpy.exp(-0.0002*t_data**2)
#signal = numpy.array(dip_data.real, dtype=float)*dap
#fourier = numpy.fft.fft(signal)
#n = signal.size
#timestep=dt
#freq = numpy.fft.fftfreq(n, d=timestep)
#plt.plot(freq,fourier.real)
#plt.plot(freq,fourier.imag)
numpy.savetxt('dipole.txt', data_dip, fmt='%.15e')
numpy.savetxt('field.txt',  data_pulse, fmt='%.15e')
numpy.savetxt('energy.txt',  data_energy, fmt='%.15e')
numpy.savetxt('spectra.txt', data_plot_spectra, fmt='%.15e')
numpy.savetxt('spectra_real.txt', data_plot_spectra_real, fmt='%.15e')
numpy.savetxt('spectra_imag.txt', data_plot_spectra_imag, fmt='%.15e')
numpy.savetxt('dipole_damp.txt', data_damped_dipole, fmt='%.15e')

plt.show()

