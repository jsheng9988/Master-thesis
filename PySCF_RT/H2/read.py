from numpy import load

data = load('ke+en+overlap+ee_twoe+dip_hf_delta_s0_h2_sto-3g_PYSCF.npz')
#data = load('td_dens_re+im_rt-tdexx_delta_s0_h2_sto-3g.npz')
lst = data.files
for item in lst:
    print(item)
    print(data[item])
