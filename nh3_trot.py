#%% ---------------- import --------------------
import numpy as np
from astropy import units as u
import numpy as np
from matplotlib import pyplot as plt
# import aplpy as apl
from astropy import wcs
from astropy.io import fits
from spectral_cube import SpectralCube as sc
from matplotlib import colors
from mpl_toolkits.mplot3d import Axes3D
from pycupid import clumpfind, fellwalker, reinhold, gaussclumps
import numpy.ma as ma
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
import matplotlib.patches as mpatches
from scipy import interpolate
from reproject import reproject_interp
from spectral_cube import SpectralCube as sc
from astropy.table import Column, Table
from lmfit import minimize, Parameters, Model
from astropy import wcs
from astropy import constants as cons
from astropy import units as u
# plt.interactive(True)
from matplotlib import rc

#%% ----------------- phy constants ---------------------------
m0 = 1.67e-24*2.33; hp = 6.626e-27; kb = 1.38e-16; Gr = 6.67e-8;
pc = 3.086e18; au = 1.496e13;
c = 2.998e10; year = 86400.0*365;
Msun = 1.99e33; Lsun = 3.826e33; Jy = 1.0e-23; Rsun = 6.96e10; Tbg = 2.73; eV = 1.602e-12;
dist = 7000 * pc;
das = 7000 * au # 1 arcsec

#%% --------- artificial spectra ------------
def T11_theory(vs, dv, tau_0=0., v0=0., Tex=0., osg1=0., isg1=0., mg=0., isg2=0., osg2=0.):
    v_0, v_1 = -19.84514, -19.31960
    v_2, v_3, v_4 = -7.88632, -7.46920, -7.35005
    v_5, v_6, v_7, v_8, v_9, v_10, v_11, v_12 = -0.46227, -0.32312, -0.30864, -0.18950, 0.07399, 0.13304, 0.21316, 0.25219
    v_13, v_14, v_15 = 7.23455, 7.37370, 7.81539
    v_16, v_17 = 19.40943, 19.54859
    ag = np.zeros(18)
    ag[0:2] = [1.0 / 27, 2.0 / 27]
    ag[2:5] = [5.0 / 108, 1.0 / 12, 1.0 / 108]
    ag[5:13] = [1.0 / 54, 1.0 / 108, 1.0 / 60, 3.0 / 20, 1.0 / 108, 7.0 / 30, 5.0 / 108, 1.0 / 60]
    ag[13:16] = [5.0 / 108, 1.0 / 108, 1.0 / 12]
    ag[16:18] = [1.0 / 27, 2.0 / 27]
    profile1 = ag[0] * np.exp(-4 * np.log(2) * ((vs - v0 - v_0) / dv) ** 2) + \
               ag[1] * np.exp(-4 * np.log(2) * ((vs - v0 - v_1) / dv) ** 2)  # osg_1
    profile2 = ag[2] * np.exp(-4 * np.log(2) * ((vs - v0 - v_2) / dv) ** 2) + \
               ag[3] * np.exp(-4 * np.log(2) * ((vs - v0 - v_3) / dv) ** 2) + \
               ag[4] * np.exp(-4 * np.log(2) * ((vs - v0 - v_4) / dv) ** 2)  # isg_1
    profile3 = ag[5] * np.exp(-4 * np.log(2) * ((vs - v0 - v_5) / dv) ** 2) + \
               ag[6] * np.exp(-4 * np.log(2) * ((vs - v0 - v_6) / dv) ** 2) + \
               ag[7] * np.exp(-4 * np.log(2) * ((vs - v0 - v_7) / dv) ** 2) + \
               ag[8] * np.exp(-4 * np.log(2) * ((vs - v0 - v_8) / dv) ** 2) + \
               ag[9] * np.exp(-4 * np.log(2) * ((vs - v0 - v_9) / dv) ** 2) + \
               ag[10] * np.exp(-4 * np.log(2) * ((vs - v0 - v_10) / dv) ** 2) + \
               ag[11] * np.exp(-4 * np.log(2) * ((vs - v0 - v_11) / dv) ** 2) + \
               ag[12] * np.exp(-4 * np.log(2) * ((vs - v0 - v_12) / dv) ** 2)  # mg
    profile4 = ag[13] * np.exp(-4 * np.log(2) * ((vs - v0 - v_13) / dv) ** 2) + \
               ag[14] * np.exp(-4 * np.log(2) * ((vs - v0 - v_14) / dv) ** 2) + \
               ag[15] * np.exp(-4 * np.log(2) * ((vs - v0 - v_15) / dv) ** 2)  # isg_2
    profile5 = ag[16] * np.exp(-4 * np.log(2) * ((vs - v0 - v_16) / dv) ** 2) + \
               ag[17] * np.exp(-4 * np.log(2) * ((vs - v0 - v_17) / dv) ** 2)  # osg_2
    tau_v = tau_0 * (osg1 * profile1 + isg1 * profile2 + mg * profile3 + isg2 * profile4 + osg2 * profile5)
    T11_theory = (1.138 / (-1 + np.exp(1.138 / Tex)) - 2.201) * (1 - np.exp(-tau_v))
    return T11_theory

def T22_theory(vs, dv, tau_0=0.0, v0=0., Tex=0., osg1=0., isg1=0., mg=0., isg2=0., osg2=0.):
    v_0, v_1, v_2 = -26.52625, -26.01112, -25.95045
    v_3, v_4, v_5 = -16.39171, -16.37929, -15.86417
    v_6, v_7, v_8, v_9, v_10, v_11, v_12, v_13, v_14, v_15, v_16, v_17 \
        = -0.56250, -0.52841, -0.52374, -0.01328, -0.01328, 0.00390, \
          0.00390, 0.01332, 0.01332, 0.50183, 0.53134, 0.58908
    v_18, v_19, v_20 = 15.85468, 16.36980, 16.38222
    v_21, v_22, v_23 = 25.95045, 26.01112, 26.52625
    ag = np.zeros(24)
    ag[0:3] = [1. / 300, 3. / 100, 1. / 60]
    ag[3:6] = [4. / 135, 14. / 675, 1. / 675]
    ag[6:18] = [1. / 60, 1. / 108, 8. / 945, 7. / 54, 1. / 12, 8. / 35, 32. / 189, 1. / 12, 1. / 30, 1. / 108, \
                8. / 945, 1. / 60]
    ag[18:21] = [1. / 675, 14. / 675, 4. / 135]
    ag[21:24] = [1. / 60, 3. / 100, 1. / 300]
    profile1 = ag[0] * np.exp(-4 * np.log(2) * ((vs - v0 - v_0) / dv) ** 2) + \
               ag[1] * np.exp(-4 * np.log(2) * ((vs - v0 - v_1) / dv) ** 2) + \
               ag[2] * np.exp(-4 * np.log(2) * ((vs - v0 - v_2) / dv) ** 2)  # osg_1
    profile2 = ag[3] * np.exp(-4 * np.log(2) * ((vs - v0 - v_3) / dv) ** 2) + \
               ag[4] * np.exp(-4 * np.log(2) * ((vs - v0 - v_4) / dv) ** 2) + \
               ag[5] * np.exp(-4 * np.log(2) * ((vs - v0 - v_5) / dv) ** 2)  # isg_1
    profile3 = ag[6] * np.exp(-4 * np.log(2) * ((vs - v0 - v_6) / dv) ** 2) + \
               ag[7] * np.exp(-4 * np.log(2) * ((vs - v0 - v_7) / dv) ** 2) + \
               ag[8] * np.exp(-4 * np.log(2) * ((vs - v0 - v_8) / dv) ** 2) + \
               ag[9] * np.exp(-4 * np.log(2) * ((vs - v0 - v_9) / dv) ** 2) + \
               ag[10] * np.exp(-4 * np.log(2) * ((vs - v0 - v_10) / dv) ** 2) + \
               ag[11] * np.exp(-4 * np.log(2) * ((vs - v0 - v_11) / dv) ** 2) + \
               ag[12] * np.exp(-4 * np.log(2) * ((vs - v0 - v_12) / dv) ** 2) + \
               ag[13] * np.exp(-4 * np.log(2) * ((vs - v0 - v_13) / dv) ** 2) + \
               ag[14] * np.exp(-4 * np.log(2) * ((vs - v0 - v_14) / dv) ** 2) + \
               ag[15] * np.exp(-4 * np.log(2) * ((vs - v0 - v_15) / dv) ** 2) + \
               ag[16] * np.exp(-4 * np.log(2) * ((vs - v0 - v_16) / dv) ** 2) + \
               ag[17] * np.exp(-4 * np.log(2) * ((vs - v0 - v_17) / dv) ** 2)  # mg
    profile4 = ag[18] * np.exp(-4 * np.log(2) * ((vs - v0 - v_18) / dv) ** 2) + \
               ag[19] * np.exp(-4 * np.log(2) * ((vs - v0 - v_19) / dv) ** 2) + \
               ag[20] * np.exp(-4 * np.log(2) * ((vs - v0 - v_20) / dv) ** 2)  # isg_2
    profile5 = ag[21] * np.exp(-4 * np.log(2) * ((vs - v0 - v_21) / dv) ** 2) + \
               ag[22] * np.exp(-4 * np.log(2) * ((vs - v0 - v_22) / dv) ** 2) + \
               ag[23] * np.exp(-4 * np.log(2) * ((vs - v0 - v_23) / dv) ** 2)  # osg_2
    tau_v = tau_0 * (osg1 * profile1 + isg1 * profile2 + mg * profile3 + isg2 * profile4 + osg2 * profile5)
    T22_theory = (1.138 / (-1 + np.exp(1.138 / Tex)) - 2.2) * (1 - np.exp(-tau_v))
    return T22_theory


'''

'''
#%% ------------- tau1/tau2 vs Trot -----------------
A11, A22 = 10**6.76650, 10**6.64
f11, f22 =  23.69449550, 23.72263330
g11, g22 = 3, 5
print((g11*f22*A11/0.7963) / (g22*f11*A22/0.6467))

#%% ----------------- image import -----------------------
from matplotlib.collections import PatchCollection
import matplotlib.patches as mpatches
from spectral_cube import SpectralCube as sc
import numpy as np
nh31_cube = sc.read('/Users/rzy/data/omc_image/nh3_11.fits')
nh32_cube = sc.read('/Users/rzy/data/omc_image/nh3_22.fits')
w11 = wcs.WCS(nh31_cube.header, naxis=2)
w22 = wcs.WCS(nh32_cube.header, naxis=2)

nh31_cube = np.nan_to_num(np.asarray(nh31_cube))
nh32_cube = np.nan_to_num(np.asarray(nh32_cube))

#%% ---------------- plot area ------------------
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import rc
rc("font", family="serif", size=5)

nh31_int = np.mean(nh31_cube, axis=0)
spec1 = np.mean(nh31_cube[:, 450:550, 150:250], axis=(1,2))
spec2 = np.mean(nh32_cube[:, 450:550, 150:250], axis=(1,2))

fig=plt.figure(figsize=(9.5, 5))

ax0 = fig.add_subplot(1,3,1)
ax0.imshow(nh31_int, origin='lower', interpolation='nearest', cmap='PuRd')

ax0 = fig.add_subplot(1, 3, 2)
ax0.plot(spec1, drawstyle='steps')

ax0 = fig.add_subplot(1, 3, 3)
ax0.plot(spec2, drawstyle='steps')

#%% ------------- HFC images -----------------------
S11_mg = np.sum(nh31_cube[24:37, :, :], axis=0)
S11_isg1 = np.sum(nh31_cube[15:25, :, :], axis=0)
S11_isg2 = np.sum(nh31_cube[38:48, :, :], axis=0)
S22_mg = np.sum(nh32_cube[25:40, :, :], axis=0)
S11_isg = S11_isg1 + S11_isg2

S11_mg_hdu = fits.ImageHDU(data=S11_mg, header=w11.to_header())
S11_isg_hdu = fits.ImageHDU(data=S11_isg2, header=w11.to_header())
S22_mg_hdu = fits.ImageHDU(data=S22_mg, header=w22.to_header())

dir1 = '/Users/rzy/data/omc_image/im4sed/'
hdu_sp250b = fits.open(dir1+'omc_sp250_crop.fits')[0]
A250 = np.pi/4*(18**2)

#%% ---------------- template 250 ------------------
im_sp250 = hdu_sp250b.data
im_sp250 = np.nan_to_num(im_sp250)/A250*0.55
w1 = wcs.WCS(hdu_sp250b.header, naxis=2)
ny, nx = np.shape(im_sp250)  # not hdu2.hdu.data, already an HDU object !!!

w1b = wcs.WCS(naxis=2)
w1b.wcs.ctype = ['RA---TAN',  'DEC--TAN']
w1b.wcs.crval = w1.wcs.crval
w1b.wcs.crpix = w1.wcs.crpix
w1b.wcs.cdelt = [-0.00166666666667, 0.00166666666667]

#%% --------------- nh3 images reprojection ---------------
_data, _fp = reproject_interp(S11_mg_hdu, output_projection=w1b, shape_out=(ny, nx))
S11_mg_rg = np.nan_to_num(_data)
_data, _fp = reproject_interp(S11_isg_hdu, output_projection=w1b, shape_out=(ny, nx))
S11_isg_rg = np.nan_to_num(_data)
_data, _fp = reproject_interp(S11_isg_hdu, output_projection=w1b, shape_out=(ny, nx))
S22_mg_rg = np.nan_to_num(_data)

#%% --------------- nh3 images plot --------------
fig = plt.figure(figsize=(10, 4))
ax = fig.add_subplot(1, 3, 1)
am1 = ax.imshow(S11_mg_rg, origin='lower', cmap='RdPu', interpolation='nearest', vmin=0, vmax=S11_mg.max())
ax.contour(S11_mg_rg, levels=[0.15], colors='black', linewidths=0.5)
plt.colorbar(am1)
ax = fig.add_subplot(1, 3, 2)
am2 = ax.imshow(S11_isg_rg, origin='lower', cmap='RdPu', interpolation='nearest')
fig.colorbar(am2)
ax.contour(S11_isg_rg,levels=[0.15], colors='black', linewidths=0.5)

ax = fig.add_subplot(1, 3, 3)
am3 = ax.imshow(S22_mg_rg,origin='lower', cmap='RdPu', \
                vmin=0, vmax=0.3, interpolation='nearest')
fig.colorbar(am3)
ax.contour(S22_mg_rg, levels=[0.15], colors='black', linewidths=0.5)

#%% ------------ run sed_model.py --------------

#%% ------------ NH3 Trot calculation ------------
thd = 0.149
x_idx, y_idx = np.where(S11_mg_rg > thd)
Tex_map = np.zeros(np.shape(S11_mg_rg))
print('lenth=', len(x_idx))
rms = 0.03

for i in np.arange(len(x_idx)):
    s1_mg = S11_mg_rg[x_idx[i], y_idx[i]]
    s1_isg = S11_isg_rg[x_idx[i], y_idx[i]]
    s2_mg = S22_mg_rg[x_idx[i], y_idx[i]]
    if s1_isg/s1_mg > 1.0:
        s1_isg = s1_mg * 0.9
    if s2_mg < rms:
        s2_mg = rms
    if s2_mg/(s1_mg+s1_isg) > 0.8:
        s2_mg = (s1_mg+s1_isg) * 0.8
    Tex_map[x_idx[i],y_idx[i]] = Tex_main(s1_mg, s1_isg, s2_mg)

#%% ------------- NH3 Trot map ------------
thd = 0.149
Tex_mask = np.ma.masked_where(S11_mg_rg < thd, Tex_map)

fig = plt.figure(figsize=(10, 5))
ax = fig.add_subplot(1, 2, 1)
am1 = ax.imshow(Tex_mask, origin='lower', cmap='jet', interpolation='nearest', \
                vmin=Tex_mask.min(), vmax=Tex_mask.max())
lvs = S11_mg_rg.max()*np.arange(0.1, 1.1, 0.2)
ax.contour(S11_mg_rg, levels=lvs, colors='black', linewidths=0.5)
plt.colorbar(am1)

# ------------ *** optimized parameters *** -------------------
'''
h1f = np.array([4.8664, -4.9366, 0.0680, 1.2724, -0.0456])
h2f = np.array([-9.3320, -3.5797, 2.1964, 5.7513, -0.3155])
Rsm0, cf0 = 0.5868718030896236, 0.9395788291127453   # (1:mg+isg+osg, 2: mg+isg)
'''

#%% ------------------cf(Rsm;  ap0,ap1) <--- ap0,ap1(Tex,dv)--------------------
# h1f = [2.31388311, -3.83775533,  0.38901939,  1.30297148, -0.10681572]
# h2f = [1.13007165, -4.04279513,  -0.8816936,  3.14942726,  0.30402135]
# Rsm0 = 0.55557; cf0 = 1.10207;   # (mg+isg)

# h1f = np.array([6.7280, -13.0197, 0.1066, 7.4822, -0.0426])
# h2f = np.array([-16.5279, 30.9374, 4.6468, -24.3743, -1.0987])
# Rsm0, cf0 = 0.5623103860366939, 1.320659500607111  # (1:mg+isg, 2: mg+isg)

h1f = np.array([1.0169, 0.8828, 0.8165, -5.7040, -0.2414])
h2f = np.array([3.8136, -24.2071, -1.2509, 31.0714, 0.4062])
Rsm0, cf0 = 0.5769996652074106, 1.0960194572608966

def ap1_fit(t_v, *h1):
    Tex, dv = t_v
    return h1[0]+h1[1]*(Tex/60.0)+h1[2]*(dv/1.0)+h1[3]*(Tex/60.0)**2+h1[4]*(dv/1.0)**2
    # +h13*np.exp(Tex/60.0)
def ap2_fit(t_v, *h2):
    Tex,dv = t_v
    return h2[0]+h2[1]*(Tex/60.0)+h2[2]*(dv/1.0)+h2[3]*(Tex/60.0)**2+h2[4]*(dv/1.0)**2

def cf_fit(Rsm1, ap1=1.0, ap2=0.3):
    return ap1*(Rsm1-Rsm0)+ap2*(Rsm1-Rsm0)**2+cf0

def Tex_main(S11_mg, S11_isg, S11_osg, S22_mg, S22_isg, dv=0.8):
    Rsm1 = np.abs(S11_isg / S11_mg)
    # R12 = np.abs((S11_mg + S11_isg) / (S22_mg + S22_isg))
    R12 = np.abs((S11_mg+S11_isg+S11_osg) / (S22_mg+S22_isg))
    Tex_i = 41.5 / np.log(1.0 * R12)
    n_iter = 3
    for i in np.arange(n_iter):
        ap1_i = ap1_fit((Tex_i, dv), *h1f)
        ap2_i = ap1_fit((Tex_i, dv), *h1f)
        cf_i = cf_fit(Rsm1, ap1=ap1_i, ap2=ap2_i)
        Tex_i = 41.5 / np.log(cf_i * R12)
        # Tex_i = 40.99 / np.log(1.0 * R12)
    return Tex_i

#%% ------------- intensity ratio method to derive Tex ----------
def tau_11m(s11_mg, s11_isg):
    tau_m = np.arange(0.001, 30, 0.02)
    ratio_ms = (1 - np.exp(-tau_m)) / (1 - np.exp(-tau_m/1.8))
    i_tau = np.argmin(np.abs(s11_mg/s11_isg) - ratio_ms)
    return tau_m[i_tau]

def Tex_intr(s11_mg, s11_isg, s22_mg):
    tau_11 = tau_11m(s11_mg, s11_isg)
    Tex_int0 = -41.5 / np.log(-0.42 / tau_11 * np.log(1 - np.abs(s22_mg/s11_mg) * (1 - np.exp(-tau_11))))
    return Tex_int0

print(tau_11m(10, 6), Tex_intr(10, 5, 4))


#%%  ---------------- simulated spectra noise -----------------
cw = 0.02
wm = 4.0
Tex_ini = 22
dv = 2.0
rms = 0.3
tau_1 = 0.7
tau_2 = tau_1/0.8038 * np.exp(-41.5 / Tex_ini)    # g11*f22*A11
v_axis = np.arange(-30.0, 30.0, cw)

i11_m = np.argwhere((v_axis < wm) & (v_axis > -wm))
i11_isg0 = np.argwhere((v_axis < (-7.59 + wm)) & (v_axis > (-7.59 - wm)))
i11_isg1 = np.argwhere((v_axis < (7.59 + wm)) & (v_axis > (7.59 - wm)))
i11_osg0 = np.argwhere((v_axis < (19.51 + wm)) & (v_axis > (19.51 - wm)))
i11_osg1 = np.argwhere((v_axis < (-19.4948 + wm)) & (v_axis > (-19.4948 - wm)))
i22_mg = np.argwhere((v_axis < (0.00068 + wm)) & (v_axis > (0.00068 - wm)))
i22_isg0 = np.argwhere((v_axis < (-16.4 + wm)) & (v_axis > (-16.4 - wm)))
i22_isg1 = np.argwhere((v_axis < (16.4 + wm)) & (v_axis > (16.4 - wm)))

T11_thy = T11_theory(v_axis, dv, tau_1, Tex=Tex_ini, osg1=1, osg2=1, mg=1, isg1=1, isg2=1)
T22_thy = T22_theory(v_axis, dv, tau_2, Tex=Tex_ini, osg1=1, osg2=1, mg=1, isg1=1, isg2=1)

#tau_1 = N_tot * 9.1288e-15 * np.exp(-23.279 / Tex_ini) \
#        / (0.7604 + 0.0519 * Tex_ini + 0.00094361 * Tex_ini ** 2)

nts = 1000
err_hf = np.zeros(nts)
err_int = np.zeros(nts)
for i in np.arange(nts):
    T11 = T11_thy + np.random.randn(len(v_axis)) * rms
    T22 = T22_thy + np.random.randn(len(v_axis)) * rms
    S11_mg = np.sum(T11[i11_m] * cw)
    S11_isg = np.sum(T11[i11_isg0]) * cw + np.sum(T11[i11_isg1]) * cw
    S11_osg = np.sum(T11[i11_osg0]) * cw + np.sum(T11[i11_osg1]) * cw
    S22_mg = np.sum(T22[i22_mg] * cw)
    S22_isg = np.sum(T22[i22_isg0]) * cw + np.sum(T22[i22_isg1]) * cw
    Tex_hf = Tex_main(S11_mg, S11_isg, S11_osg, S22_mg, S22_isg, dv=dv)
    Tex_int = Tex_intr(S11_mg, S11_isg, S22_mg)
    err_hf[i] = Tex_hf - Tex_ini
    err_int[i] = Tex_int - Tex_ini

# err_hf -= err_hf.mean()
# err_int -= err_int.mean()

# %% -------- error plot ----------
rc("font", family="serif", size=12)
fig = plt.figure(figsize=(15, 6))
fig.clear()
ax0 = fig.add_subplot(1, 3, 1)
ax0.scatter(np.arange(nts), err_hf, c='red', marker='o', s=10)
ax0.scatter(np.arange(nts), err_int, c='blue', marker='s', s=10)
ax0 = fig.add_subplot(1, 3, 2)
ax0.plot(v_axis, T11, c='blue', drawstyle='steps', linewidth=1)
ax0.plot(v_axis, T22+3, c='red', drawstyle='steps', linewidth=1)

dT = 0.2
bins = np.arange(-6, 6, dT)
hist_Trot, _tp = np.histogram(err_int - err_int.mean(), bins=bins)
hist_Trot = hist_Trot/np.sum(hist_Trot)/dT
ax = fig.add_subplot(1, 3, 3)
ax.plot(bins[:-1]+dT/2, hist_Trot, c='blue', drawstyle='steps', linewidth=2)
ax.set_xlabel(r'$\mathrm{ \Delta T_{rot}}$')
ax.set_ylabel('ratio')
ax.set_title('statistics')
plt.savefig('Trot_hist.pdf', fmt='pdf')
plt.show()

print(err_hf.std(), err_int.std())
print(err_hf.mean(), err_int.mean())

#%% ------------- dT_error vs dv ------------------
cw = 0.02
wm = 4.0
Tex_ini = 24
rms = 0.1
tau_1 = 1.7
tau_2 = tau_1/0.8038 * np.exp(-41.5 / Tex_ini)    # g11*f22*A11
v_axis = np.arange(-30.0, 30.0, cw)

dv_arr = np.arange(0.11, 3.0, 0.2)
n_dv = len(dv_arr)
nts = 250

dT_hf_arr = np.zeros(n_dv)
dT_int_arr = np.zeros(n_dv)
std_hf_arr = np.zeros(n_dv)
std_int_arr = np.zeros(n_dv)

for j in np.arange(n_dv):
    T11_thy = T11_theory(v_axis, dv_arr[j], tau_1, Tex=Tex_ini, osg1=1, osg2=1, mg=1, isg1=1, isg2=1)
    T22_thy = T22_theory(v_axis, dv_arr[j], tau_2, Tex=Tex_ini, osg1=1, osg2=1, mg=1, isg1=1, isg2=1)
    err_hf = np.zeros(nts)
    err_int = np.zeros(nts)
    for i in np.arange(nts):
        T11 = T11_thy + np.random.randn(len(v_axis)) * rms
        T22 = T22_thy + np.random.randn(len(v_axis)) * rms
        S11_mg = np.sum(T11[i11_m] * cw)
        S11_isg = np.sum(T11[i11_isg0]) * cw + np.sum(T11[i11_isg1]) * cw
        S11_osg = np.sum(T11[i11_osg0]) * cw + np.sum(T11[i11_osg1]) * cw
        S22_mg = np.sum(T22[i22_mg] * cw)
        S22_isg = np.sum(T22[i22_isg0]) * cw + np.sum(T22[i22_isg1]) * cw
        Tex_hf = Tex_main(S11_mg, S11_isg, S11_osg, S22_mg, S22_isg, dv=dv)
        Tex_int = Tex_intr(S11_mg, S11_isg, S22_mg)
        err_hf[i] = Tex_hf - Tex_ini
        err_int[i] = Tex_int - Tex_ini
    dT_hf_arr[j] = err_hf.mean()
    dT_int_arr[j] = err_int.mean()
    std_hf_arr[j] = err_hf.std()
    std_int_arr[j] = err_int.std()

# ---------
rc("font", family="serif", size=12)
rc('ytick', direction='in', labelsize=12)
rc('xtick', direction='in', labelsize=12)
fig = plt.figure(figsize=(5, 4))
fig.clear()
fig.subplots_adjust(0.2, 0.3, 0.9, 0.9)
ax0 = fig.add_subplot(1, 1, 1)
ax0.errorbar(dv_arr, dT_hf_arr, yerr=std_hf_arr/2, fmt='rs', markersize=6, label='HFGR')
ax0.errorbar(dv_arr, dT_int_arr, yerr=std_int_arr/2, fmt='bs', markersize=6, label='Intensity ratio')
ax0.set_xlabel('Line width (km/s)', fontsize=12)
ax0.set_ylabel(r'$T_{rot}$ erro (K)', fontsize=12)
ax0.legend(loc=4)
plt.show()

#%% ---------------- HFC fitting error -------------
import pyspeckit
from astropy import units as u
from pyspeckit.spectrum.models import ammonia_constants, ammonia, ammonia_hf
from pyspeckit.spectrum.models.ammonia_constants import freq_dict
from pyspeckit.spectrum.units import SpectroscopicAxis, SpectroscopicAxes

d2s = np.sqrt(8.0*np.log(2.0))

# %% ------------ HF fitting and method ------------
wm = 4.0; Tex_ini = 28; rms = 0.2; dv = 0.8;
tau_1 = 2.5
tau_2 = tau_1/0.8038 * np.exp(-41.5 / Tex_ini)    # g11*f22*A11

# cw_arr = np.arange(0.1, 1.3, 0.3)
cw_arr = np.arange(0.05, 0.5, 0.11)
n_cw = len(cw_arr)
nts = 6

dT_hf_arr = np.zeros(n_cw)
dT_kit_arr = np.zeros(n_cw)
std_hf_arr = np.zeros(n_cw)
std_kit_arr = np.zeros(n_cw)
err_hf = np.zeros((n_cw, nts))
err_kit = np.zeros((n_cw, nts))

#
'''
fig = plt.figure(figsize=(5, 4))
fig.clear()
fig.subplots_adjust(0.1, 0.1, 0.9, 0.9)
ax1 = fig.add_subplot(1, 1, 1)
'''

# pre-fit for Tex and tau_1:

'''
cw = cw_arr[0]
v_axis = np.arange(-30.0, 30.0, cw)
T11_thy = T11_theory(v_axis, dv, tau_1, Tex=Tex_ini, osg1=1, osg2=1, mg=1, isg1=1, isg2=1)
xarr11 = SpectroscopicAxis(v_axis * u.km / u.s, velocity_convention='radio',
                           refX=freq_dict['oneone']).as_unit(u.GHz)
nh3_sp = pyspeckit.Spectrum(xarr=xarr11, data=T11_thy, header={})
nh3_sp.plotter(axis=ax0, clear = True)
nh3_sp.specfit(fittype='ammonia_tau',
               guesses=[Tex_ini, Tex0, tau_1, dv/d2s, 0, 0.5],
               fixed=[1, 0, 0, 0, 0, 1],
               quiet=False)
'''

for i in np.arange(n_cw):
    cw = cw_arr[i]
    # v_axis ~~~ cw:
    v_axis = np.arange(-30.0, 30.0, cw)
    i11_m = np.argwhere((v_axis < wm) & (v_axis > -wm))
    i11_isg0 = np.argwhere((v_axis < (-7.59 + wm)) & (v_axis > (-7.59 - wm)))
    i11_isg1 = np.argwhere((v_axis < (7.59 + wm)) & (v_axis > (7.59 - wm)))
    i11_osg0 = np.argwhere((v_axis < (19.51 + wm)) & (v_axis > (19.51 - wm)))
    i11_osg1 = np.argwhere((v_axis < (-19.4948 + wm)) & (v_axis > (-19.4948 - wm)))
    i22_mg = np.argwhere((v_axis < (0.00068 + wm)) & (v_axis > (0.00068 - wm)))
    i22_isg0 = np.argwhere((v_axis < (-16.4 + wm)) & (v_axis > (-16.4 - wm)))
    i22_isg1 = np.argwhere((v_axis < (16.4 + wm)) & (v_axis > (16.4 - wm)))
    T11_thy = T11_theory(v_axis, dv, tau_1, Tex=Tex_ini, osg1=1, osg2=1, mg=1, isg1=1, isg2=1)
    T22_thy = T22_theory(v_axis, dv, tau_2, Tex=Tex_ini, osg1=1, osg2=1, mg=1, isg1=1, isg2=1)
    for j in np.arange(nts):
        T11 = T11_thy + np.random.randn(len(v_axis)) * rms
        T22 = T22_thy + np.random.randn(len(v_axis)) * rms
        # ------------ hf Trot ---------------
        S11_mg = np.sum(T11[i11_m] * cw)
        S11_isg = np.sum(T11[i11_isg0]) * cw + np.sum(T11[i11_isg1]) * cw
        S11_osg = np.sum(T11[i11_osg0]) * cw + np.sum(T11[i11_osg1]) * cw
        S22_mg = np.sum(T22[i22_mg] * cw)
        S22_isg = np.sum(T22[i22_isg0]) * cw + np.sum(T22[i22_isg1]) * cw
        Tex_hf = Tex_main(S11_mg, S11_isg, S11_osg, S22_mg, S22_isg, dv=dv)
        # ----------- speckit Tex ------------
        xarr11 = SpectroscopicAxis(v_axis * u.km / u.s, velocity_convention='radio',
                                   refX=freq_dict['oneone']).as_unit(u.GHz)
        xarr22 = SpectroscopicAxis(v_axis * u.km / u.s, velocity_convention='radio',
                                   refX=freq_dict['twotwo']).as_unit(u.GHz)
        xarr = SpectroscopicAxes([xarr11, xarr22])
        synthspec = np.hstack((T11, T22))
        nh3_sp = pyspeckit.Spectrum(xarr=xarr, data=synthspec, header={})
        # nh3_sp.plotter(axis=nh3_sp.plotter.axis,clear=True)
        # trot=20, tex=None, ntot=14.8, width=1, xoff_v=0.0, fortho=0.5
        # nh3_sp.plotter(axis=ax1, clear=False)  # axis=ax1
        nh3_sp.specfit(fittype='ammonia_tau', guesses=[Tex_ini, Tex_ini-10, tau_1/2, dv/d2s, 0, 0.5],
                       fixed=[0, 1, 0, 1, 1, 1]);
        Tex_kit = nh3_sp.specfit.parinfo[0]['value']
        plt.show()
        # ----------- Tex error -----------
        err_hf[i, j] = Tex_hf
        err_kit[i, j] = Tex_kit

# %%
fig = plt.figure(figsize=(5, 4))
fig.clear()
fig.subplots_adjust(0.1, 0.1, 0.9, 0.9)
ax0 = fig.add_subplot(1, 1, 1)

for i in np.arange(n_cw):
    cw_xi = np.ones(nts) * cw_arr[i]
    # ax0.scatter(cw_xi, err_hf[i, :], c='red', marker='.', s=6)
    # ax0.scatter(cw_xi, err_kit[i, :], c='green', marker='.', s=6)
    dT_hf_arr[i] = err_hf[i, :].mean()
    dT_kit_arr[i] = err_kit[i, :].mean()
    std_hf_arr[i] = err_hf[i, :].std()
    std_kit_arr[i] = err_kit[i, :].std()

ax0.errorbar(cw_arr, dT_hf_arr, yerr=std_hf_arr/2, fmt='rs', markersize=5, label='HFGR')
ax0.errorbar(cw_arr, dT_kit_arr, yerr=std_kit_arr/2, fmt='gs', markersize=5, label='HF fitting')
ax0.set_xlabel('Line width (km/s)', fontsize=12)
ax0.set_ylabel(r'$T_{rot}$ erro (K)', fontsize=12)
#ax0.legend(loc=4)
plt.show()

#%% ------------- dT_error vs dv +++ HFC fitting ------------------
cw = 0.02
wm = 4.0
Tex_ini = 22
rms = 0.1
tau_1 = 1.9
tau_2 = tau_1/0.8038 * np.exp(-41.5 / Tex_ini)    # g11*f22*A11
v_axis = np.arange(-30.0, 30.0, cw)

dv_arr = np.arange(0.11, 3.0, 0.2)
n_dv = len(dv_arr)
nts = 12

dT_hf_arr = np.zeros(n_dv)
dT_int_arr = np.zeros(n_dv)
dT_kit_arr = np.zeros(n_dv)
std_hf_arr = np.zeros(n_dv)
std_int_arr = np.zeros(n_dv)
std_kit_arr = np.zeros(n_dv)

for j in np.arange(n_dv):
    T11_thy = T11_theory(v_axis, dv_arr[j], tau_1, Tex=Tex_ini, osg1=1, osg2=1, mg=1, isg1=1, isg2=1)
    T22_thy = T22_theory(v_axis, dv_arr[j], tau_2, Tex=Tex_ini, osg1=1, osg2=1, mg=1, isg1=1, isg2=1)
    err_hf = np.zeros(nts)
    err_int = np.zeros(nts)
    err_kit = np.zeros(nts)
    for i in np.arange(nts):
        T11 = T11_thy + np.random.randn(len(v_axis)) * rms
        T22 = T22_thy + np.random.randn(len(v_axis)) * rms
        S11_mg = np.sum(T11[i11_m] * cw)
        S11_isg = np.sum(T11[i11_isg0]) * cw + np.sum(T11[i11_isg1]) * cw
        S11_osg = np.sum(T11[i11_osg0]) * cw + np.sum(T11[i11_osg1]) * cw
        S22_mg = np.sum(T22[i22_mg] * cw)
        S22_isg = np.sum(T22[i22_isg0]) * cw + np.sum(T22[i22_isg1]) * cw
        # ---------------- HFGR -------------------
        Tex_hf = Tex_main(S11_mg, S11_isg, S11_osg, S22_mg, S22_isg, dv=dv)
        # ---------------- intensity ratio ---------------
        Tex_int = Tex_intr(S11_mg, S11_isg, S22_mg)
        # ----------- speckit Tex ------------
        xarr11 = SpectroscopicAxis(v_axis * u.km / u.s, velocity_convention='radio',
                                   refX=freq_dict['oneone']).as_unit(u.GHz)
        xarr22 = SpectroscopicAxis(v_axis * u.km / u.s, velocity_convention='radio',
                                   refX=freq_dict['twotwo']).as_unit(u.GHz)
        xarr = SpectroscopicAxes([xarr11, xarr22])
        synthspec = np.hstack((T11, T22))
        nh3_sp = pyspeckit.Spectrum(xarr=xarr, data=synthspec, header={})
        # nh3_sp.plotter(axis=nh3_sp.plotter.axis,clear=True)
        # trot=20, tex=None, ntot=14.8, width=1, xoff_v=0.0, fortho=0.5
        # nh3_sp.plotter(axis=ax1, clear=False)  # axis=ax1
        nh3_sp.specfit(fittype='ammonia_tau', guesses=[Tex_ini, Tex_ini-10, tau_1/2, dv/d2s, 0, 0.5],
                       fixed=[0, 1, 0, 1, 1, 1]);
        Tex_kit = nh3_sp.specfit.parinfo[0]['value']
        err_hf[i] = Tex_hf - Tex_ini
        err_int[i] = Tex_int - Tex_ini
        err_kit[i] = Tex_kit - Tex_ini
    dT_hf_arr[j] = err_hf.mean()
    std_hf_arr[j] = err_hf.std()
    dT_int_arr[j] = err_int.mean()
    std_int_arr[j] = err_int.std()
    dT_kit_arr[j] = err_kit.mean()
    std_kit_arr[j] = err_kit.std()

#%% ---------
rc("font", family="serif", size=12)
rc('ytick', direction='in', labelsize=12)
rc('xtick', direction='in', labelsize=12)
fig = plt.figure(figsize=(6, 5))
fig.clear()
fig.subplots_adjust(0.2, 0.2, 0.9, 0.9)
ax0 = fig.add_subplot(1, 1, 1)
ax0.plot(dv_arr, dT_hf_arr, 'r')
ax0.plot(dv_arr, dT_int_arr, 'b--')
ax0.plot(dv_arr, dT_kit_arr-dT_kit_arr[0], 'g:')
ax0.errorbar(dv_arr, dT_hf_arr, yerr=std_hf_arr/2, fmt='rs', markersize=4, label='HFGR')
ax0.errorbar(dv_arr, dT_int_arr, yerr=std_int_arr/2, fmt='b^', markersize=4, label='Intensity ratio')
ax0.errorbar(dv_arr, dT_kit_arr-dT_kit_arr[0], yerr=std_kit_arr/2, fmt='gD', markersize=4, label='HF fitting')
ax0.set_xlabel('Line width (km/s)', fontsize=12)
ax0.set_ylabel(r'$T_{rot}$ error (K)', fontsize=12)
ax0.legend(loc=4, fontsize=11)
plt.savefig('error_compare.pdf', fmt='pdf')
plt.show()


# %% ---------------- single spectra ------------------
'''
G14_4876-P1-NH3_1-1_K.txt	G14_4876-P3-NH3_1-1_K.txt	G34_7391-P2-NH3_11_K.txt
G14_4876-P1-NH3_2-2_K.txt	G14_4876-P3-NH3_2-2_K.txt	G34_7391-P2-NH3_22_K.txt
G14_4876-P2-NH3_1-1_K.txt	G34_7391-P1-NH3_11_K.txt	G34_7391-P3-NH3_11_K.txt
G14_4876-P2-NH3_2-2_K.txt	G34_7391-P1-NH3_22_K.txt	G34_7391-P3-NH3_22_K.txt
'''

df11_arr = ['G14_4876-P1-NH3_1-1_K.txt', 'G14_4876-P2-NH3_1-1_K.txt',
            'G14_4876-P3-NH3_1-1_K.txt', 'G34_7391-P1-NH3_11_K.txt',
            'G34_7391-P2-NH3_11_K.txt', 'G34_7391-P3-NH3_11_K.txt']
df22_arr = ['G14_4876-P1-NH3_2-2_K.txt', 'G14_4876-P2-NH3_2-2_K.txt',
            'G14_4876-P3-NH3_2-2_K.txt', 'G34_7391-P1-NH3_22_K.txt',
            'G34_7391-P2-NH3_22_K.txt', 'G34_7391-P3-NH3_22_K.txt']

pos_arr = ['G14_4876-P1', 'G14_4876-P2', 'G14_4876-P3', 'G34_7391-P1', 'G34_7391-P2', 'G34_7391-P3']

dir1 = '/Users/rzy/data/wangsheng/fsy_data/'

#%%
for i in np.arange(6):
    D_11 = np.loadtxt(dir1+df11_arr[i])
    D_22 = np.loadtxt(dir1+df22_arr[i])
    v11_ax = D_11[:, 0]
    T11_arr = D_11[:, 1]
    v22_ax = D_22[:, 0]
    T22_arr = D_22[:, 1]
    '''
    fig = plt.figure(figsize=(6, 5))
    ax0 = fig.add_subplot(1, 1, 1)
    ax0.plot(v11_ax, T11_arr, 'b', drawstyle='steps')
    ax0.plot(v22_ax, T22_arr-0.5, 'r', drawstyle='steps')
    plt.show()
    '''
    #%%
    # wm = 4.5; Tex_ini = 28; rms = 0.2;
    dv = 1.2
    iv0 = np.argmax(T11_arr)
    v_sys = v11_ax[iv0]

    v11_ax = D_11[:, 0] - v_sys
    v22_ax = D_22[:, 0] - v_sys
    cw_11 = v11_ax[1] - v11_ax[0]
    cw_22 = v11_ax[1] - v11_ax[0]

    i11_m = np.argwhere((v11_ax < wm) & (v11_ax > -wm))
    i11_isg0 = np.argwhere((v11_ax < (-7.59 + wm)) & (v11_ax > (-7.59 - wm)))
    i11_isg1 = np.argwhere((v11_ax < (7.59 + wm)) & (v11_ax > (7.59 - wm)))
    i11_osg0 = np.argwhere((v11_ax < (19.51 + wm)) & (v11_ax > (19.51 - wm)))
    i11_osg1 = np.argwhere((v11_ax < (-19.4948 + wm)) & (v11_ax > (-19.4948 - wm)))

    i22_mg = np.argwhere((v22_ax < (0.00068 + wm)) & (v22_ax > (0.00068 - wm)))
    i22_isg0 = np.argwhere((v22_ax < (-16.4 + wm)) & (v22_ax > (-16.4 - wm)))
    i22_isg1 = np.argwhere((v22_ax < (16.4 + wm)) & (v22_ax > (16.4 - wm)))

    S11_mg = np.sum(T11_arr[i11_m] * cw_11)
    S11_isg = np.sum(T11_arr[i11_isg0]) * cw_11 + np.sum(T11_arr[i11_isg1]) * cw_11
    S11_osg = np.sum(T11_arr[i11_osg0]) * cw_22 + np.sum(T11_arr[i11_osg1]) * cw_22
    S22_mg = np.sum(T22_arr[i22_mg] * cw_22)
    S22_isg = np.sum(T22_arr[i22_isg0]) * cw_22 + np.sum(T22_arr[i22_isg1]) * cw_22
    # ---------------- HFGR -------------------
    Tex_hf = Tex_main(S11_mg, S11_isg, S11_osg, S22_mg, S22_isg, dv=dv)
    # ---------------- intensity ratio ---------------
    Tex_int = Tex_intr(S11_mg, S11_isg, S22_mg)
    print('%s: Trot(HFGR) = %.2f, Trot(intensity ratio) = %.2f' %(pos_arr[i], Tex_hf, Tex_int))
