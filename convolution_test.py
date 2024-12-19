from Post_BEAGLE_Analysis import *
from Helper_Functions import *
#import seaborn as sb
plt.rcParams['xtick.labelsize'] = 17
plt.rcParams['ytick.labelsize'] = 17
import pickle

tab = Table.read('jwst_nirspec_prism_disp.fits')
r_wave, r_curve = tab['WAVELENGTH'], tab['R']
r_wave *= 1e4


bgl_output = Beagle_Output('NEW_ENV_w_LSF_and_Freez/1395_BEAGLE.fits.gz')
redshift = bgl_output['gal_prop']['redshift']
z = redshift[0]


tb = Table.read('1395_Simple_NIRSpec_Spectra_w_Rkms.fits')
obj_spec_wave = tb['wav']*1e10
bgl_output.generate_convolved_spectra(r_wave, r_curve, tb['wav']*1e10)

obs_wave_full_model = bgl_output['full_sed_wvln'] * (1+redshift[0])
data_AA = tb['wav']*1e10

data_mask = (obs_wave_full_model > data_AA[0]) & (obs_wave_full_model < data_AA[-1])

full_sed = bgl_output['full_sed'][0][data_mask]
agn_sed = bgl_output['agn_sed'][0][data_mask]
model_wave = obs_wave_full_model[data_mask]

#LOOKING ONLY AT CIV LINE

CIV_range = (20400 < model_wave) & (model_wave <21000)
test_sed = full_sed/(1+redshift[0])
in_range_sed = test_sed[CIV_range]
in_range_wave = model_wave[CIV_range]

#FULL Model
civ_func = interp1d(in_range_wave, in_range_sed)
delta_lambda = in_range_wave[1:] - in_range_wave[:-1]
binc_full = (in_range_wave[1:] + in_range_wave[:-1])/2
civ_binc = civ_func(binc_full)
print(np.sum(delta_lambda*civ_binc))

#data 
mask = (tb['wav']*1e10 > 20400) & (tb['wav']*1e10< 21000)
civ_wav, civ_f = tb['wav'][mask]*1e10, tb['flux'][mask]
delta_lam = civ_wav[1:] - civ_wav[:-1]
binc_data = (civ_wav[1:] + civ_wav[:-1])/2
civ_func_data = interp1d(civ_wav, civ_f)
print(np.sum(delta_lam* civ_func_data(binc_data)))

#Convolution Model
conv_mask = (bgl_output['convolved_wave'] > 20400) & (bgl_output['convolved_wave']< 21000)
civ_wav_conv, civ_f_conv = bgl_output['convolved_wave'][conv_mask], bgl_output['convolved_spectra'][0][conv_mask]/(1+z)
delta_lam = civ_wav_conv[1:] - civ_wav_conv[:-1]
binc_conv= (civ_wav_conv[1:] + civ_wav_conv[:-1])/2
civ_func_data_conv = interp1d(civ_wav_conv, civ_f_conv)
print(np.sum(delta_lam* civ_func_data_conv(binc_conv)))

#getting lSF from the NIRSPEC Prism curve

LSF = r_wave/r_curve #delta_lambda in units of angstroms

near_civ = (r_wave > 20400) & (r_wave < 21000)

med_LSF_CIV = np.median(LSF[near_civ])

#making a kernel with that med LSF near CIV
sigma = med_LSF_CIV #unsure if this is sigma or fwhm otherwise a factor of 2.355 is needed if FWHM
kernel = np.exp(-0.5 * (np.linspace(-3*sigma, 3*sigma, 100) / sigma)**2)
kernel /= np.sum(kernel)  # Normalize the kernel

# Convolve the spectrum with the kernel
smoothed_spectrum = np.convolve(in_range_sed, kernel, mode='same')

plt.figure(figsize = (10, 5), constrained_layout = True)
plt.step(civ_wav_conv, civ_f_conv, label = 'conv', color = 'black')
plt.step(civ_wav, civ_f, label = 'data', color = 'red', where = 'mid')
plt.step(in_range_wave, in_range_sed, label = 'full_model', where = 'mid', color = 'purple')
plt.step(in_range_wave, smoothed_spectrum, label = 'convolved manually', where = 'mid', color = 'orange')
plt.legend()
plt.savefig('CIV_Convolved_Testing.png')