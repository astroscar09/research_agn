from astropy.table import Table
from scipy.interpolate import interp1d
from astropy import units as u
from astropy.io import fits

def read_jwst_prism_resolution():

    prism_resolution = Table.read('data/jwst_nirspec_prism_disp.fits')

    return prism_resolution


def generate_prism_R_function(wave, R):

    interp_prism = interp1d(wave, R, bounds_error=False, fill_value='extrapolate')
    return interp_prism
    
def convert_resolution_to_kms(resolution):

    R_kms = (resolution**-1)*3e5

    return R_kms

def miri_spectral_resolution(wave_micron):
    
    '''
    This is the miri spectral resolution as a function of wavelength, this was taken from:
    https://jwst-docs.stsci.edu/jwst-mid-infrared-instrument/miri-observing-modes/miri-medium-resolution-spectroscopy#gsc.tab=0
    
    See Figure 3 for equation used in this function
    
    '''
    R = 4603 - 128*wave_micron + 10**(-7.4*wave_micron)

    return R

def convert_fnu_to_flam(wave, fnu, fnu_err, flux_units = u.Jansky, wave_units = u.micrometer):
    
    conversion = (3e8*1e10)/(wave*wave_units.to(u.AA))**2
    flam = (fnu*flux_units.to(u.erg/u.s/u.cm**2/u.Hz))*conversion
    flamerr = (fnu_err*flux_units.to(u.erg/u.s/u.cm**2/u.Hz))*conversion
    
    return flam, flamerr
    

def create_binary_table_spectra(wave, flux, fluxerr, R_kms, wave_unit, flux_unit, redshift):

    t = Table()
    t['wav'] = wave[idx_sorted] *wave_unit
    t['flux'] = flux[idx_sorted] *flux_unit
    t['fluxerr'] = fluxerr[idx_sorted]*flux_unit
    t['R_kms'] = R_kms[idx_sorted] *u.km/u.s

    bin_hdu = fits.BinTableHDU(data = t)
    bin_hdu.header['REDSHIFT'] = redshift
    return bin_hdu

def making_spectra_fits_file(bin_hdu, filename):
    hdu = fits.PrimaryHDU()
    hdul = fits.HDUList([hdu, bin_hdr])

    hdul.writeto(f'{filename}', overwrite=True)