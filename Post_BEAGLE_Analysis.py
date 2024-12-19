from astropy.io import fits
from astropy.table import Table
import numpy as np
import pandas as pd
from scipy.interpolate import Akima1DInterpolator
from scipy.integrate import trapezoid
from astropy import units as u
from scipy.interpolate import interp1d
from astropy.convolution import convolve, convolve_fft
import spectres as spectres
import matplotlib.pyplot as plt

cobaltblue = '#2e37fe'
plt.rcParams['axes.linewidth'] = 1.5
plt.rcParams['font.family'] = 'serif'
plt.rcParams['xtick.labelsize'] = 13
plt.rcParams['ytick.labelsize'] = 13


class Beagle_Output(dict):

    def __init__(self, filename):

        #self["gal_prop"]   =  

        self["gal_prop"]      = None
        self['SFH']           = None
        self['marg_wvln']     = None
        self['marg_sed']      = None
        self['agn_emission']  = None
        self['hii_emission']  = None
        self['full_sed_wvln'] = None
        self['full_sed']      = None
        self['app_mag']       = None
        self['abs_mag']       = None
        self['posterior']     = None
        self['SFR']           = None
        
        self.read_file(filename)

    def read_beagle_output_file(self, filename):
    
        hdu = fits.open(filename)
    
        return hdu
    
    def read_extension_file(self, filename, extension):
    
        tab = Table.read(filename, hdu = extension)
    
        return tab
    
    def grab_spectra_arrays(self, hdu):
        
        '''
        Reading the inpout file and grabbing the spectra array which is in extension 12 as seen below:
    
        # 10  MARGINAL SED    1 ImageHDU         8   (388, 1480)   float32   
         # 11  FULL SED WL    1 BinTableHDU     12   1R x 1C   [13391E]   
         # 12  FULL SED      1 ImageHDU         9   (13391, 1480)   float32   
         # 13  APPARENT MAGNITUDES    1 BinTableHDU     30   1480R x 7C   [1E, 1E, 1E, 1E, 1E, 1E, 1E]   
         # 14  ABSOLUTE MAGNITUDES    1 BinTableHDU     30   1480R x 7C   [1E, 1E, 1E, 1E, 1E, 1E, 1E]   
         # 15  POSTERIOR PDF    1 BinTableHDU     35   1480R x 13C   [1E, 1E, 1E, 1E, 1E, 1E, 1E, 1E, 1E, 1E, 1E, 1E, 1E] 
         
        '''
         
        extension = 'FULL SED'
    
        spec_array = hdu[extension].data
        
        self['full_sed'] = spec_array 
        
    
    def grab_spec_wvln(self, hdu):
    
        '''
        Reading the inpout file and grabbing the spectra array which is in extension 11 as seen below:
    
        # 10  MARGINAL SED    1 ImageHDU         8   (388, 1480)   float32   
         # 11  FULL SED WL    1 BinTableHDU     12   1R x 1C   [13391E]   
         # 12  FULL SED      1 ImageHDU         9   (13391, 1480)   float32   
         # 13  APPARENT MAGNITUDES    1 BinTableHDU     30   1480R x 7C   [1E, 1E, 1E, 1E, 1E, 1E, 1E]   
         # 14  ABSOLUTE MAGNITUDES    1 BinTableHDU     30   1480R x 7C   [1E, 1E, 1E, 1E, 1E, 1E, 1E]   
         # 15  POSTERIOR PDF    1 BinTableHDU     35   1480R x 13C   [1E, 1E, 1E, 1E, 1E, 1E, 1E, 1E, 1E, 1E, 1E, 1E, 1E] 
         
        '''
    
        extension = 'FULL SED WL'
    
        wvln_array = Table(hdu[extension].data)['wl'].value[0]
        self['full_sed_wvln'] = wvln_array
        
        #return wvln_array
    
    
    def grab_spectra_info(self, hdu):
    
        '''
        Reading the input file and grabbing the spectra information, i.e. wavelength and flux
        '''
    
        self.grab_spectra_arrays(hdu)
        
        self.grab_spec_wvln(hdu)
    
        #return wvln, twoD_spec_arr
    
    
    def generate_plot_setup(self, figsize = (12, 6)):
    
        fig, ax = plt.subplots(figsize = figsize, constrained_layout = True)
    
        return fig, ax
    
    
    def get_bounds_from_spec(self, spectra):
    
        '''
        Function to get the 16th and 84th percentile of the spectra array
        '''
    
        l16_spec, u84_spec = np.percentile(spectra, q = (16, 84), axis = 0)
    
        return l16_spec, u84_spec 
    
    def get_abs_mag_ext(self, hdu):
        
        extension = 'ABSOLUTE MAGNITUDES'
        
        abs_mag_tab = Table(hdu[extension].data)

        self['abs_mag'] = abs_mag_tab
        
        #return abs_mag_tab
    
    def get_app_mag_ext(self, hdu):
        
        extension = 'APPARENT MAGNITUDES'
        
        app_mag_tab = Table(hdu[extension].data)

        self['app_mag'] = app_mag_tab
        
        #return app_mag_tab
    
    def convert_mag_to_Fnu(self, mag_tab):
    
        mag_df = mag_tab.to_pandas()
    
        flux_fnu_df = 10**((mag_df + 48.6)/-2.5)
    
        return flux_fnu_df
    
    def get_Fnu_dist_abs_mag_fit(self):
        
        mag_tab = self['abs_mag']
        fnu_df = self.convert_mag_to_Fnu(mag_tab)
    
        return fnu_df
    
    def get_Fnu_dist_app_mag_fit(self):
        
        mag_tab = self['app_mag']
        fnu_df = self.convert_mag_to_Fnu(mag_tab)
    
        return fnu_df
    
    def read_filters(self, filepath):
    
        filt_tab = Table.read(filepath)
        self['Filters'] = filt_tab
        #return filt_tab
    
    def read_filt_col(self, col):
        
        filt_tab = self['Filters']
        
        wave = filt_tab[col].value[0][0]
        tlam = filt_tab[col].value[0][1]
    
        return wave, tlam
    
    def avg_wavelength_filter(self, wavelength, Tlam):
    
        num = wavelength*Tlam
        denom = Tlam
        
        int_num = trapezoid(num, dx = wavelength[1] - wavelength[0])
        int_denom = trapezoid(denom, dx =  wavelength[1] - wavelength[0])
    
        avg_wave = int_num/int_denom
    
        return avg_wave
    
    def compute_avg_wave_per_filter(self, filepath):
        
        self.read_filters(filepath)
        
        filters = self['Filters']
        columns = filters.colnames
        
        pivot_waves = []
        
        for col in columns:
            wave, tlam = self.read_filt_col(col)
            avg_wave = self.avg_wavelength_filter(wave, tlam)
            pivot_waves.append(avg_wave)
        
        self['pivot_wvln'] = np.array(pivot_waves)
        
            
    def integrating_spectra_filter_curves(self, wave_spec, flux_spec, wave_filt, T_filt):
    
        min_filt_wave = np.amin(wave_filt)
        max_filt_wave = np.amax(wave_filt)
        
        mask = (min_filt-wave< wave_spec) & (wave_filt < max_filt_wave)
        
        masked_wave = wave_spec[mask]
        masked_flux = flux_spec[mask]
    
        interp_spec = Akima1DInterpolator(x = masked_wave, y = masked_flux)
    
        mapped_flux = interp_spec(wave_filt)
    
        num = wave_filt*mapped_flux*T_filt
        denom = wave_filt*T_filt
    
        num_integral = traepzoid(num)
        denom_integral = trapezoid(denom)
        
        avg_flux = num_integral/denom_integral
    
        return avg_flux
    
    def read_spectra(self, filepath):
        
        spec_tab = Table.read(filepath)

        self['Input_Spectra'] = spec_tab
    
    
    def getting_spec_quantities(self, filepath):
    
        self.read_spectra(filepath)
        
        spec_tab = self['Input_Spectra']
        
        wave, flux, flux_err = spec_tab['wav'], spec_tab['flux'], spec_tab['fluxerr']
    
        wave_angstrom = wave*1e10
    
        return wave_angstrom, flux, flux_err
    
    def convert_spectra_to_fnu(self, wave, flux, fluxerr):
    
        c_m_s = 3e8
        c_A_s = c_m_s * 1e10
        
        fnu_conversion_factor = wave**2/c_A_s
    
        fnu_spec = flux*fnu_conversion_factor
        fnu_spec_err = fluxerr*fnu_conversion_factor
    
        return fnu_spec, fnu_spec_err
        
    def compute_spectra_model_photom(self, filter_file):
    
        wvln, spec_2D = self['full_sed_wvln'], self['full_sed']
        
        self.read_filters(filter_file)
        filters = self['Filters']
        z = self['z']
        
        columns = filters.colnames
    
        obs_wvln = wvln*(1+z)
    
        c_m_s = 3e8
        c_A_s = c_m_s * 1e10
        fnu_conversion_factor = obs_wvln**2/c_A_s
        spec_2D = spec_2D * fnu_conversion_factor
    
        full_phot_dist = {}
        
        for col in columns:
            print(f'Getting {col} Photom Distribution')
            
            wave, tlam = self.read_filt_col(filters, col)
            
            wave_mask = (np.amin(wave) < obs_wvln) & (obs_wvln < np.amax(wave))
            
            in_filter_wave = obs_wvln[wave_mask]
            in_filter_flux = spec_2D.T[wave_mask].T
    
            freq = c_A_s / wave
    
            avg_flux_dist = []
            
            print('Looping over all the posteriors')
            
            for row in in_filter_flux:
                try:
                    interp_spec = Akima1DInterpolator(in_filter_wave, row)
                except ValueError:
                    print(in_filter_wave)
                    print(row)
                num = (1/freq) * interp_spec(wave) * tlam
                denom = (1/freq) * tlam
    
                int_num = trapezoid(num)
                int_denom = trapezoid(denom)
    
                avg_flux = int_num/int_denom
    
                avg_flux_dist.append(avg_flux)
    
            full_phot_dist[col] = np.array(avg_flux_dist)
            print()
            
        return full_phot_dist
    
    def get_posterior_PDF(self, hdu):
    
        '''
        Reading the input file and grabbing the posterior PDF which is in extension 15 as seen below:
    
        # 10  MARGINAL SED    1 ImageHDU         8   (388, 1480)   float32   
         # 11  FULL SED WL    1 BinTableHDU     12   1R x 1C   [13391E]   
         # 12  FULL SED      1 ImageHDU         9   (13391, 1480)   float32   
         # 13  APPARENT MAGNITUDES    1 BinTableHDU     30   1480R x 7C   [1E, 1E, 1E, 1E, 1E, 1E, 1E]   
         # 14  ABSOLUTE MAGNITUDES    1 BinTableHDU     30   1480R x 7C   [1E, 1E, 1E, 1E, 1E, 1E, 1E]   
         # 15  POSTERIOR PDF    1 BinTableHDU     35   1480R x 13C   [1E, 1E, 1E, 1E, 1E, 1E, 1E, 1E, 1E, 1E, 1E, 1E, 1E] 
         
        '''
        
        extension = 'POSTERIOR PDF'
        
        posterior_tab = Table(hdu[extension].data)

        self['posterior'] = posterior_tab
    
        #return posterior_tab
    
    def get_chi2_column(self):
    
        '''
        Grabbing just the chi2 column from a table
         
        '''
        
        col = 'chi_square'
        
        chi2 = self['posterior'][col].value
    
        return chi2
    
    def get_min_chi2_idx(self, table):
        
        '''
        Grabbing the index of the minimum chi 2 and the chi2 value 
         
        '''
        
        chi2 = self.get_chi2_column()
    
        min_chi2_idx = np.argmin(chi2)
    
        return min_chi2_idx, chi2[min_chi2_idx]
    
    
    def get_min_chi2_spectra(self, min_idx, spec2d):
    
        return spec2d[min_idx]
    
    def grab_property(self, table, property):
    
        columns = {  'mass',
                     'nebular_logu',
                     'nebular_z',
                     'tauv_eff',
                     'tau',
                     'agn_logu',
                     'agn_z',
                     'agn_xi',
                     'metallicity',
                     'specific_sfr',
                     'current_sfr_timescale',
                     'formation_redshift'}
    
        if property not in columns:
            data = np.nan*np.ones(1000)
        else:
            
            data = posterior_tab[property].value
    
        return data 
        
    def plot_posteriors(self, data, xlabel, bins = 40):
        
        fsize = 15
        ylabel = 'Counts'
        
        fig, ax = self.generate_plot_setup()
    
        ax.hist(data, bins)
        ax.set_xlabel(xlabel, fontsize = fsize)
        ax.set_ylabel(ylabel, fontsize = fsize)
    
        return fig, ax
    
    def plot_posterior_from_file(self, filename, property, xlabel):
    
        post_tab = get_posterior_PDF(filename)
        data = grab_property(post_tab, property)
    
        fig, ax = plot_posteriors(data, xlabel)
    
        return fig, ax
    
    def get_SFH(self, hdu):
        
        extension = 'STAR FORMATION HISTORIES'
        
        SFH_tab = Table(hdu[extension].data)

        self['SFH'] = SFH_tab
    
        #return SFH_tab
    
    
    def get_age_bins(self):
        
        time = self['SFH']['lookback_age']
    
        return time.value
    
    def get_sfr_vs_time(self):
    
        SFR = self['SFH']['SFR']
    
        return SFR.value
    
    
    def compute_mass_per_time(self, time, sfr):
    
        mass = time*sfr
    
        return mass
    
    def compute_mass_over_range(self, time, mass, start = 0, end = 5e6):
    
        mask = (time > start) & (time < end)
        
        tot_mass = np.sum(mass[mask])
    
        return tot_mass
    
    def compute_mass_during_time_bin(self, filename, start = 0, end = 5e6):
    
        sfh_tab = self.get_SFH(filename)
    
        time = self.get_age_bins(sfh_tab)
        sfr = self.get_sfr_vs_time(sfh_tab)
    
        mass_vs_time = compute_mass_per_time(time, sfr)
    
        mass = []
        for i in range(sfr.shape[0]):
    
            m = compute_mass_over_range(time[i], mass_vs_time[i], start = start, end = end)
            mass.append(m)
    
        return mass
    
    def read_marginal_wvln(self, hdu):
        
        extension = 'MARGINAL SED WL'
        
        wvln = Table(hdu[extension].data)['wl'].value[0]

        self['marg_wvln'] = wvln
        
    
    def read_marginal_SED(self, hdu):
    
        extension = 'MARGINAL SED'
    
        marginal_SED = hdu[extension].data
        self['marg_sed'] = marginal_SED
        
    
    def read_marginal_info(self, hdu):
    
        self.read_marginal_wvln(hdu)
        self.read_marginal_SED(hdu)
    
        #return wvln, marginal_SED

    def read_agn_spectra(self, hdu):

        extension = 'AGN FULL SED'

        agn_spectra = hdu[extension].data

        self['agn_sed'] = agn_spectra
        
    def read_gal_properties(self, hdu):
    
        extension = 'GALAXY PROPERTIES'
        
        gal_prop = Table(hdu[extension].data)
        self['gal_prop'] = gal_prop
        #return gal_prop

    def read_agn_emission(self, hdu):
        
        extension = 'AGN EMISSION'
        
        agn_emission = Table(hdu[extension].data)
        self['agn_emission'] = agn_emission

    def read_hii_emission(self, hdu):
        
        extension = 'HII EMISSION'
        
        hii_emission = Table(hdu[extension].data)
        self['hii_emission'] = hii_emission
        
    def read_SFR_extension(self, hdu):
    
        extension = 'STAR FORMATION'
    
        sfr_tab = Table(hdu[extension].data)    
        self['SFR'] = sfr_tab
        #return sfr_tab
    
    def get_sfr(self, sfr_tab, time = 10):
    
        good_times = {10, 100}
    
        if time not in good_times:
            print('No Valid Time Step for SFR')
            
        sfr = sfr_tab[f'SFR_{time}'].value
    
        return sfr
    
    def get_redshift(self):
        
        gal_prop = self['gal_prop']
        zarr = gal_prop['redshift'].value
    
        z_val = np.median(zarr)

        self['z'] = z_val
    
    def generate_summary_stats_post(self):
    
        post_tab =self['posterior']
    
        summary_dict = {}
        
        for cols in post_tab.colnames:
            
            data = post_tab[cols].value
            data16, data50, data84 = np.percentile(data, q = (16, 50, 84))
    
            summary_dict[f'{cols}_16'] = data16
            summary_dict[f'{cols}_50'] = data50
            summary_dict[f'{cols}_84'] = data84
    
        summary_DF = pd.DataFrame(summary_dict, index = ['Source'])
        
        return summary_DF

    def read_file(self, filename, units = 'Fnu'):
        
        hdu = self.read_beagle_output_file(filename)
        
        self.grab_spectra_info(hdu)
        #self.get_abs_mag_ext(hdu)
        #self.get_app_mag_ext(hdu)
        self.get_posterior_PDF(hdu)
        self.get_SFH(hdu)
        self.read_marginal_wvln(hdu)
        self.read_marginal_SED(hdu)
        self.read_gal_properties(hdu)
        self.read_SFR_extension(hdu)
        self.read_hii_emission(hdu)
        self.get_redshift()
        try:
            self.read_agn_spectra(hdu)
            self.read_agn_emission(hdu)
        
        except:
            pass
        if units == 'Fnu':
            self.convert_full_agn_sed_to_Fnu()
            self.convert_full_sed_to_Fnu()
            self.convert_marginal_sed_to_Fnu()


    def selecting_line_emission(self, line, type = 'SF'):
         
        line_list = {'Al2_1671': 'Al2_1671',
                     'Al3_1855': 'Al3_1855',
                     'Al3_1863': 'Al3_1863',
                     'Al5_29052': 'Al5_29052', 
                     'Al6_36600': 'Al6_36600',
                     'Al6_91160': 'Al6_91160',
                     'Al8_36900': 'Al8_36900',
                     'Al8_58480': 'Al8_58480',
                     'Ar10_5534': 'Ar10_5534',
                     'Ar11_25950': 'Ar11_25950',
                     'Ar2_69800': 'Ar2_69800',
                     'Ar3_218300': 'Ar3_218300',
                     'Ar3_7135': 'Ar3_7135',
                     'Ar3_7325': 'Ar3_7325',
                     'Ar3_7751': 'Ar3_7751',
                     'Ar3_90000': 'Ar3_90000',
                     'Ar4_4740': 'Ar4_4740',
                     'Ar4_7331': 'Ar4_7331',
                     'Ar5_131000': 'Ar5_131000',
                     'Ar5_7005': 'Ar5_7005',
                     'Ar5_80000': 'Ar5_80000',
                     'Ar6_45300': 'Ar6_45300',
                     'C1_9850': 'C1_9850',
                     'C2_1335': 'C2_1335',
                     'C2_1576000': 'C2_1576000',
                     'C2_2326': 'C2_2326',
                     'C3_1176': 'C3_1176',
                     'C3_1907': 'C3_1907',
                     'C3_1910': 'C3_1910',
                     'C3_2297': 'C3_2297',
                     'C3_977':  'C3_977',
                     'C4_1548': 'C4_1548',
                     'C4_1551': 'C4_1551',
                     'C5_3998': 'C5_3998',
                     'Ca2H_3969': 'Ca2H_3969',
                     'Ca2K_3934': 'Ca2K_3934',
                     'Ca2X_8498': 'Ca2X_8498',
                     'Ca2Y_8542': 'Ca2Y_8542',
                     'Ca2Z_8662': 'Ca2Z_8662',
                     'Ca2_3933' : 'Ca2_3933',
                     'Ca2_8579' : 'Ca2_8579',
                     'Ca5_2413': 'Ca5_2413',
                     'Ca5_5309': 'Ca5_5309',
                     'Ca5_6086': 'Ca5_6086',
                     'CaF1_7291': 'CaF1_7291',
                     'CaF2_7324': 'CaF2_7324',
                     'Cl3_3354': 'Cl3_3354',
                     'Cl4_8047': 'Cl4_8047',
                     'Cl9_7334': 'Cl9_7334',
                     'Co11_5168': 'Co11_5168',
                     'Fe10_6375': 'Fe10_6375',
                     'Fe11_7892': 'Fe11_7892',
                     'Fe12_2169': 'Fe12_2169',
                     'Fe14_5303': 'Fe14_5303',
                     'Fe2_1080': 'Fe2_1080',
                     'Fe2_1216': 'Fe2_1216',
                     'Fe2_1500': 'Fe2_1500',
                     'Fe2_1786':'Fe2_1786',
                     'Fe2_2300': 'Fe2_2300',
                     'Fe2_2400': 'Fe2_2400',
                     'Fe2_2500': 'Fe2_2500',
                     'Fe2_4300': 'Fe2_4300',
                     'Fe2_6200': 'Fe2_6200',
                     'Fe2_8900': 'Fe2_8900',
                     'Fe3_1126': 'Fe3_1126',
                     'Fe7_3586': 'Fe7_3586',
                     'Fe7_3759': 'Fe7_3759',
                     'Fe7_4699': 'Fe7_4699',
                     'Fe7_4894': 'Fe7_4894',
                     'Fe7_4943': 'Fe7_4943',
                     'Fe7_4989': 'Fe7_4989',
                     'Fe7_5159': 'Fe7_5159',
                     'Fe7_5277': 'Fe7_5277',
                     'Fe7_5721': 'Fe7_5721',
                     'Fe7_6087': 'Fe7_6087',
                     'H1_9229': 'H1_9229',
                     'H1_9546': 'H1_9546',
                     'Halpha': 'HBaA_6563',
                     'Hbeta':  'HBaB_4861',
                     'Hdelta':'HBaD_4102',
                     'Hgamma': 'HBaG_4340',
                     'Lya': 'HLyA_1216',
                     'LyB': 'HLyB_1026',
                     'paschen_a': 'HPaA_18751',
                     'paschen_b': 'HPaB_12818',
                     'paschen_d': 'HPaD_10049',
                     'paschen_g': 'HPaG_10938',
                     'He1_3889': 'He1_3889',
                     'He1_5876': 'He1_5876',
                     'He2BaB_1215': 'He2BaB_1215',
                     'He2BaG_1085': 'He2BaG_1085',
                     'He2PaB_3203': 'He2PaB_3203',
                     'He2_4686': 'He2_4686',
                     'HeBaA_1640': 'HeBaA_1640',
                     'LyG_973': 'LyG_973',
                     'Mg10_610': 'Mg10_610',
                     'Mg10_625': 'Mg10_625',
                     'Mg2_2796': 'Mg2_2796',
                     'Mg2_2803': 'Mg2_2803',
                     'Mn9_7968': 'Mn9_7968',
                     'N1_5200': 'N1_5200',
                     'N2_2141': 'N2_2141',
                     'N2_5755': 'N2_5755',
                     'N2_6548': 'N2_6548',
                     'N2_6584': 'N2_6584',
                     'N3_1750': 'N3_1750',
                     'N3_678': 'N3_678',
                     'N3_752': 'N3_752',
                     'N3_991': 'N3_991',
                     'N4_1486':  'N4_1486',
                     'N4_1719' : 'N4_1719',
                     'N4_765'  : 'N4_765',
                     'N5_1239' : 'N5_1239',
                     'N5_1243' : 'N5_1243',
                     'Ne3_1815': 'Ne3_1815',
                     'Ne3_3343': 'Ne3_3343',
                     'Ne3_3869': 'Ne3_3869',
                     'Ne4_1602': 'Ne4_1602',
                     'Ne4_2424': 'Ne4_2424',
                     'Ne4_4720': 'Ne4_4720',
                     'Ne5_1141': 'Ne5_1141',
                     'Ne5_1575': 'Ne5_1575',
                     'Ne5_2976': 'Ne5_2976',
                     'Ne5_3426': 'Ne5_3426',
                     'Ne7_895': 'Ne7_895',
                     'Ne8_770': 'Ne8_770',
                     'Ne8_780': 'Ne8_780',
                     'Ni12_4231': 'Ni12_4231',
                     'O1_5577': 'O1_5577',
                     'O1_6300': 'O1_6300',
                     'O1_6363': 'O1_6363',
                     'O1_8446': 'O1_8446',
                     'O2_2471': 'O2_2471',
                     'O2_3726': 'O2_3726',
                     'O2_3727': 'O2_3726',
                     'O2_3729': 'O2_3729',
                     'O2_3730': 'O2_3729',
                     'O2_834': 'O2_834',
                     'O3_1661': 'O3_1661',
                     'O3_1666': 'O3_1666',
                     'O3_2321': 'O3_2321',
                     'O3_4363': 'O3_4363',
                     'O3_4959': 'O3_4959',
                     'O3_5007': 'O3_5007',
                     'O4_1402': 'O4_1402',
                     'O5_1218': 'O5_1218',
                     'O6_1032': 'O6_1032',
                     'O6_1038': 'O6_1038',
                     'P5_1121': 'P5_1121',
                     'S2_1256': 'S2_1256',
                     'S2_4070': 'S2_4070',
                     'S2_4078': 'S2_4078',
                     'S2_6716': 'S2_6716',
                     'S2_6731': 'S2_6731',
                     'S3_1198': 'S3_1198',
                     'S3_1720': 'S3_1720',
                     'S3_3722': 'S3_3722',
                     'S3_6312': 'S3_6312',
                     'S3_9069': 'S3_9069',
                     'S3_9532': 'S3_9532',
                     'S4_1086': 'S4_1086',
                     'S4_1406': 'S4_1406',
                     'S5_1198': 'S5_1198',
                     'S8_9914': 'S8_9914',
                     'Si2_1263': 'Si2_1263',
                     'Si2_1308': 'Si2_1308',
                     'Si2_1531': 'Si2_1531',
                     'Si2_1814': 'Si2_1814',
                     'Si2_2335': 'Si2_2335',
                     'Si3_1207': 'Si3_1207',
                     'Si3_1883': 'Si3_1883',
                     'Si3_1892': 'Si3_1892',
                     'Si4_1394': 'Si4_1394',
                     'Si4_1403': 'Si4_1403',
                     'Si7_2148': 'Si7_2148'}

        if type == 'SF':
            
            column_obs = f'{line_list[line]}_lum_obs'
            column_em = f'{line_list[line]}_lum_em'
            
            obs_lum = self['hii_emission'][column_obs]
            em_lum  = self['hii_emission'][column_obs]

        elif type == 'AGN':

            column_obs = f'{line_list[line]}_lum_obs'
            column_em = f'{line_list[line]}_lum_em'
            
            obs_lum = self['agn_emission'][column_obs]
            em_lum  = self['agn_emission'][column_obs]

        return obs_lum, em_lum

    def getting_full_line_info(self):
        
        agn_line_tab = self['agn_emission']
        sf_line_tab  = self['hii_emission']

        line_cols = sf_line_tab.colnames[6:]

        full_line_info = agn_line_tab[line_cols].to_pandas() + sf_line_tab[line_cols].to_pandas()

        self['full_line_info'] = full_line_info

        self['line_cols'] = line_cols
        
        return full_line_info
        
    def computing_agn_fraction(self):

        agn_seds = self['agn_sed']
        full_seds = self['full_sed']

        agn_ratio = agn_seds/full_seds

        return agn_ratio
        
    
    def compute_difference_in_spectra(self):

        full_spectra = self['full_sed']
        agn_spectra = self['agn_sed']

        difference = full_spectra - agn_spectra
        
        return difference

    def convert_full_sed_to_Fnu(self):
        
        wvln = self['full_sed_wvln']
        spec_2D = self['full_sed']
        
        z = self['z']
        obs_wvln = wvln * (1+z)
        
        c_m_s = 3e8
        c_A_s = c_m_s * 1e10
        
        fnu_conversion_factor = obs_wvln**2/c_A_s
        spec_2D = spec_2D * fnu_conversion_factor
        #spec_flux, spec_fluxerr = self.convert_spectra_to_fnu(spec_wave, spec_flux, spec_fluxerr)

        self['full_sed_Fnu'] = spec_2D

    def convert_marginal_sed_to_Fnu(self):
        
        wvln = self['marg_wvln']
        spec_2D = self['marg_sed']
        
        z = self['z']
        obs_wvln = wvln * (1+z)
        
        c_m_s = 3e8
        c_A_s = c_m_s * 1e10
        
        fnu_conversion_factor = obs_wvln**2/c_A_s
        spec_2D = spec_2D * fnu_conversion_factor

        self['marg_sed_Fnu'] = spec_2D

    def convert_full_agn_sed_to_Fnu(self):
        
        wvln = self['full_sed_wvln']
        spec_2D = self['agn_sed']
        
        z = self['z']
        obs_wvln = wvln * (1+z)
        
        c_m_s = 3e8
        c_A_s = c_m_s * 1e10
        
        fnu_conversion_factor = obs_wvln**2/c_A_s
        spec_2D = spec_2D * fnu_conversion_factor
        #spec_flux, spec_fluxerr = self.convert_spectra_to_fnu(spec_wave, spec_flux, spec_fluxerr)

        self['agn_sed_Fnu'] = spec_2D

    def plot_SEDs(self, type = 'full', unit = 'Fnu'):

        if unit != 'Fnu':
            if type == 'full':
    
                wvln, spec_2D = self['full_sed_wvln'], self['full_sed']
    
            elif type == 'marginal':
                
                wvln, spec_2D = self['marg_wvln'], self['marg_sed']
            
            agn_sed = self['agn_sed']
            
        else:

            if type == 'full':
    
                wvln, spec_2D = self['full_sed_wvln'], self['full_sed_Fnu']
    
            elif type == 'marginal':
                
                wvln, spec_2D = self['marg_wvln'], self['marg_sed_Fnu']
            
            agn_sed = self['agn_sed_Fnu']
        
        z = self['z']
        obs_wvln = wvln * (1+z)

        fig, ax = self.generate_plot_setup()

        spec_2D = spec_2D
        agn_sed = agn_sed

        l16, med, u84 = np.percentile(spec_2D, q = (16, 50, 84), axis = 0)
        l16_agn, med_agn, u84_agn = np.percentile(agn_sed, q = (16, 50, 84), axis = 0)

        difference = spec_2D - agn_sed

        l16_diff, med_diff, u84_diff = np.percentile(difference, q = (16, 50, 84), axis = 0)
        
        ax.fill_between(wvln, y1 = u84, y2 = l16, 
                        color = '#54278f', alpha = 0.5, label = 'Full SED')

        ax.fill_between(wvln, y1 = u84_agn, y2 = l16_agn, 
                        color = '#fb6a4a', alpha = 0.5, label = 'AGN_SED')

        ax.fill_between(wvln, y1 = u84_diff, y2 = l16_diff, 
                        color = '#7bccc4', alpha = 0.7, label = 'Difference SED')

        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel(r'Observed Wavelength [$\AA$]', fontsize = 15)
        
        ax.legend()

        return fig, ax
        
        
    def plot_SFH(self, xlim = (0, 1000), ylim = (0, 500)):
        
        fsize = 15
    
        time = self.get_age_bins()
        sfr = self.get_sfr_vs_time()
    
        l16_sfr, u84_sfr = self.get_bounds_from_spec(sfr)
    
        post_tab = self['posterior']
        min_chi2_idx, chi2_val = self.get_min_chi2_idx(post_tab)

        Myr_conversion = 1e6
    
        fig, ax = self.generate_plot_setup()
        
        ax.fill_between(time[0]/Myr_conversion, 
                        y1 = u84_sfr, y2 = l16_sfr, 
                        color = cobaltblue)
        
        ax.plot(time[0]/Myr_conversion, sfr[min_chi2_idx], 
                color = 'black', 
                label = fr'Minimum $\chi^2$: {chi2_val:.2f}')
        
        ax.set_xlabel('Time [Myr]', 
                      fontsize = fsize)
        
        ax.set_ylabel(r'SFR [M$_{\odot}$ yr$^{-1}$]', 
                      fontsize = fsize)
    
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        
        ax.legend()
        
        return fig, ax
    
    def plotting_model_spectra_w_actual_spectra(self, spec_wave, spec_flux, spec_fluxerr, units = 'Flambda',
                                                wavelength_unit = 'Angstrom', 
                                               fontsize = 15):
        
        wvln = self['marg_wvln']
        if units != 'Fnu':
            spec_2D = self['marg_sed']
        
        else:
            spec_2D = self['marg_sed_wvln'], self['marg_sed_Fnu']
        
        z = self['z']
        obs_wvln = wvln
    
        #converting the input spectrum into Fnu
        if units == 'Fnu':
            
            c_m_s = 3e8
            c_A_s = c_m_s * 1e10
            fnu_conversion_factor = obs_wvln**2/c_A_s
            spec_flux, spec_fluxerr = self.convert_spectra_to_fnu(spec_wave, spec_flux, spec_fluxerr)
    
        
        reduced_wvln = obs_wvln
        xlabel = r'Wavelength [$\AA$]'
        l16, med, u84 = np.percentile(spec_2D, q = (16, 50, 84), axis = 0)

        if wavelength_unit == 'microns':
            
            reduced_wvln = reduced_wvln * u.Angstrom.to(u.micrometer)
            spec_wave = spec_wave * u.Angstrom.to(u.micrometer)
            xlabel = r'Wavelength [$\mu m$]'

        elif wavelength_unit == 'nanometer':
            
            reduced_wvln = reduced_wvln * u.Angstrom.to(u.nanometer)
            spec_wave = spec_wave * u.Angstrom.to(u.nanometer)
            xlabel = r'Wavelength [nm]'
            
        fig, ax = self.generate_plot_setup()
        
        ax.step(spec_wave, spec_flux, 
                where = 'mid', color = 'red', label = 'Spectra', alpha = 0.5)
        
        ax.errorbar(spec_wave, spec_flux, 
                    yerr = spec_fluxerr, 
                    fmt = 'none', capsize=1, elinewidth = 0.5, 
                    ecolor = "black")
    
        ax.plot(reduced_wvln, med, color = '#0504aa', lw = 1, label = 'Modeled Spectra')
        ax.fill_between(reduced_wvln, u84, l16, color = 'dodgerblue', step = 'mid')
        
        ax.set_xlabel(xlabel, fontsize = fontsize)
        ax.legend()
        
        return fig, ax 

    def read_input_photom(self, filename, extension = 1):
        
        hdu = fits.open(filename)
        
        photom_tab = Table(hdu[extension].data)
        
        self['Input_Photom'] = photom_tab

    def get_flux_flux_errors(self, units):
        
        photom_tab = self['Input_Photom'] #.to(u.erg/u.s/u.cm**2/u.Hz)

        flux_cols = [x for x in photom_tab.colnames if 'FLUX_F' in x]
        flux_err_cols = [x for x in photom_tab.colnames if 'FLUXERR_F' in x]

        photom_df = photom_tab.to_pandas()
        
        conversion = 1
        
        if units == 'microJy':
            
            conversion = u.microJansky.to(u.erg/u.s/u.cm**2/u.Hz)
       
        elif units == 'nanoJy':
            
            conversion = u.nanoJansky.to(u.erg/u.s/u.cm**2/u.Hz)
            
        return photom_df[flux_cols].values*conversion, photom_df[flux_err_cols].values*conversion

    def convolve_model_spec(self, r_wave, r_curve, model_wave, model_flux, wave_obs_spectra, oversample = 10, f_LSF = 1):

        '''
        #oversample = define some over-sampling value, I usually use 10 
        #f_LSF = fudge factor increasing the resolution by a constant factor, based on e.g. the position in the slit 
                 (I usually use ~1.3 or so for point sources)
        
        '''
        
        # Construct wavelength grid (x is an array of wavelengths with equal separations in R space)
        print('generating grid')
        wav_obs_fine = [0.95*model_wave[0]]
        while wav_obs_fine[-1] < 1.05*model_wave[-1]:
            R_val = np.interp(wav_obs_fine[-1], r_wave, r_curve)
            dwav = wav_obs_fine[-1]/np.abs(R_val)/oversample
            wav_obs_fine.append(wav_obs_fine[-1] + dwav)
        print('done_generating grid')
        wav_obs_fine = np.array(wav_obs_fine)

        print('interpolating model onto grid')
        # Construct your model on the grid of wavelengths you defined above 
        model = interp1d(model_wave, model_flux, 
                         bounds_error=False, fill_value='extrapolate')
        
        flux_fine = model(wav_obs_fine)

        plt.figure()
        plt.plot(wav_obs_fine, flux_fine)
        plt.show()
        
        # Convolve with the resolution curve
        sigma_pix = oversample/2.35/f_LSF  # sigma width of kernel in pixels
        k_size = 5*int(sigma_pix+1)
        x_kernel_pix = np.arange(-k_size, k_size+1) 
        kernel = np.exp(-(x_kernel_pix**2)/(2*sigma_pix**2)) # construct Gaussian kernel
        kernel /= np.trapz(kernel)  # Explicitly normalise kernel
        print('performing the convolution')
        flux_fine = convolve(flux_fine, kernel)
        
        # Downsample to the wavelength grid of the instrument (here I used Adam Carnall's spectres package) 
        #need to map onto actual observed spectral data

        print('Downsampling')
        flux = spectres.spectres(wave_obs_spectra, wav_obs_fine, flux_fine, fill=0, verbose=False)

        return flux


    def generate_convolved_spectra(self, r_wave, r_curve, obs_wave):
        
        #r_wave, r_curve, model_wave, model_flux, wave_obs_spectra
        redshift = self['gal_prop']['redshift']
        
        

        #idx_in_range = np.where(((r_wave[0] < wav_obs) & (wav_obs < r_wave[-1])))[0]
        
        #in_range_wave = wav_obs[idx_in_range]
        
        in_range_flux = self['full_sed']
        in_range_agn  = self['agn_sed']

        convolved_flux = np.zeros((len(in_range_flux), len(obs_wave)))
        convolved_agn = np.zeros((len(in_range_flux), len(obs_wave)))
        
        for i in range(convolved_flux.shape[0]):
            
            z = redshift[i]
            model_wav = self['full_sed_wvln'] * (1+z)
            model_flux = in_range_flux[i]
            model_agn  = in_range_agn[i]
            
            convolved_flux[i] = self.convolve_model_spec(r_wave, r_curve, model_wav, model_flux, obs_wave)
            convolved_agn[i] = self.convolve_model_spec(r_wave, r_curve, model_wav, model_agn, obs_wave)

        self['convolved_spectra']   = convolved_flux
        self['convolved_agn']       = convolved_agn
        self['convolved_wave']      = obs_wave
        self['agn_ratio_convolved'] = convolved_agn/convolved_flux


    def plot_convolved_spectra_with_actual_spectra(self, spec_wave, spec_flux, spec_err, ax = False):
        z = self['z']
        wave = self['convolved_wave']
        l16, flux, u84 = np.percentile(self['convolved_spectra'], q = (15, 50, 84), axis = 0)/(1+z)
        
        if ax == False:
            
            fig, ax = plt.subplots(figsize = (10, 5), constrained_layout = True)
            ax.step(wave, flux, where = 'mid', color = 'purple', label = 'Convolved Model')
            ax.step(spec_wave, spec_flux, color = 'gray', label = 'Data')
            ax.errorbar(spec_wave, spec_flux, yerr = spec_err, 
                        color = 'red', fmt = 'none', alpha = 0.5)
            
            ax.fill_between(wave, y1 = u84, y2 = l16, color = 'dodgerblue')
            
            ax.set_xlabel("Wavelength", fontsize = 15)
            ax.set_ylabel('Flux', fontsize = 15)
            ax.legend()
            return fig, ax
        
        else:
            
            ax.step(wave, flux, where = 'mid', color = 'blue', label = 'Convolved Model')
            ax.step(spec_wave, spec_flux, color = 'black', label = 'Data')
            ax.errorbar(spec_wave, spec_flux, yerr = spec_err, color = 'red')
            ax.set_xlabel("Wavelength", fontsize = 15)
            ax.set_ylabel('Flux', fontsize = 15)
            #ax.axvline()
            ax.legend()
        
    def plot_agn_ratio_convolved_per_line(self, input_line, window_width, ax = False):
        
        z = self['z']
        line = input_line * (1+z)
        width =  window_width * (1+z)
        
        obs_wave = self['convolved_wave']
        convolved_ratio = self['agn_ratio_convolved']
        
        idx = np.where((line - width < obs_wave) & (obs_wave < line + width))[0]
        in_range_wave = obs_wave[idx]
        in_range_flux = convolved_ratio[:, idx]

        l16, med, u84 = np.percentile(in_range_flux, q = (16, 50, 84), axis = 0)
        
        if ax == False:
            
            fig, ax = plt.subplots(figsize = (10, 5), constrained_layout = True)
            ax.step(in_range_wave/(1+z), med, where = 'mid', color = 'black')
            ax.fill_between(in_range_wave/(1+z), y1 = u84, y2 = l16, color = 'dodgerblue', step='mid')
            ax.set_xlabel("Rest Wavelength", fontsize = 15)
            ax.set_ylabel('AGN Fraction', fontsize = 15)
            
            return fig, ax
        
        else:
            
            ax.step(in_range_wave/(1+z), med, where = 'mid', color = 'black')
            ax.fill_between(in_range_wave/(1+z), y1 = u84, y2 = l16, color = 'dodgerblue', step = 'mid')
            ax.set_xlabel("Rest Wavelength", fontsize = 15)
            ax.set_ylabel('AGN Fraction', fontsize = 15)
        
        
    def plot_full_spec_posteriors(self, input_photom_file, filename_filters, extension = 1, 
                                  photom_units = 'nanoJy', units = 'Fnu', xlim = (1e3, 1e5)):

        self.read_input_photom(input_photom_file, extension = extension)
        
        input_flux, input_flux_err = self.get_flux_flux_errors(units = photom_units)
        
        ylabel = r'F$_{\lambda}$ erg s$^{-1}$ cm$^{-2}$ $\AA^{-1}$'
        fsize = 15
        
        self.compute_avg_wave_per_filter(filename_filters)
        
        pivot_waves = self['pivot_wvln']
        
        z = self['z']
        
        post_tab = self['posterior']
        min_chi2_idx, chi2_val = self.get_min_chi2_idx(post_tab)
        
        wvln, spec_2D = self['full_sed_wvln'], self['full_sed'] 
    
        obs_wvln = wvln*(1+z)
    
        if units == 'Fnu':
    
            # c_m_s = 3e8
            # c_A_s = c_m_s * 1e10
            # fnu_conversion_factor = obs_wvln**2/c_A_s
            # spec_2D = spec_2D * fnu_conversion_factor
            ylabel = r'F$_{\nu}$ erg s$^{-1}$ cm$^{-2}$ $Hz^{-1}$'
            photom_dist = self.get_Fnu_dist_app_mag_fit()
    
        spec_2D = spec_2D/(1+z)
        
        l16_spec, u84_spec = self.get_bounds_from_spec(spec_2D)
        min_chi2_spec = self.get_min_chi2_spectra(min_chi2_idx, spec_2D)
    
        fig, ax = self.generate_plot_setup()
        
        ax.fill_between(x = obs_wvln, y1 = u84_spec, y2 = l16_spec, 
                        color = cobaltblue, alpha = 0.7)
        
        ax.plot(obs_wvln, min_chi2_spec, 
                color = 'black', alpha = 0.7, lw = 0.5, 
                label = fr'$\chi^2$: {chi2_val:.2f}')
    
        l16_photom, med_photom, u84_photom = np.percentile(photom_dist.values, q = (16, 50, 84), axis = 0)
        
        ax.scatter(pivot_waves, med_photom, facecolors='none', edgecolors='blue', label='Median Modeled Photometry')
        
        ax.errorbar(pivot_waves, input_flux[0], yerr = input_flux_err[0], 
                    fmt = 'o', fillstyle = 'none', markeredgecolor = 'red', 
                    ecolor = 'black', elinewidth = 1, linestyle = '--', 
                    capsize = 2.5, label = 'Input Photometry')
            
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel(r'Wavelength [$\AA$]', fontsize = fsize)
        ax.set_ylabel(ylabel, fontsize = fsize)
        ax.set_xlim(xlim)
        ax.legend()
        
        return fig, ax


    