import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
from astropy.io import fits
from astropy.table import Table
import emcee
from scipy.interpolate import Akima1DInterpolator, interp1d
from scipy.optimize import curve_fit
import seaborn as sb
from scipy.ndimage import gaussian_filter1d
from astropy.stats import sigma_clip

plt.rcParams['axes.linewidth'] = 1.5
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.size"] = 15

def gaussian(x, A, mu, sigma):
    
    '''
    Gaussian Model for Line Fitting. This is of the form:
    
    Gaussian = Ae^(-(x-mu)^2/sigma^2)
    
    Input
    -------------
    x: array or single value to evaluate the Gaussian 
    A: amplitude of the Gaussian
    mu: Center of the Gaussian
    sigma: the standard deviation of the Gaussian
    
    
    Returns
    --------------
    Evaluated Gaussian for the given A, mu and sigma at the point(s) in x
    
    '''
    
    return A * np.exp(-(x - mu)**2/ (sigma**2))

def line(x, m, b, line_center):
    
    '''
    Continuum of the spectra using y = b
    
    Input
    ------------
    x: array of values
    b: value to plot y = b
    
    
    Returns
    ------------
    An array of values at b, the same size as x
    
    '''
    
    return  m*(x - line_center) + b

def line_model(x, A, mu, sigma, m, b, line_center):
    
    '''
    Emission Line model using Gaussian and the continuum
    
    Inputs
    ------------

    x: array of values to evaluate the Gaussian 
    A: amplitude of the Gaussian
    mu: Center of the Gaussian
    sigma: the standard deviation of the Gaussian
    b: value to plot y = b
    '''
    
    
    return gaussian(x, A, mu, sigma) + line(x, m, b, line_center)

def double_gaussian(x, A1, mu1, sigma, m, b, line_center1, line_center2):

    ratio = line_center2/line_center1

    return gaussian(x, A1, mu1, sigma) +  gaussian(x, A2, mu1*ratio, sigma) + line(x, m, b, line_center)

def log_likelihood(theta, x, y, yerr, line_center):
    '''
    This is the likelihood function we are using for emcee to run
    
    This likelihood function is the maximum likelihood assuming gaussian errors.
    
    '''
    ################
    
    # The value we are trying to fit
    #A, mu, sigma, m, b = theta
    
    #Making the model of the emission line
    model = line_model(x, *theta, line_center)
    
    #getting the log likelihood, this is similar to chi2 = sum((data - model)^2/sigma^2)
    lnL = -0.5 * np.sum((y - model) ** 2 / yerr**2)
    
    return lnL


def log_prior(theta, wave_center, Amp_max, m, b, max_b, min_b):
    '''
    The prior function to be used against the parameters to impose certain criteria for the fitting making 
    sure that they do not go and explore weird values
    
    '''

    scale = 10
    
    #Theta values that goes into our Gaussian Model
    A, mu, sigma, m, b = theta
    
    #the left most bound and right most bound that the central wavelength can vary
    left_mu = wave_center - 250  # this is how much mu can vary
    right_mu = wave_center + 250 # this is how much mu can vary
    
    #min and max amplitude of the emission line
    min_A = 0
    max_A = Amp_max * 3
    
    sigma_window_left = .01 #had to change these for the input spectra these are left bounds for sigma
    sigma_window_right = 500 #had to change these for the input spectra these are right bounds for sigma

    #power_m = np.log10(np.abs(m))
    #power_b = np.log10(np.abs(b))

    if m < 0:
        low_bounds_m = m * scale
        high_bounds_m = m/scale
    else:
        low_bounds_m = m/scale
        high_bounds_m = m*scale
        
    
    low_bounds_b = min_b
    high_bounds_b = max_b
        
    m_bounds = (low_bounds_m < m) & (m < high_bounds_m)  
    b_bounds = (low_bounds_b < b) & (b < high_bounds_b) 
        
    if (0 < A < max_A) & (left_mu <= mu <= right_mu) & (sigma_window_left <= sigma < sigma_window_right) & (m_bounds) & (b_bounds):
        return 0.0
    else:
        return -np.inf

def log_probability(theta, x, y, yerr, center, Amp_max, m, b, line_center, max_b, min_b):
    
    lp = log_prior(theta, center, Amp_max, m, b, max_b, min_b)
    if not np.isfinite(lp):
        #print('Probability is infinite')
        return -np.inf
    prob = lp + log_likelihood(theta, x, y, yerr, line_center)
    #print(f'Prob:{prob:.3E}')
    return prob


def initial_fits(wave, spectrum, err_spec,  line_window, fit_window, line_center, diagnose = False):
    
    '''
    This function does an initial fit on the data using curve fit which we then pass in those parameters into emcee
    to do the full MCMC fit later
    
    Inputs
    -------------
    wave: Wavelength Array
    spectrum: Full spectrum array
    err_spec: the Error spectra
    window: the window to look around an emission line in units of the wavelength array
    line_center: The line center of the emission line
    
    Returns:
    
    result: The output of the intial curve fit which would be an array with output in order of the parameters
            in the model np.array([A, mu, sigma, b])
    
    '''
    
    
    #the range where curve_fit will look between 
    min_window = line_center - fit_window #in units of Angstroms
    max_window = line_center + fit_window #in units of Angstroms
    
    #getting emission line near the line center
    #line_center +/- window
    indx = np.where((min_window < wave) & ((wave < max_window)))[0]

    #this grabs the full input curve fit spectra
    spec_window = spectrum[indx]
    wave_window = wave[indx]
    err_spec_window = err_spec[indx]

    smoothed_spectrum = gaussian_filter1d(spec_window, sigma=2)
    sigma_arr = wave_window/interp_R(wave_window)
    sigma_low = np.amin(sigma_arr)

    if diagnose:
        plt.figure(figsize = (10, 5), constrained_layout = True)
        plt.title('Input Curve Fit Spectra')
        plt.step(wave_window, spec_window, color = 'black')
        plt.step(wave_window, smoothed_spectrum, color = 'red')
        plt.errorbar(wave_window, spec_window, yerr = err_spec_window, color = 'gray', fmt = 'none')
        
        plt.show()

    
    #initial guesses for the optimization
    guess_A = np.amax(spectrum[indx])
    guess_mu = line_center
    
    #We interpolate the spectrum near the emission line, we do this to get an estimate on sigma by computing 
    #the full width at half-maximum
    # spec_interp = Akima1DInterpolator(wave_window, spec_window)
    
    # #making a wavelength array near the emission line
    # x = np.linspace(wave_window[0], wave_window[-1], 10000)
    
    # #applying the wavelength array to the interpolated function
    # spec = spec_interp(x)
    
    # #getting the value at half maximum
    # half_max = np.amax(spec)/2
    
    # #finding index where the spectrum is higher than the half-maximum value
    # #the first and last indexes are the wavelength where the sigma can be computed
    # idx = np.where(spec > half_max)[0]
    
    # #getting the left and right most wavelengths
    # wave_left, wave_right = x[idx[0]], x[idx[-1]]

    # if diagnose:
    #     plt.figure(figsize = (10, 5))
    #     plt.step(x, spec, where = 'mid', color = 'red')
    #     plt.scatter([wave_left, wave_right], [spec_interp(wave_left), spec_interp(wave_right)], color = 'black')
    #     plt.axhline(half_max+np.median(spec))
    #     plt.show()
    
    #taking the difference between the right and left wavelength and divide it by 2 to get a guess for the sigma
    guess_sigma = np.median(sigma_arr)

    min_idx = np.argmin(np.abs(line_center - wave_window))

    y_cont = np.concatenate((np.array(spec_window[min_idx - 5:min_idx - 2]), np.array(spec_window[min_idx + 2:min_idx + 5])))
    x_cont = np.concatenate((np.array(wave_window[min_idx - 5:min_idx - 2]), np.array(wave_window[min_idx + 2:min_idx + 5])))
    
    #print(x_cont)
    #print(y_cont)
    
    if diagnose:
        
        #xarr = np.linspace(x[0], x[-1], 100)
        plt.figure()
        plt.title("Continuum Data")
        plt.scatter(x_cont, y_cont, label = 'Data', zorder = 99, color = 'black')
    
    
    m_guess, b_guess = np.polyfit(x_cont, y_cont, 1)
    fit = np.poly1d([m_guess, b_guess])
    
    if diagnose:
        
        xarr = np.linspace(x_cont[0], x_cont[-1], 100)
        plt.figure()
        plt.title("Continuum Fit")
        plt.scatter(x_cont, y_cont, label = 'Data', zorder = 99, color = 'black')
        #fit = np.poly1d([m_guess, b_guess])
        
        plt.plot(xarr, fit(xarr), label = 'Model Fit', color = 'red')
        plt.legend()
        plt.show()
    
    if diagnose == True:
        
        print('Minimization Guesses')
        print(f"A: {guess_A}")
        print(f"mu: {guess_mu}")
        print(f"sigma: {guess_sigma}")
        print(f'm: {m_guess}')
        print(f"b: {b_guess}")
        print(f'Median: {np.median(spec_window)}')
        print() 

    #making initial guesses
    x0 = [guess_A - fit(guess_mu), guess_mu, guess_sigma, m_guess, b_guess]#np.median(spec_window)]

    if diagnose:
        xarr = np.linspace(wave_window[0], wave_window[-1], 1000)
        plt.figure()
        plt.plot(xarr, line_model(xarr, *x0), label = 'Intitial Guess Model')
        plt.step(wave_window, spec_window, where = 'mid')
        plt.show()

    #making lower and upper bounds to use into curve_fit
    if b_guess < 0:
        low_bounds = [0, min_window, sigma_low, -np.inf, -np.inf]
        high_bounds = [2*guess_A, max_window, 500, np.inf, np.inf]
    else:
        low_bounds = [0, min_window, sigma_low, -np.inf, -np.inf]
        high_bounds = [2*guess_A, max_window, 500, np.inf, np.inf]

    if diagnose:
        print('Intitial Guesses for the parameters are: ')
        print(x0)

        print('Curve Fit Lower Bounds are: ')
        print(low_bounds)
        print()
        print('Curve Fit Upper Bounds are: ')
        print(high_bounds)

    
    # Optimization of the initial gaussian fit
    result,_ = curve_fit(line_model, wave_window, spec_window, p0 = x0, 
                          bounds = [low_bounds, high_bounds], absolute_sigma = True, sigma = err_spec_window)                 
    
    
    ########
    # Diagnostic Plotting: making sure we are getting the emission line
    ########
    if diagnose == True:
        
        print('Curve Fit Results')
        print(f"A: {result[0]}")
        print(f"mu: {result[1]}")
        print(f"sigma: {result[2]}")
        print(f"m: {result[3]}")
        print(f"b: {result[4]}")
        print()
        
        xarr = np.linspace(wave_window[0], wave_window[-1], 100)
        
        plt.figure()
        plt.step(wave_window, spec_window, color = 'blue', where = 'mid')
        plt.plot(xarr, line_model(xarr, *result), color = 'black', label = 'Model')
        plt.axhline(0, linestyle = '--')
        plt.ylabel('Flux')
        plt.xlabel(r'Wavelength $\mu$m')
        plt.title('Initial curve_fit Fitting')
        plt.legend()
        plt.show()

        plt.figure()
        #plt.plot(wave_window, spec_window, color = 'blue', label = 'Data')
        plt.step(wave_window, spec_window, color = 'blue', where = 'mid')
        plt.plot(xarr, line(xarr, result[3], result[4]), label = 'Line', color = 'red')
        plt.plot(xarr, gaussian(xarr, *result[:3]), label = 'Gaussian', color = 'black')
        
        #plt.plot(xarr, line_model(xarr, *result), color = 'black', label = 'Model')
        #plt.axhline(0, linestyle = '--')
        plt.ylabel('Flux')
        plt.xlabel(r'Wavelength $\mu$m')
        plt.title('Separate Components')
        plt.legend()
        plt.show()
    
    
    return result

def initial_fits_user_input(wave, spectrum, err_spec):
    
    '''
    This function does an initial fit on the data using curve fit which we then pass in those parameters into emcee
    to do the full MCMC fit later
    
    Inputs
    -------------
    wave: Wavelength Array
    spectrum: Full spectrum array
    err_spec: the Error spectra
    
    Returns:
    
    result: An array with the output in order of the parameters
            in the model np.array([A, mu, sigma, m, b])
    
    '''

    line_center = float(input('Enter the Line Center you want to fit: '))
    print(f'Recieved {line_center} to fit')

    window = float(input('Enter the spectral window you want to fit around: '))
    
    #the range where curve_fit will look between 
    min_window = line_center - window #in units of Angstroms
    max_window = line_center + window #in units of Angstroms

    print(f'Looking Between {min_window:.3f} -- {max_window:.3f}')
    
    indx = np.where((min_window < wave) & ((wave < max_window)))[0]

    print('Extracting the spectra in the fit window')
    
    #this grabs the full input curve fit spectra
    spec_window = spectrum[indx]
    wave_window = wave[indx]
    err_spec_window = err_spec[indx]
    
    
    plt.figure(figsize = (10, 5), constrained_layout = True)
    plt.title('Input Spectra within Window')
    plt.step(wave_window, spec_window, color = 'black', where = 'mid')
    plt.errorbar(wave_window, spec_window, yerr = err_spec_window, color = 'gray', fmt = 'none')
    plt.show()

    #asking user for best guess Amplitude
    guess_A = float(input('Enter the guess for the amplitude for the fit (Peak flux of the line is preferred): '))
     
    guess_mu = line_center

    #asking user for best guess sigma
    guess_sigma = float(input('Enter the guess for sigma: '))

    #Getting the user defined continuum window
    cont_window_est = input('Enter the bounds to compute the continuum [ex: cont_left_min,cont_left_max,cont_right_min,cont_right_max]: ')

    #this piece of code converts the strings we get back from the input into floats we can use in our functions
    cont_left_min, cont_left_max, cont_right_min, cont_right_max = [float(x) for x in cont_window_est.split(',')]

    #This gets the index in the full spectrum that are within the bounds provided by the user
    cont_idx = np.where(((wave >= cont_left_min) & (wave <= cont_left_max)) | ((wave >= cont_right_min) & (wave <= cont_right_max)))[0]

    #gets the x and y value for the continuum
    y_cont = np.array(spectrum[cont_idx])
    x_cont = np.array(wave[cont_idx])

    #fitting the values with a line
    m_guess, b_guess = np.polyfit(x_cont - line_center, y_cont, 1)

    #making the continuum model
    fit = np.poly1d([m_guess, b_guess])

    #generating x-array for finer plotting 
    xarr = np.linspace(x_cont[0], x_cont[-1], 100)
    
    plt.figure()
    plt.title("Continuum Fit")
    plt.scatter(x_cont, y_cont, label = 'Data', zorder = 99, color = 'black')
    plt.plot(xarr, fit(xarr - line_center), label = 'Model Fit', color = 'red')
    plt.legend()
    plt.show()

    #making initial guesses
    x0A = guess_A - fit(guess_mu - line_center) #we subtract the continuum from the peak of the gaussian to get its actual amplitude
    
    x0 = [x0A, guess_mu, guess_sigma, m_guess, b_guess]
    
    print('Curve_Fit Guesses')
    print(f"A: {x0A:.2e}")
    print(f"mu: {guess_mu:.3f}")
    print(f"sigma: {guess_sigma:.4f}")
    print(f'm: {m_guess:.2e}')
    print(f"b: {b_guess:.2e}")
    print() 

    
    xarr = np.linspace(wave_window[0], wave_window[-1], 1000)
    plt.figure()
    plt.title('Model Using Initial Guesses')
    plt.plot(xarr, line_model(xarr, *x0, line_center), label = 'Intitial Guess Model')
    plt.step(wave_window, spec_window, where = 'mid')
    plt.show()    
    
    return x0, window

def user_input(wave, flux, flux_err):

    '''
    Function that fits emcee on a spectrum with inputs:
    
    wave: Wavelength Array (Units: angstroms)
    flux: Flux Array (units: erg/s/cm^2/Angstroms)
    fluxerr: Flux Error Array (Units: erg/s/cm^2/Angstroms)

    returns
    emcee_df: DataFrame holding the parameter values for line_model along side LnL and Fluxes

    #NOTE The Fluxes are assumed to be in erg/s/cm^2
    
    '''

    #initializing window for emcee around the best result mu
    result, window = initial_fits_user_input(wave, flux, flux_err)
    guess_A = result[0]
    guess_mu = result[1]
    guess_sigma = result[2]
    guess_m = result[3]
    guess_b = result[4]
    
    emcee_window = window

    
    
    #getting indexes near the emission line based off of the emcee_window
    #looking at line_center +/- emcee_window
    emcee_indx = np.where((wave >= (guess_mu - emcee_window)) & 
                          (wave <= (guess_mu + emcee_window)))[0] 

    #emcee subsections
    emcee_spec = flux[emcee_indx]
    emcee_wave = wave[emcee_indx]
    emcee_err = flux_err[emcee_indx]
    
    
    #making walkers so that we can use emcee to explore the parameter space
    #centered on the best results from minimization
    amp_jump = np.random.normal(loc = guess_A,            #centered on best A from curve_fit
                                scale = guess_A/2,       #can wander 1/2 of the value of A
                                size = 32).reshape(-1, 1) 

    ###################
    #NOTE: The scale of the wavelengths here were used for a spectrum that was in angstroms, for spectrum in Microns you will need to  
    #      Change the scale numbers to have the appropriate jump scale for your data
    ###################
    wavelength_jump = np.random.normal(loc = guess_mu,    #centered on best mu from curve_fit
                                       scale = 20,      #can wander +/- 20 microns Angstroms
                                       size = 32).reshape(-1, 1)#data so if you are working with other spectra 
                                                                #you may need ot update this)
    
    sigma_jump = np.random.normal(loc = guess_sigma,       #centered on best sigma from curve_fit
                                  scale = 20,            #can wander +/- 20 Angstroms
                                  size = 32).reshape(-1, 1)

    powerm = np.log10(np.abs(guess_m))
    
    m_jump = np.random.normal(loc = guess_m,       #centered on best guess m from input
                                  scale = 1*10**powerm,        
                                  size = 32).reshape(-1, 1)

    
    #getting the power of 10 that the linear fit is
    
    

    if (guess_b < np.amin(emcee_spec)) | (guess_b > np.amax(emcee_spec)):
        l16_spec = np.percentile(emcee_spec, q = 16)
        powerb = np.log10(np.abs(l16_spec))
        b_jump = np.random.normal(loc = l16_spec,           #centered on best b from user input
                                  scale = 1*10**powerb,    #making it wander 10^powerb (if b = .05, it can wander .01)
                                  size = 32).reshape(-1, 1)
    else:
        powerb = np.log10(np.abs(guess_b))
        b_jump = np.random.normal(loc = guess_b,           #centered on best b from user input
                                  scale = 1*10**powerb,    #making it wander 10^powerb (if b = .05, it can wander .01)
                                  size = 32).reshape(-1, 1)

    
    #################
    # Diagnostic plotting to see if the parameters were jumping to large values
    # The should be concentrated near their best fit results values
    #################
    
    print('Checking the Walker Jumps')
    fig, ax = plt.subplots(figsize = (10, 14), nrows = 3, ncols = 2, constrained_layout = True)
    
    ax[0, 0].hist(amp_jump)
    ax[0, 0].set_xlabel('Amplitude')
    
    ax[0, 1].hist(wavelength_jump)
    ax[0, 1].set_xlabel(r'$\mu$')
    
    ax[1, 0].hist(sigma_jump)
    ax[1, 0].set_xlabel(r'$\sigma$')

    ax[1, 1].hist(m_jump)
    ax[1, 1].set_xlabel(r'm')
    
    ax[2, 0].hist(b_jump)
    ax[2, 0].set_xlabel('b')
    
    plt.show()
    

    #stacking along the columns and generating the starting walkers
    starting_walkers = np.hstack((amp_jump,
                                  wavelength_jump, 
                                  sigma_jump,
                                  m_jump,
                                  b_jump))

    #plotting the input emcee spectrum
    plt.figure(figsize = (10, 5), constrained_layout = True)
    plt.title('Input Emcee Spectra')
    plt.step(emcee_wave, emcee_spec, color = 'black', where = 'mid')
    plt.errorbar(emcee_wave, emcee_spec, yerr = emcee_err, color = 'gray', fmt= 'none')
    plt.show()

    
    #initializing walker positions
    pos = starting_walkers
    nwalkers, ndim = pos.shape

    #initializing sampler
    sampler = emcee.EnsembleSampler(nwalkers, #giving emcee the walker positions
                                    ndim,     #giving it the dimension of the model(same as number of model parameters)
                                    log_probability, #giving it the log_probability function
                                    args=(emcee_wave, emcee_spec, emcee_err, guess_mu, guess_A, 
                                          guess_m, guess_b, guess_mu, 
                                          np.amax(emcee_spec), np.amin(emcee_spec)), #arguments to pass into log_probability
                                    #moves = [(emcee.moves.DEMove(), 0.5),        
                                    #         (emcee.moves.DESnookerMove(), 0.5)]
                                   )

    #running emcee
    state = sampler.run_mcmc(pos, 1000)
    sampler.reset()
    sampler.run_mcmc(state, 3000, progress=False)

    #getting values back
    flat_samples = sampler.get_chain(flat=True)
    LnL_chain = sampler.flatlnprobability
    
    emcee_df = pd.DataFrame()
    emcee_df['A'] = flat_samples[:, 0]
    emcee_df['mu'] = flat_samples[:, 1]
    emcee_df['sigma'] = flat_samples[:, 2]
    emcee_df['m'] = flat_samples[:, 3]
    emcee_df['b'] = flat_samples[:, 4]
    emcee_df['LnL'] = LnL_chain[:]

    #return emcee_df
    
    #removing values where the log_likelihood was infinite as these are bad fits
    emcee_df = emcee_df[np.isfinite(emcee_df.LnL.values)]
    
    #getting the flux from the parameter values
    fluxes_emcee = emcee_df['A'] * emcee_df['sigma'] * np.sqrt(2 * np.pi)
    
    emcee_df['Fluxes'] = fluxes_emcee

    print('Checking Prameter Posterior Distributions')
    fig, ax = plt.subplots(figsize = (10, 12), nrows = 3, ncols = 2, constrained_layout = True)
    
    emcee_df.A.hist(ax = ax[0, 0])
    emcee_df.mu.hist(ax = ax[0, 1])
    emcee_df.sigma.hist(ax = ax[1, 0])
    emcee_df.m.hist(ax = ax[1, 1])
    emcee_df.b.hist(ax = ax[2, 0])

    ax = ax.flatten()
    ax[0].set_xlabel('Ampitude')
    ax[1].set_xlabel(r'$\mu$')
    ax[2].set_xlabel(r'$\sigma$')
    ax[3].set_xlabel('m')
    ax[4].set_xlabel('b')
    
    plt.show()

    xarr = np.linspace(emcee_wave[0], emcee_wave[-1], 100)

    line_center = guess_mu
    
    plt.figure()
    plt.title('Input Emcee Spectra and Emcee Fit')
    plt.step(emcee_wave, emcee_spec, color = 'red', alpha = 0.5, label = 'Data', where = 'mid')
    plt.errorbar(emcee_wave, emcee_spec, yerr = emcee_err, color = 'gray', alpha = 0.5, fmt = 'none')
    
    l16_params = emcee_df.quantile(q = 0.16).values[:-2]
    u84_params = emcee_df.quantile(q = 0.84).values[:-2]

    l16_model = line_model(xarr, *l16_params, line_center )
    med_model = line_model(xarr, *emcee_df.quantile(q = 0.5).values[:-2], line_center )
    u84_model = line_model(xarr, *u84_params, line_center )
    
    plt.plot(xarr, med_model, label = 'Model', color = 'black')
    
    plt.fill_between(xarr, u84_model, l16_model, color = 'dodgerblue', alpha = 0.5)
    
    plt.xlabel(r'Wavelength [$\mu$m]')
    plt.ylabel('Flux')
    plt.legend()
    plt.show()

    spec_dict = {}
    spec_dict['input_wave'] = emcee_wave
    spec_dict['input_spec'] = emcee_spec
    spec_dict['input_spec_err'] = emcee_err
    spec_dict['model_wave'] = xarr
    spec_dict['model_spec'] = med_model
    spec_dict['model_spec_l16'] = l16_model
    spec_dict['model_spec_u84'] = u84_model

    return emcee_df, spec_dict




def fitting_line(wave, flux, flux_err, line_center, line_window, cont_window, run = 3000,
                 diagnose = False,save_df=True, save_spec = False, 
                 file_spec = 'Emcee_Spectra.txt', 
                 filename = 'Emcee_Chains_Galaxy.txt'):
    
    '''
    The code that fits the line using the emcee approach
    
    Inputs
    -----------
    wave: Wavelength array
    flux: Flux array
    flux_err: Flux error array
    line_center: the line center
    window_wavelength: The window near emission line in units of wavelength
    run: how many iterations to run emcee on default is 3000
    diagnose: An optional argument to output diagnostic plots as the fitting is proceeding
              Outputs plots from the initial fits, walker locations prior to using emcee and output emcee plots
    save_df: Saving the emcee df output to a file
    save_spec: Saving spectra used in emcee fitting to a file
    file_spec: name of the file for the emcee spectra
    filename:The name of the file to save the emcee output
    
    Returns
    -----------
    emcee_wave: The wavelength array used in the emcee fitting 
    emcee_spec: the flux spectra array used in the emcee fitting
    emcee_err: The error array used in the emcee fitting 
    emcee_df: the output emcee data frame with parameter values and flux estimates using Flux = A*sigma*sqrt(2 pi)
    '''
    
    
    #calling the function that does the initial fitting
    result = initial_fits(wave, flux, flux_err, line_window, cont_window, line_center, diagnose = diagnose)
    
    #getting the results from the initial fit to then pass into emcee
    guess_A = result[0]
    guess_mu = result[1]
    guess_sigma = result[2]
    guess_m = result[3]
    guess_b = result[4]
    
    
    #making walkers so that we can use emcee to explore the parameter space
    #centered on the best results from minimization
    amp_jump = np.random.normal(loc = guess_A,            #centered on best A from curve_fit
                                scale = guess_A/2,       #can wander 1/2 of the value of A
                                size = 32).reshape(-1, 1) 
    
    wavelength_jump = np.random.normal(loc = guess_mu,    #centered on best mu from curve_fit
                                       scale = 20,      #can wander +/- 0.005 microns (again tailored to nirspec
                                       size = 32).reshape(-1, 1)#data so if you are working with other spectra 
                                                                #you may need ot update this)
    
    sigma_jump = np.random.normal(loc = guess_sigma,       #centered on best sigma from curve_fit
                                  scale = 20,            #can wander +/- 0.002 microns (tailored for nirspec data)
                                  size = 32).reshape(-1, 1)

    powerm = np.log10(np.abs(guess_m))
    
    m_jump = np.random.normal(loc = guess_m,       #centered on best sigma from curve_fit
                                  scale = 1*10**powerm,        
                                  size = 32).reshape(-1, 1)

    
    #getting the power of 10 that the linear fit is
    powerb = np.log10(np.abs(guess_b))
    
    #
    b_jump = np.random.normal(loc = guess_b,           #centered on best b from curve_fit
                              scale = 1*10**powerb,    #making it wander 10^powerb (if b = .05, it can wander .01)
                              size = 32).reshape(-1, 1)

    
    #################
    # Diagnostic plotting to see if the parameters were jumping to large values
    # The should be concentrated near their best fit results values
    #################
    if diagnose == True:
        print('Checking the Walker Jumps')
        fig, ax = plt.subplots(nrows = 3, ncols = 2, constrained_layout = True)
        
        ax[0, 0].hist(amp_jump)
        ax[0, 0].set_xlabel('Amplitude')
        
        ax[0, 1].hist(wavelength_jump)
        ax[0, 1].set_xlabel(r'$\mu$')
        
        ax[1, 0].hist(sigma_jump)
        ax[1, 0].set_xlabel(r'$\sigma$')

        ax[1, 1].hist(m_jump)
        ax[1, 1].set_xlabel(r'm')
        
        ax[2, 0].hist(b_jump)
        ax[2, 0].set_xlabel('b')
        
        plt.show()
    

    #stacking along the columns and generating the starting walkers
    starting_walkers = np.hstack((amp_jump,
                                  wavelength_jump, 
                                  sigma_jump,
                                  m_jump,
                                  b_jump))

    #initializing window for emcee around the best result mu
    emcee_window = line_window

    
    #masks = (np.abs(vr) < 3000)
    
    #getting indexes near the emission line based off of the emcee_window
    #looking at line_center +/- emcee_window
    emcee_indx = np.where((wave >= (line_center - emcee_window)) & 
                          (wave <= (line_center + emcee_window)))[0] 

    #emcee subsections
    emcee_spec = flux[emcee_indx]
    emcee_wave = wave[emcee_indx]
    emcee_err = flux_err[emcee_indx]

    if diagnose:
        plt.figure(figsize = (10, 5), constrained_layout = True)
        plt.title('Input Emcee Spectra')
        plt.step(emcee_wave, emcee_spec, color = 'black', where = 'mid')
        plt.errorbar(emcee_wave, emcee_spec, yerr = emcee_err, color = 'gray', fmt= 'none')
        plt.show()

    
    #initializing walker positions
    pos = starting_walkers
    nwalkers, ndim = pos.shape

    #initializing sampler
    sampler = emcee.EnsembleSampler(nwalkers, #giving emcee the walker positions
                                    ndim,     #giving it the dimension of the model(same as number of model parameters)
                                    log_probability, #giving it the log_probability function
                                    args=(emcee_wave, emcee_spec, emcee_err, guess_mu, guess_A), #arguments to pass into log_probability
                                    moves = [(emcee.moves.DEMove(), 0.5),        
                                             (emcee.moves.DESnookerMove(), 0.5)])

    #running emcee
    state = sampler.run_mcmc(pos, 1000)
    sampler.reset()
    sampler.run_mcmc(state, run, progress=False)

    #getting values back
    flat_samples = sampler.get_chain(flat=True)
    LnL_chain = sampler.flatlnprobability
    
    emcee_df = pd.DataFrame()
    emcee_df['A'] = flat_samples[:, 0]
    emcee_df['mu'] = flat_samples[:, 1]
    emcee_df['sigma'] = flat_samples[:, 2]
    emcee_df['m'] = flat_samples[:, 3]
    emcee_df['b'] = flat_samples[:, 4]
    emcee_df['LnL'] = LnL_chain[:]
    
    #removing values where the log_likelihood was infinite as these are bad fits
    emcee_df = emcee_df[np.isfinite(emcee_df.LnL.values)]
    
    #getting the flux from the parameter values
    fluxes_emcee = emcee_df['A'] * emcee_df['sigma'] * np.sqrt(2 * np.pi)
    
    emcee_df['Fluxes'] = fluxes_emcee
    
    if diagnose == True:
        
        print('Checking Prameter Posterior Distributions')
        fig, ax = plt.subplots(nrows = 3, ncols = 2, constrained_layout = True)
        
        emcee_df.A.hist(ax = ax[0, 0])
        emcee_df.mu.hist(ax = ax[0, 1])
        emcee_df.sigma.hist(ax = ax[1, 0])
        emcee_df.m.hist(ax = ax[1, 1])
        #emcee_df.m.hist(ax = ax[1, 0])
        emcee_df.b.hist(ax = ax[2, 0])
        
        plt.show()
    
    if diagnose == True:
        xarr = np.linspace(emcee_wave[0], emcee_wave[-1], 100)
        
        plt.figure()
        plt.title('Input Emcee Spectra and Emcee Fit')
        plt.step(emcee_wave, emcee_spec, color = 'black', alpha = 0.5, label = 'Data')
        #plt.scatter(emcee_wave, emcee_spec, color = 'black')
        plt.plot(xarr, line_model(xarr, *emcee_df.quantile(q = 0.5).values[:-2]), label = 'Model')
        plt.xlabel(r'Wavelength [$\mu$m]')
        plt.ylabel('Flux')
        plt.legend()
        plt.show()
    
    ###########
    #NOTE:
    #need to also give the filename argument otherwise it will overwrite the default file
    ###########
    if save_df == True:
        emcee_df.to_csv(filename, sep = ' ')
        
    else:
        return emcee_wave, emcee_spec, emcee_err, emcee_df