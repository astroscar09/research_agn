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

sigma_left = 5
sigma_right = 50

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

def full_model(x, A1, mu1, A2, mu2, sigma, m, b, line_center):
    '''
    Generalized model for multiple Gaussians plus a line.
    
    Inputs
    ------------
    x: array of values to evaluate the model
    params: list of parameters [A1, mu1, sigma1, A2, mu2, sigma2, ..., m, b, line_center]
    
    Returns
    ------------
    Evaluated model for the given parameters at the points in x
    '''
    #num_gaussians = (len(params) - 3) // 3

    model = gaussian(x, A1, mu1, sigma)+ gaussian(x, A2, mu2, sigma) + line(x, m, b, line_center)
    
    return model

def single_gaussian_model(x, A, mu, sigma, m, b, line_center):

    model = gaussian(x, A, mu, sigma)+ line(x, m, b, line_center)
    
    return model

def log_likelihood(theta, x, y, yerr, mu1, mu2, line_center):
    '''
    This is the likelihood function we are using for emcee to run
    
    This likelihood function is the maximum likelihood assuming gaussian errors.
    '''

    A1, A2, mu, sigma, m, b = theta

    ratio = mu2/mu1
    
    new_theta = [A1, mu, A2, mu*ratio, sigma, m, b]
    
    model = full_model(x, *new_theta, line_center)
    lnL = -0.5 * np.sum((y - model) ** 2 / yerr**2)
    return lnL


def log_prior(theta, wave_center, Amp_max, m, b, min_b):
    '''
    The prior function to be used against the parameters to impose certain criteria for the fitting
    
    '''
    scale = 10
    #Theta values that goes into our Gaussian Model
    A1, A2, mu, sigma, m, b = theta
    
    #min and max amplitude of the emission line
    min_A = 0
    max_A = Amp_max * 5
    
    sigma_window_left = sigma_left #had to change these for the input spectra these are left bounds for sigma
    sigma_window_right = sigma_right #had to change these for the input spectra these are right bounds for sigma

    if m < 0:
        low_bounds_m = m * scale
        high_bounds_m = m/scale
    else:
        low_bounds_m = m/scale
        high_bounds_m = m*scale

    mu_min = wave_center - 25
    mu_max = wave_center + 25

    mu_prior = (mu_min < mu) & (mu < mu_max)
    
    low_bounds_b = min_b
    high_bounds_b = Amp_max
        
    m_bounds = (low_bounds_m < m) & (m < high_bounds_m)  
    b_bounds = (low_bounds_b < b) & (b < high_bounds_b) 
        
    if (0 < A1 < max_A) & \
        (0 < A2 < max_A) & \
        (mu_prior) & \
        (sigma_window_left <= sigma < sigma_window_right) & \
        (m_bounds) & (b_bounds):
        return 0.0
    else:
        return -np.inf


def log_probability(theta, x, y, yerr, center, Amp_max, m, b, mu1, mu2, min_b):
    
    lp = log_prior(theta, center, Amp_max, m, b, min_b)
    if not np.isfinite(lp):
        #print('Probability is infinite')
        return -np.inf
    prob = lp + log_likelihood(theta, x, y, yerr, mu1, mu2, mu1)
    #print(f'Prob:{prob:.3E}')
    return prob


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

    
    num_gaussians = 2
    line_centers = []
    for i in range(num_gaussians):
        line_center = float(input(f'Enter the Line Center for Gaussian {i+1}: '))
        line_centers.append(line_center)
    print(f'Received line centers: {line_centers}')

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
    #guess_A = float(input('Enter the guess for the amplitude for the fit (Peak flux of the line is preferred): '))
     
    guesses_A = []
    
    for i in range(num_gaussians):
        guess_A = float(input(f'Enter the guess for the amplitude for Gaussian {i+1}: '))
        guesses_A.append(guess_A)
        
    guess_sigma = float(input(f'Enter sigma for the 2 gaussians: '))

    global sigma_left
    global sigma_right
    
    sigma_left = float(input(f'Enter the lower bound for sigma: '))
    sigma_right = float(input(f'Enter the lower bound for sigma: '))

    
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
    #x0A = guess_A - fit(guess_mu) #we subtract the continuum from the peak of the gaussian to get its actual amplitude
    #A1, mu1, A2, mu2, sigma, m, b, line_center

    guess_A1 = guesses_A[0] - b_guess
    guess_A2 = guesses_A[1] - b_guess

    if guess_A1 < 0:
        guess_A1 = 0.001 *10**np.log10(np.abs(guess_A1))

    if guess_A2 < 0:
        guess_A2 = 0.001 *10**np.log10(np.abs(guess_A2))
    
    x0 = [guess_A1, line_centers[0], guess_A2, line_centers[1], guess_sigma, m_guess, b_guess]
    
    print('Curve_Fit Guesses')
    for x in x0:
        print(f'{x}')

    
    xarr = np.linspace(wave_window[0], wave_window[-1], 1000)
    plt.figure()
    plt.title('Model Using Initial Guesses')
    plt.plot(xarr, full_model(xarr, *x0, line_centers[0]), label = 'Intitial Guess Model')
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

    result, window = initial_fits_user_input(wave, flux, flux_err)

    guess_A1 = result[0]
    mu1 = result[1]
    guess_A2 = result[2]
    mu2 = result[3]
    guess_sigma = result[4]
    guess_m = result[5]
    guess_b = result[6]
    
    #A1, A2, sigma, m, b
    #making walkers so that we can use emcee to explore the parameter space
    #centered on the best results from minimization
    
    #if guess_A1 == 0:
        
    amp1_jump = np.random.normal(loc = guess_A1,            #centered on best A from curve_fit
                                scale = guess_A1/2,       #can wander 1/2 of the value of A
                                size = 32).reshape(-1, 1) 

    amp2_jump = np.random.normal(loc = guess_A2,            #centered on best A from curve_fit
                                scale = guess_A2/2,       #can wander 1/2 of the value of A
                                size = 32).reshape(-1, 1) 

    mu_jump = np.random.normal(loc = mu1,            #centered on best mu from user input
                                scale = 25,         #can wander +/- 25 AA 
                                size = 32).reshape(-1, 1) 

    ###################
    #NOTE: The scale of the wavelengths here were used for a spectrum that was in angstroms, for spectrum in Microns you will need to  
    #      Change the scale numbers to have the appropriate jump scale for your data
    ###################
    
    sigma_jump = np.random.normal(loc = guess_sigma,       #centered on best sigma from curve_fit
                                  scale = 20,            #can wander +/- 20 Angstroms
                                  size = 32).reshape(-1, 1)

    powerm = np.log10(np.abs(guess_m))
    
    m_jump = np.random.normal(loc = guess_m,       #centered on best guess m from input
                                  scale = 1*10**powerm,        
                                  size = 32).reshape(-1, 1)

    
    #getting the power of 10 that the linear fit is
    powerb = np.log10(np.abs(guess_b))
    
    #
    b_jump = np.random.normal(loc = guess_b,           #centered on best b from user input
                              scale = 1*10**powerb,    #making it wander 10^powerb (if b = .05, it can wander .01)
                              size = 32).reshape(-1, 1)

    
    #################
    # Diagnostic plotting to see if the parameters were jumping to large values
    # The should be concentrated near their best fit results values
    #################
    
    print('Checking the Walker Jumps')
    fig, ax = plt.subplots(figsize = (10, 14), nrows = 3, ncols = 2, constrained_layout = True)
    
    ax[0, 0].hist(amp1_jump)
    ax[0, 0].set_xlabel('Amplitude 1')

    ax[0, 1].hist(amp2_jump)
    ax[0, 1].set_xlabel('Amplitude 2')

    ax[1, 0].hist(mu_jump)
    ax[1, 0].set_xlabel('mu')
    
    ax[1, 1].hist(sigma_jump)
    ax[1, 1].set_xlabel(r'$\sigma$')

    ax[2, 0].hist(m_jump)
    ax[2, 0].set_xlabel(r'm')
    
    ax[2, 1].hist(b_jump)
    ax[2, 1].set_xlabel('b')
    
    plt.show()
    

    #stacking along the columns and generating the starting walkers
    starting_walkers = np.hstack((amp1_jump,
                                  amp2_jump,
                                  mu_jump,
                                  sigma_jump,
                                  m_jump,
                                  b_jump))

    #initializing window for emcee around the best result mu
    emcee_window = window

    
    
    #getting indexes near the emission line based off of the emcee_window
    #looking at line_center +/- emcee_window
    emcee_indx = np.where((wave >= (mu1 - emcee_window)) & 
                          (wave <= (mu1 + emcee_window)))[0] 

    #emcee subsections
    emcee_spec = flux[emcee_indx]
    emcee_wave = wave[emcee_indx]
    emcee_err = flux_err[emcee_indx]

    #plotting the input emcee spectrum
    plt.figure(figsize = (10, 5), constrained_layout = True)
    plt.title('Input Emcee Spectra')
    plt.step(emcee_wave, emcee_spec, color = 'black', where = 'mid')
    plt.errorbar(emcee_wave, emcee_spec, yerr = emcee_err, color = 'gray', fmt= 'none')
    plt.show()

    
    #initializing walker positions
    pos = starting_walkers
    nwalkers, ndim = pos.shape

    #initializing sampler center, Amp_max, m, b, mu1, mu2
    sampler = emcee.EnsembleSampler(nwalkers, #giving emcee the walker positions
                                    ndim,     #giving it the dimension of the model(same as number of model parameters)
                                    log_probability, #giving it the log_probability function
                                    args=(emcee_wave, emcee_spec, emcee_err, 
                                          mu1, np.amax(emcee_spec), guess_m, guess_b, 
                                          mu1, mu2, 
                                          np.amin(emcee_spec)), #arguments to pass into log_probability
                                    
                                   )

    #running emcee
    state = sampler.run_mcmc(pos, 1000)
    sampler.reset()
    sampler.run_mcmc(state, 3000, progress=False)

    #getting values back
    flat_samples = sampler.get_chain(flat=True)
    LnL_chain = sampler.flatlnprobability
    
    emcee_df = pd.DataFrame()
    emcee_df['A1'] = flat_samples[:, 0]
    emcee_df['A2'] = flat_samples[:, 1]
    emcee_df['mu'] = flat_samples[:, 2]
    emcee_df['sigma'] = flat_samples[:, 3]
    emcee_df['m'] = flat_samples[:, 4]
    emcee_df['b'] = flat_samples[:, 5]
    emcee_df['LnL'] = LnL_chain[:]
    
    #removing values where the log_likelihood was infinite as these are bad fits
    emcee_df = emcee_df[np.isfinite(emcee_df.LnL.values)]
    
    #getting the flux from the parameter values
    fluxes_emcee1 = emcee_df['A1'] * emcee_df['sigma'] * np.sqrt(2 * np.pi)
    fluxes_emcee2 = emcee_df['A2'] * emcee_df['sigma'] * np.sqrt(2 * np.pi)

    sum_fluxes = fluxes_emcee1 + fluxes_emcee2
    
    emcee_df['Fluxes_1'] = fluxes_emcee1
    emcee_df['Fluxes_2'] = fluxes_emcee2
    emcee_df['Combined_Fluxes'] = sum_fluxes

    print('Checking Prameter Posterior Distributions')
    fig, ax = plt.subplots(figsize = (10, 12), nrows = 3, ncols = 2, constrained_layout = True)
    
    emcee_df.A1.hist(ax = ax[0, 0])
    emcee_df.A2.hist(ax = ax[0, 1])
    emcee_df.mu.hist(ax = ax[1, 0])
    emcee_df.sigma.hist(ax = ax[1, 1])
    emcee_df.m.hist(ax = ax[2, 0])
    emcee_df.b.hist(ax = ax[2, 1])

    ax = ax.flatten()
    ax[0].set_xlabel('Ampitude 1')
    ax[1].set_xlabel('Ampitude 2')
    ax[2].set_xlabel('mu')
    ax[3].set_xlabel(r'$\sigma$')
    ax[4].set_xlabel('m')
    ax[5].set_xlabel('b')
    
    plt.show()

    xarr = np.linspace(emcee_wave[0], emcee_wave[-1], 100)

    line_center = mu1
    
    plt.figure()
    plt.title('Input Emcee Spectra and Emcee Fit')
    plt.step(emcee_wave, emcee_spec, color = 'red', alpha = 0.5, label = 'Data', where = 'mid')
    plt.errorbar(emcee_wave, emcee_spec, yerr = emcee_err, color = 'gray', alpha = 0.5, fmt = 'none')

    print(emcee_df.quantile(q = (.16, .5, .84)))
    l16_params = emcee_df.quantile(q = 0.16).values[:-3]
    med_params = emcee_df.quantile(q = 0.5).values[:-3]
    u84_params = emcee_df.quantile(q = 0.84).values[:-3]

    ratio = mu2/mu1
    
    l16_model_params = [l16_params[0], l16_params[2], l16_params[1], l16_params[2]*ratio, l16_params[3], l16_params[4], l16_params[5]]
    med_model_params = [med_params[0], med_params[2], med_params[1], med_params[2]*ratio, med_params[3], med_params[4], med_params[5]]
    u84_model_params = [u84_params[0], u84_params[2], u84_params[1], u84_params[2]*ratio, u84_params[3], u84_params[4], u84_params[5]]

    l16_model = full_model(xarr, *l16_model_params, line_center)
    med_model = full_model(xarr, *med_model_params, line_center )
    u84_model = full_model(xarr, *u84_model_params, line_center )

    l16_model_gauss1 = [l16_params[0], l16_params[2], l16_params[3], l16_params[4], l16_params[5], mu1]
    med_model_gauss1 = [med_params[0], med_params[2], med_params[3], med_params[4], med_params[5], mu1]
    u84_model_gauss1 = [u84_params[0], u84_params[2], u84_params[3], u84_params[4], u84_params[5], mu1]

    l16_model_gauss2 = [l16_params[1], l16_params[2]*ratio, l16_params[3], l16_params[4], l16_params[5], mu1]
    med_model_gauss2 = [med_params[1], med_params[2]*ratio, med_params[3], med_params[4], med_params[5], mu1]
    u84_model_gauss2 = [u84_params[1], u84_params[2]*ratio, u84_params[3], u84_params[4], u84_params[5], mu1]

    gauss1_l16 = single_gaussian_model(xarr, *l16_model_gauss1)
    gauss1_med = single_gaussian_model(xarr, *med_model_gauss1)
    gauss1_u84 = single_gaussian_model(xarr, *u84_model_gauss1)

    gauss2_l16 = single_gaussian_model(xarr, *l16_model_gauss2)
    gauss2_med = single_gaussian_model(xarr, *med_model_gauss2)
    gauss2_u84 = single_gaussian_model(xarr, *u84_model_gauss2)
    
    plt.plot(xarr, med_model, label = 'Model', color = 'black', zorder = 99, alpha = 0.7)
    
    plt.fill_between(xarr, u84_model, l16_model, color = 'dodgerblue', alpha = 0.5)

    plt.plot(xarr, gauss1_med, color = 'red', label = 'Gauss1', ls= '--')
    plt.plot(xarr, gauss2_med, color = 'purple', label = 'Gauss2', ls = '--')
    
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
    spec_dict['gauss1_l16'] = gauss1_l16
    spec_dict['gauss1_med'] = gauss1_med
    spec_dict['gauss1_u84'] = gauss1_u84

    spec_dict['gauss2_l16'] = gauss2_l16
    spec_dict['gauss2_med'] = gauss2_med
    spec_dict['gauss2_u84'] = gauss2_u84

    return emcee_df, spec_dict