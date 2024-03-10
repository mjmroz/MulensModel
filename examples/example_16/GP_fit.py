import sys
import yaml
import numpy as np
import celerite2
from celerite2 import terms
from ulens_model_fit import UlensModelFit, import_failed
try:
    import MulensModel as mm
except Exception:
    raise ImportError('\nYou have to install MulensModel first!\n')
    
class UlensModelFitVariableBaseline(UlensModelFit):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._gp =  None
       
    
    
    
    
    def _setup_GP(self):
        '''
        GP_theta = []
        GP_names=['GP_ln_omega_0',
                                  'GP_Q_0',
                                  'GP_ln_S_0_omega_0^4',
                                  'GP_ln_rho',
                                  'GP_ln_sigma',
                                  'GP_ln_K_err^2',
                                  ]
        for name in GP_names:
            try:
                value = self._other_parameters_dict[name]
            except Exception:
                value = self._fixed_parameters[name]
            GP_theta.append(value)
                
        
        omega_0=np.exp(GP_theta[0])
        Q_0=GP_theta[1]
        S_0=np.exp(GP_theta[2])*np.power(omega_0,-4.)
        rho=np.exp(GP_theta[3])
        sigma=np.exp(GP_theta[4])
        K_err2=np.exp(GP_theta[5])
        

        kernel=terms.SHOTerm(w0=omega_0,Q=Q_0,S0=S_0)+terms.Matern32Term(sigma=sigma,rho=rho)
        self._gp = celerite2.GaussianProcess(kernel, mean=0.0)
        '''
        kernel=terms.SHOTerm(w0=1.,Q=1.,S0=1.)+terms.Matern32Term(sigma=1.,rho=1.)
        self._gp = celerite2.GaussianProcess(kernel, mean=0.0)
    
    
    def _set_default_parameters(self):
        """
        Extend the set of available parameters
        """
        super()._set_default_parameters()
        self._other_parameters = ['flux_0',
                                  'flux_b_1',
                                  'GP_ln_omega_0',
                                  'GP_Q_0',
                                  'GP_ln_S_0_omega_0^4',
                                  'GP_ln_rho',
                                  'GP_ln_sigma',
                                  'GP_ln_K_err^2',
                                  ]
        self._latex_conversion_other = {'flux_0_1':'f_{\\rm{base},1}',
                                        'flux_b_1': 'f_{{\rm b},1}',
                                        'GP_ln_omega_0':'\\rm{GP}_{ln \\omega,0}',
                                        'GP_Q_0': '\\rm{GP}_{Q,0}',
                                        'GP_ln_S_0_omega_0^4': '\\rm{GP}_{ln (S_0,\\omega_0^4)}',
                                        'GP_ln_rho': '\\rm{GP}_{ln \\rho}',
                                        'GP_ln_sigma': '\\rm{GP}_{ln \\sigma}',
                                        'GP_ln_K_err^2': '\\rm{GP}_{ln K_{\\rm{err}}^2}',
                                        }
    def _set_GP_parameters(self,t,yerr,GP_mu,GP_theta):
        """
        Setting all the parameters of the Gasussin Process
        """
        omega_0=np.exp(GP_theta[0])
        Q_0=GP_theta[1]
        S_0=np.exp(GP_theta[2])*np.power(omega_0,-4.)
        rho=np.exp(GP_theta[3])
        sigma=np.exp(GP_theta[4])
        K_err2=np.exp(GP_theta[5])
        
        '''
        fluxes = []
        times=[]
        yerr=[]
        for dataset in self._datasets:
            fluxes.append(np.copy(dataset._flux))
            times.append(np.copy(dataset.time))
       
        if len(self._datasets)>1:
            idx=np.argsort(times)
            times=times[idx]
            fluxes=fluxes[idx]
            yerr=yerr[idx]
        '''
        #self._gp.mean=GP_mu
        self._gp.kernel=terms.SHOTerm(w0=omega_0,Q=Q_0,S0=S_0)+terms.Matern32Term(sigma=sigma,rho=rho)
        self._gp.compute(t, diag=(yerr**2.) * K_err2, quiet=True)
        

        
        K_err2=np.exp(GP_theta[5])
        
        fluxes = []
        times=[]
        yerr=[]
                      

                            

                                        
    def _get_ln_probability_for_other_parameters(self):
        """
        We have to define this function and it has to return a float but in
        this case the change of ln_prob is coded in _ln_like().
        """
        out=0.
        GP_mu=0.0
        GP_theta = []
        GP_names=['GP_ln_omega_0',
                          'GP_Q_0',
                          'GP_ln_S_0_omega_0^4',
                          'GP_ln_rho',
                          'GP_ln_sigma',
                          'GP_ln_K_err^2',
                          ]
        for name in GP_names:
            try:
                value = self._other_parameters_dict[name]
            except Exception:
                value = self._fixed_parameters[name]
            GP_theta.append(value)
        
        
        
        show_bad=True
        ndata=0
        data_ref=0
        
        (f_source_0, f_blend_0) = self._event.get_flux_for_dataset(data_ref)
        
        for i, data in enumerate(self._datasets):
            # Evaluate whether or nor it is necessary to calculate the model
            # for bad datapoints.
            if show_bad:
                bad = True
            else:
                bad = False

            (y, yerr) = self._event.fits[i].get_residuals(
                        phot_fmt='scaled', source_flux=f_source_0,
                        blend_flux=f_blend_0, bad=show_bad)

            t=data.time
            
            self._set_GP_parameters(t,yerr,GP_mu,GP_theta)
            out+= self._gp.log_likelihood(y)
            
        
        #lnprior
        return out

    def _ln_like(self, theta):
        """
        likelihood function
        """
        self._set_model_parameters(theta)

        # changed - getting parameters:
        params_flux = []
        params_flux_names=['flux_0',
                                  'flux_b_1',] #have to be change when fitting to more than one data set
        
        flux_ratios=[1.] #have to be change when fitting to more than one data set
        
        for name in params_flux_names:
            try:
                value = self._other_parameters_dict[name]
            except Exception:
                value = self._fixed_parameters[name]
            params_flux.append(value)

       
        
        for i, dataset in enumerate(self._datasets):
            
            flux_s_1=(params_flux[0]-params_flux[1])/flux_ratios[i]
            self._event = mm.Event(self._datasets, self._model,fix_source_flux={dataset : flux_s_1},fix_blend_flux={dataset : params_flux[1]})
            self._event.sum_function = 'numpy.sum'
            self._set_n_fluxes()


        chi2 = self._event.get_chi2()
        
      
             
        out = 0. #-0.5 * chi2
        out = 0. -0.5 * chi2


        if self._print_model:
            self._print_current_model(theta, chi2)

        if self._task == 'fit' and len(self._other_parameters_dict) > 0:
            out += self._get_ln_probability_for_other_parameters()

        return out

# https://github.com/rpoleski/MulensModel/compare/master...ex16_galactic_model

# _get_samples_for_triangle_plot
# _get_labels_for_triangle_plot
# _get_parameters


    def run_fit(self):
        """
        Run the fit, print the output, and make the plots.

        This function does not accept any parameters. All the settings
        are passed via __init__().
        """
        if self._task != "fit":
            raise ValueError('wrong settings to run .run_fit()')

        self._check_plots_parameters()
        self._check_model_parameters()
        self._check_other_fit_parameters()
        self._parse_other_output_parameters()
        self._get_datasets()
        self._check_ulens_model_parameters()
        self._get_parameters_ordered()
        self._get_parameters_latex()
        self._parse_fitting_parameters()
        self._set_prior_limits()
        self._parse_fit_constraints()
        self._setup_GP()
        if self._fit_method == "EMCEE":
            self._parse_starting_parameters()

        self._check_fixed_parameters()
        self._make_model_and_event()
        if self._fit_method == "EMCEE":
            self._get_starting_parameters()
            

        self._setup_fit()

        self._run_fit()
        self._finish_fit()
        self._parse_results()
        self._write_residuals()
        self._make_plots()







if __name__ == '__main__':
    if len(sys.argv) != 2:
        raise ValueError('Exactly one argument needed - YAML file')
    if 'yaml' in import_failed:
        raise ImportError('module "yaml" could not be imported :(')

    input_file = sys.argv[1]
    input_file = '/home/mateusz/mulens/MulensModel/examples/example_16/ob08092-o4_GP.yaml'
    with open(input_file, 'r') as data:
        settings = yaml.safe_load(data)

    ulens_model_fit = UlensModelFitVariableBaseline(**settings)

    ulens_model_fit.run_fit()
