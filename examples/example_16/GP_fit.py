import sys
import matplotlib.pyplot as plt
from matplotlib import gridspec, rcParams, rcParamsDefault
import emcee
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
        self._gp = None

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
        kernel = None
        self._gp = celerite2.GaussianProcess(kernel, mean=0.0)
        freq = np.linspace(1.0 / 200, 1.0 / 0.001, 1000)
        self._psd_omega = 2 * np.pi * freq

        n = 0
        m = 0
        for name in self._fixed_parameters.keys():
            if name[:2] == 'GP':
                m += 1
        for name in self._other_parameters_dict.keys():
            if name[:2] == 'GP':
                n += 1

        self._n_GP_parms_fitted = n
        self._n_GP_parms_fix = m

    def _check_plots_parameters(self):
        super()._check_plots_parameters
        allowed_keys = set(['best model', 'trajectory',
                           'triangle', 'trace', 'GP residual'])

    def _set_default_parameters(self):
        """
        Extend the set of available parameters
        """
        super()._set_default_parameters()
        self._other_parameters = ['flux_0',
                                  'flux_b_1',
                                  'GP_ln_sigma_1',
                                  'GP_ln_rho_1',
                                  'GP_ln_sigma_2',
                                  'GP_ln_rho_2',
                                  'GP_ln_K_err^2',
                                  'GP_Q_1'
                                  ]
        self._latex_conversion_other = {'flux_0_1': 'f_{\\rm{base},1}',
                                        'flux_b_1': 'f_{{\\rm b},1}',
                                        'GP_ln_sigma_1': '\\rm{GP}_{ln \\sigma,1}',
                                        'GP_ln_rho_1': '\\rm{GP}_{ln \\rho,1}',
                                        'GP_ln_sigma_2': '\\rm{GP}_{ln \\sigma,2}',
                                        'GP_ln_rho_2': '\\rm{GP}_{ln \\rho,2}',
                                        'GP_ln_K_err^2': '\\rm{GP}_{ln K^2_{\\rm{err}}}',
                                        'GP_Q_1': '\\rm{GP}_{Q,1}',
                                        }

    def _set_GP_parameters(self, t, yerr, GP_mu, GP_theta):
        """
        Setting all the parameters of the Gasussin Process
        """
        sigma1 = np.exp(GP_theta[0])
        rho1 = np.exp(GP_theta[1])
        sigma2 = np.exp(GP_theta[2])
        rho2 = np.exp(GP_theta[3])
        K_err2 = np.exp(GP_theta[4])
        Q1 = GP_theta[5]

        # self._gp.mean=GP_mu
        self._gp.kernel = terms.SHOTerm(
            sigma=sigma1, rho=rho1, Q=Q1)+terms.Matern32Term(sigma=sigma2, rho=rho2)
        self._gp.compute(t, diag=(yerr**2.) * K_err2, quiet=True)

    def _get_ln_probability_for_other_parameters(self):
        """
        We have to define this function and it has to return a float but in
        this case the change of ln_prob is coded in _ln_like().
        """
        out = 0.
        GP_mu = 0.0
        GP_theta = []
        GP_names = ['GP_ln_sigma_1',
                    'GP_ln_rho_1',
                    'GP_ln_sigma_2',
                    'GP_ln_rho_2',
                    'GP_ln_K_err^2',
                    'GP_Q_1'
                    ]
        for name in GP_names:
            try:
                value = self._other_parameters_dict[name]
            except Exception:
                value = self._fixed_parameters[name]
            GP_theta.append(value)

        show_bad = True
        ndata = 0
        data_ref = 0

        (f_source_0, f_blend_0) = self._event.get_flux_for_dataset(data_ref)

        for i, data in enumerate(self._datasets):
            # Evaluate whether or nor it is necessary to calculate the model
            # for bad datapoints.
            if show_bad:
                bad = True
            else:
                bad = False

            (y, yerr) = self._event.fits[i].get_residuals(
                phot_fmt='flux', source_flux=f_source_0,
                blend_flux=f_blend_0, bad=show_bad)

            t = data.time

            self._set_GP_parameters(t, yerr, GP_mu, GP_theta)
            out += self._gp.log_likelihood(y)

        # lnprior
        return out

    def _ln_like(self, theta):
        """
        likelihood function
        """
        self._set_model_parameters(theta)

        # changed - getting parameters:
        params_flux = []
        params_flux_names = ['flux_0',
                             'flux_b_1', ]  # have to be change when fitting to more than one data set

        # have to be change when fitting to more than one data set
        flux_ratios = [1.]

        for name in params_flux_names:
            try:
                value = self._other_parameters_dict[name]
            except Exception:
                value = self._fixed_parameters[name]
            params_flux.append(value)

        for i, dataset in enumerate(self._datasets):

            flux_s_1 = (params_flux[0]-params_flux[1])/flux_ratios[i]
            self._event = mm.Event(self._datasets, self._model, fix_source_flux={
                                   dataset: flux_s_1}, fix_blend_flux={dataset: params_flux[1]})
            self._event.sum_function = 'numpy.sum'
            self._set_n_fluxes()

        chi2 = self._event.get_chi2()

        out = 0.  # -0.5 * chi2
        out = 0. - 0.5 * chi2

        if self._print_model:
            self._print_current_model(theta, chi2)

        if self._task == 'fit' and len(self._other_parameters_dict) > 0:
            out += self._get_ln_probability_for_other_parameters()

        return (out, self._gp.kernel.get_psd(self._psd_omega))

# https://github.com/rpoleski/MulensModel/compare/master...ex16_galactic_model

# _get_samples_for_triangle_plot
# _get_labels_for_triangle_plot
# _get_parameters

    def _get_fluxes_to_print_EMCEE(self):
        """
        prepare values to be printed for EMCEE fitting
        """
        try:
            blobs = np.array(self._sampler.blobs)
        except Exception as exception:
            raise ValueError('There was some issue with blobs:\n' +
                             str(exception))
        blob_sampler = np.transpose(blobs, axes=(1, 0, 2))
        blob_samples = blob_sampler[:,
                                    self._fitting_parameters['n_burn']:, :self._n_fluxes]
        blob_samples = blob_samples.reshape((-1, self._n_fluxes))

        return blob_samples

    def _ln_prob(self, theta):
        """
        Log probability of the model - combines _ln_prior(), _ln_like(),
        and constraints which include fluxes.

        NOTE: we're using np.log(), i.e., natural logarithms.
        """
        ln_prior = self._ln_prior(theta)
        if not np.isfinite(ln_prior):
            return self._return_ln_prob(-np.inf)

        ln_like, psd = self._ln_like(theta)
        if not np.isfinite(ln_like):
            return self._return_ln_prob(-np.inf)

        ln_prob = ln_prior + ln_like

        fluxes = self._get_fluxes()

        ln_prior_flux = self._run_flux_checks_ln_prior(fluxes)
        if not np.isfinite(ln_prior_flux):
            return self._return_ln_prob(-np.inf)

        ln_prob += ln_prior_flux

        self._update_best_model_EMCEE(ln_prob, theta, fluxes)

        return self._return_ln_prob(ln_prob, fluxes, psd)

    def _return_ln_prob(self, value, fluxes=None, psd=None):
        """
        used to parse output of _ln_prob() in order to make that function
        shorter
        """
        if value == -np.inf:
            if self._return_fluxes:
                return (value,  np.concatenate(([0.] * self._n_fluxes, [0.]*len(self._psd_omega))))
            else:
                return (value, [0.]*len(self._psd_omega))
        else:
            if self._return_fluxes:
                if fluxes is None:
                    raise ValueError('Unexpected error!')
                return (value, np.concatenate((fluxes, psd)))
            else:
                return (value, psd)

    def _setup_fit_EMCEE(self):
        """
        Setup EMCEE fit
        """
        self._sampler = emcee.EnsembleSampler(
            self._n_walkers, self._n_fit_parameters, self._ln_prob)

    def _print_psd(self):
        """
        prepare psd to be printed 
        """
        try:
            blobs = self._sampler.get_blobs(
                flat=True, discard=self._fitting_parameters['n_burn'])
        except Exception as exception:
            raise ValueError('There was some issue with blobs:\n' +
                             str(exception))
        psds = blobs[:, self._n_fluxes:]

        q = np.percentile(psds, [16, 50, 84], axis=0)

        freq = self._psd_omega/2./np.pi
        plt.loglog(freq, q[1], color="C0")
        plt.fill_between(freq, q[0], q[2], color="C0", alpha=0.1)

        plt.xlim(freq.min(), freq.max())
        plt.xlabel("frequency [1 / day]")
        plt.ylabel("power [day ppt$^2$]")
        _ = plt.title("posterior psd using emcee")
        plt.show()

    def _plot_res(self):

        dpi = 300

        self._ln_like(self._best_model_theta)  # Sets all parameters to
        self._reset_rcParams()

        if 'rcParams' in self._plots['best model']:
            for (key, value) in self._plots['best model']['rcParams'].items():
                rcParams[key] = value

        kwargs_all = self._get_kwargs_for_best_model_plot()
        (kwargs_grid, kwargs_model, kwargs, xlim, t_1, t_2) = kwargs_all[:6]
        (kwargs_axes_1, kwargs_axes_2) = kwargs_all[6:]
        (ylim, ylim_residuals) = self._get_ylim_for_best_model_plot(t_1, t_2)

        true_t = np.linspace(t_1, t_2, 500)
        GP_mu = 0.0
        if kwargs_all[2]['subtract_2450000']:
            true_t -= 2450000.

        n = 0
        m = 0
        for name in self._fixed_parameters.keys():
            if name[:2] == 'GP':
                m += 1
        for name in self._other_parameters_dict.keys():
            if name[:2] == 'GP':
                n += 1

        self._n_GP_parms_fitted = n
        self._n_GP_parms_fix = m

        chain = self._sampler.get_chain(
            discard=self._fitting_parameters['n_burn'], flat=True)
        chain = chain[:, -self._n_GP_parms_fitted:]
        #chain= self._best_model_theta
        # sample=chain[-self._n_GP_parms_fitted:]

        data_ref = 0
        (f_source_0, f_blend_0) = self._event.get_flux_for_dataset(data_ref)

        for i, data in enumerate(self._datasets):
            # Evaluate whether or nor it is necessary to calculate the model
            # for bad datapoints.

            (y, yerr) = self._event.fits[i].get_residuals(
                phot_fmt='scaled', source_flux=f_source_0,
                blend_flux=f_blend_0, )

            t = data.time.copy()

            if kwargs_all[2]['subtract_2450000']:
                t -= 2450000.

            name = 'GP_Q_1'
            Q2 = self._fixed_parameters[name]

            for sample in chain[np.random.randint(len(chain), size=50)]:

                GP_theta = np.concatenate((sample, [Q2]))
                self._set_GP_parameters(t, yerr, GP_mu, GP_theta)
                conditional = self._gp.condition(y, true_t)
                plt.plot(true_t, conditional.sample(), color="C0", alpha=0.1)
                #plt.plot(true_t, [0.], "k", lw=1.5, alpha=0.3, label="data")
                plt.errorbar(t, y, yerr=yerr, fmt=".", capsize=0,
                             label="truth", color=data.plot_properties['color'])

        plt.xlim(*xlim)
        # plt.ylim(*ylim_residuals)
        plt.tick_params(**kwargs_axes_2)

        self._save_figure(self._plots['GP residual'].get('file'),  dpi=dpi)
        plt.show()

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
        self._print_psd()
        self._plot_res()


if __name__ == '__main__':
    if len(sys.argv) != 2:
        raise ValueError('Exactly one argument needed - YAML file')
    if 'yaml' in import_failed:
        raise ImportError('module "yaml" could not be imported :(')

    input_file = sys.argv[1]
#   input_file='ob08092-o4_GP.yaml'

    with open(input_file, 'r') as data:
        settings = yaml.safe_load(data)

    ulens_model_fit = UlensModelFitVariableBaseline(**settings)

    ulens_model_fit.run_fit()
    self = ulens_model_fit
