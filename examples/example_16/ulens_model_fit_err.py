import sys
import matplotlib.pyplot as plt
import emcee
import yaml
import numpy as np
from copy import copy, deepcopy

from ulens_model_fit import UlensModelFit, import_failed
try:
    import MulensModel as mm
except Exception:
    raise ImportError('\nYou have to install MulensModel first!\n')


class UlensModelFitErrorsScales(UlensModelFit):
    def __init__(
            self, photometry_files,
            starting_parameters=None, prior_limits=None, model=None,
            fixed_parameters=None,
            min_values=None, max_values=None, fitting_parameters=None,
            fit_constraints=None, plots=None, other_output=None
    ):
        self._check_MM_version()
        self._photometry_files = photometry_files
        self._starting_parameters_input = starting_parameters
        self._prior_limits = prior_limits
        self._model_parameters = model
        self._fixed_parameters = fixed_parameters
        self._min_values = min_values
        self._max_values = max_values
        self._fitting_parameters = fitting_parameters
        self._fit_constraints = fit_constraints
        self._plots = plots
        self._other_output = other_output

        self._which_task()
        self._set_default_parameters()
        self._set_data_cleaning()
        self._set_errorbars_scales_fitting()

        if self._task == 'fit':
            self._guess_fitting_method()
            self._check_starting_parameters_type()
            self._set_fit_parameters_unsorted()
        self._check_imports()

    def _set_errorbars_scales_fitting(self):
        """Checking if errorbars scales should be fitted 
        """
        files = [f if isinstance(f, dict) else {'file_name': f}
                 for f in self._photometry_files]
        self._fit_errorbars = []
        for file_ in files:
            if file_["fit_errorbars"] is None:
                file_["fit_errorbars"] = False
            self._fit_errorbars.append(file_["fit_errorbars"])
        if True in self._fit_errorbars:
            self._get_datasets()
            self._set_fit_errorbars_scales_params()
            
    def _set_data_cleaning(self):
        """Checking if bad data points should be clean
        """
        files = [f if isinstance(f, dict) else {'file_name': f}
                 for f in self._photometry_files]
        self._clean_datapoints = []
        for file_ in files:
            if file_["clean_points"] is None:
                file_["clean_points"] = False
            self._clean_datapoints.append(file_["clean_points"])
    
    
    def _get_datasets(self):
        """
        construct a list of MulensModel.MulensData objects
        """
        kwargs = {'add_2450000': True}
        if isinstance(self._photometry_files, str):
            self._photometry_files = [self._photometry_files]
        elif not isinstance(self._photometry_files, list):
            raise TypeError(
                'photometry_files should be a list or a str, but you '
                'provided ' + str(type(self._photometry_files)))
        files = [f if isinstance(f, dict) else {'file_name': f}
                 for f in self._photometry_files]
        self._datasets = []
        self._datasets_init = []

        for file_ in files:
            dataset = self._get_1_dataset(
                file_, kwargs)
            self._datasets.append(dataset)
            self._datasets_init.append(dataset.copy())

        if self._residuals_output:
            if len(self._residuals_files) != len(self._datasets):
                out = '{:} vs {:}'.format(
                    len(self._datasets), len(self._residuals_files))
                raise ValueError('The number of datasets and files for '
                                 'residuals ouptut do not match: ' + out)

    def _get_1_dataset(self, file_, kwargs):
        """
        Construct a single dataset and possibly rescale uncertainties in it.
        """
        scaling = file_.pop("scale_errorbars", None)
        fit_errorbars = file_.pop("fit_errorbars", None)
        clean_points= file_.pop("clean_points", None)
        bad= file_.pop("bad", None)
    
        try:
            dataset = mm.MulensData(**{**kwargs, **file_})

        except FileNotFoundError:
            raise FileNotFoundError(
                'Provided file path does not exist: ' +
                str(file_['file_name']))
        except Exception:
            print('Something went wrong while reading file ' +
                  str(file_['file_name']), file=sys.stderr)
            raise

        if scaling is not None:
            dataset.scale_errorbars(**scaling)

        if bad is not None:
            self._set_bad(bad,dataset)
            
        return dataset

    def _set_bad(self, bad,dataset):
        """
        Setting bad flags for dataset base on argument photometry_files['bad'] in yaml file
        """
        if os.path.isfile(bad): 
            bad_array=np.genfromtxt(bad)
            if  bad_array.dtype== np.dtype('bool'):
                if len(bad)==dataset.n_epochs: 
                    bad_bool=bad_array
                else: 
                    raise ValueError(
                        'File {:s} with boolean values shoud have the same lenght as the corresponding dataset'.format(str(bad))
                    )
            elif bad_array.dtype== np.dtype('int'):
                if max(bad_array)>=dataset.n_epochs and min(bad_array)<=0: 
                    bad_bool=np.full(dataset.n_epochs,False)
                    bad_bool[bad_array]=True
                else:
                    raise ValueError(
                        'Indexes in {:s} do not match the corresponding dataset'.format(str(bad))
                    )
            elif bad_array.dtype== np.dtype('float'):
                bad_bool=np.full(dataset.n_epochs,False)
                for (i , time) in enumerate(dataset.time()):
                    if time in bad_array : bad_bool[i]=False
            else:  
                raise ValueError(
                        'Wrong declaration of bad data points in file {:s}'.format(str(bad)),
                        'File should consists of boolean array of dataset lenght or identivies of bad epochs in form of indexes:*int*  or HJD stamps:*floats*'
                    )
        else:
           bad_bool=bad             
        try : dataset.bad(bad_bool)
        except:
            raise ValueError( 
                             'Something wrong with provided bad flags for dataset' + dataset)
            
    def _clean_data(self):
        """
        Clearing bad epochs in datasets 
        """
        for (i, dataset) in enumerate(self._event._datasets):
            if self._clean_datapoints[i]:
                (f_source_0, f_blend_0) = self._event.get_flux_for_dataset(i)
                if  self._fit_errorbars[i]:
                    index1= np.full(dataset.n_epochs, False)
                else:
                    index1=self._clean_by_residuals(self._event.fits[i],f_source_0, f_blend_0)
                index2=self._clean_by_nearby_points(dataset)
                index= np.logical_or(index1, index2,dataset.bad)
                dataset.bad=index    
        
    def _clean_by_residuals(self,fit,f_source_0, f_blend_0,limit=3):
        """
        Getting indexes of point with errors *limit* times smaller than  model residuals
        """                        
        residuals, err_residuals = fit.get_residuals(
                phot_fmt='flux', source_flux=f_source_0,
                blend_flux=f_blend_0,)
        
        index=limit*err_residuals < abs(residuals)
        return index
    
    def _clean_by_nearby_points(self,dataset,limit=3):
        """
        Getting indexes of point with errors *limit* times bigger than median of errors of 10 nearby point 
        """
        index = np.full(dataset.n_epochs, False)
        err_flux= dataset.err_flux 
        for k in range(dataset.n_epochs):
            if k < 5:
                median_err = np.median(err_flux[:10])
            elif k >= dataset.n_epochs - 5:
                median_err = np.median(err_flux[-10:])
            else:
                median_err = np.median(err_flux[k-5:k+5])
                
            if err_flux[k] > limit * median_err:
                index[k] = True
        return index
    
                          
    def _set_fit_errorbars_scales_params(self):
        """
        Setting parameters for errorbars scales fitting 
        """
        for (i, dataset) in enumerate(self._datasets):
            if self._fit_errorbars[i]:
                label = dataset.plot_properties['label']
                label_mod = label.replace(' ', '_')
                self._other_parameters.append('ERR_k_'+label_mod)
                self._latex_conversion_other['ERR_k_' +
                                             label_mod] = 'EER_{\\rm{k '+label+'}}'
                self._other_parameters.append('ERR_e_'+label_mod)
                self._latex_conversion_other['ERR_e_' +
                                             label_mod] = 'EER_{\\rm{e '+label+'}}'

        self._check_errorbars_scales_starting_params()
        self._check_errorbars_scales_ranges()

    def _check_errorbars_scales_starting_params(self):
        """
        Define starting values for errorbars scales fitting if there not already defined in yaml file 
        """
        declared = {**self._starting_parameters_input,
                    **self._fixed_parameters}.keys()
        for key in self._other_parameters:
            if key[:3] == 'ERR':
                if key not in declared:
                    if key[4] == 'k':
                        self._starting_parameters_input[key] = 'gauss 1. 0.1'
                    if key[4] == 'e':
                        self._starting_parameters_input[key] = 'gauss 0. 0.01'

    def _check_errorbars_scales_ranges(self):
        """
        Define max and min values for errorbars scales fitting if there not already defined in yaml file 
        """
        for key in self._other_parameters:
            if key[:3] == 'ERR':
                if key not in self._fixed_parameters.keys():
                    if key not in self._min_values.keys():
                        self._min_values[key] = 0.
                    if key not in self._max_values.keys():
                        if key[4] == 'k':
                            self._max_values[key] = 10.
                        if key[4] == 'e':
                            self._max_values[key] = 0.5

    def _get_ln_probability_for_other_parameters(self):
        """
        Function that defines calculation of
        ln(probability(other_parameters)).
        The logarithm is to the base of the constant e.

        If you have defined other_parameters, then you have to implement
        child class of UlensModelFit and re-define this function.

        NOTE: your implementation should primarily use *dict*
        `self._other_parameters_dict`, but all MM parameters are already
        set and kept in *MM.Event* instance `self._event`.
        """
        out = 0.
        if True in self._fit_errorbars:
            out += self._ln_prob_errors()

        return out

    def _ln_prob_errors(self):
        """
        Returns ln(probability())
        """

        out = 0
        for (i, _data) in enumerate(self._event.datasets):
            if _data.chi2_fmt == "flux":
                err = _data.err_flux
            elif _data.chi2_fmt == "mag":
                err = _data.err_mag
           
            out += np.sum(np.log(2*np.pi*np.power(err, 2.)))
        out *= -0.5
        return out

    def _update_datasets(self, scales):
        """
        Updates scales of errorbars
        """
        index = 0
        self._event.datasets=deepcopy(self._datasets_init)
        for (i, dataset) in enumerate(self._event.datasets):
            #dataset=self._datasets_init[i].copy()
            if self._fit_errorbars[i]:
                scaling = {"factor": scales[index], "minimum":  scales[index+1]}
                dataset.scale_errorbars(**scaling)
                index += 2
        self._clean_data()

    def _set_model_parameters(self, theta):
        """
        Set microlensing parameters of self._model
        and scales errorbars if there are fitted 

        Note that if only plotting functions are called,
        then self._fit_parameters and theta are empty.
        """

        if self._task == 'plot':
            return

        if len(self._fit_parameters_other) == 0:
            for (parameter, value) in zip(self._fit_parameters, theta):
                setattr(self._model.parameters, parameter, value)
        else:
            for (parameter, value) in zip(self._fit_parameters, theta):
                if parameter not in self._fit_parameters_other:
                    setattr(self._model.parameters, parameter, value)
                else:
                    self._other_parameters_dict[parameter] = value

        if True in self._fit_errorbars:
            scales = {k: v for k, v in self._other_parameters_dict.items()
                      if k.startswith('ERR')}
            scales_sorted = []
            for key in self._other_parameters:
                if key.startswith('ERR'):
                    scales_sorted.append(scales[key])
            
            self._update_datasets(scales_sorted)

    def _save_bad_indexes(self):
        """
        save indexes of bad points for each dataset
        """
        for (i, _data) in enumerate(self._event.datasets):
            if self._clean_datapoints[i]:
                bad = np.where(_data.bad==True)
                name = self._settings['photometry_files'][i]['file_name']
                name = os.path.split(name)[1][:-4]
                file = os.path.join(self._results_dir, name+'_bad_idx.dat')
                np.savetxt(file, bad, fmt="%d")


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
        self._save_bad_indexes()
        self._write_residuals()
        self._make_plots()

if __name__ == '__main__':
    # if len(sys.argv) != 2:
    #     raise ValueError('Exactly one argument needed - YAML file')
    # if 'yaml' in import_failed:
    #     raise ImportError('module "yaml" could not be imported :(')

    # input_file = sys.argv[1]

    input_file = '/home/data/bl401/mmroz/mulens/OB151609/err_scal/close_up_pin/OB151609_close_close_up_pin.yaml'

    with open(input_file, 'r') as data:
        settings = yaml.safe_load(data)

    ulens_model_fit = UlensModelFitErrorsScales(**settings)

    ulens_model_fit.run_fit()
    self = ulens_model_fit
