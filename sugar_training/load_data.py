"""script that load the data used to train sugar. """

import numpy as np
import pickle
import os
import sys

class load_data_sugar(object):
    """function that allowed to access to observed sed and spectral features."""

    def __init__(self, path_input='data_input/', training=True, validation=False):
        """
        Load all the data needed for training sugar.

        will load:

        -all spectra
        -spectra at max
        -spectal indicators at max
        """
        self.file_spectra = os.path.join(path_input, 'sugar_data_release.pkl')

        if sys.version_info[0] < 3:
            self.dic = pickle.load(open(self.file_spectra))
        else:
            self.dic = pickle.load(open(self.file_spectra, 'rb'), encoding='latin1')

        self.sn_name = list(self.dic.keys())
        Filter = np.array([True] * len(self.sn_name))

        for i in range(len(self.sn_name)):
            if self.dic[self.sn_name[i]]['sample'] == 'training' and not training:
                Filter[i] = False
            if self.dic[self.sn_name[i]]['sample'] == 'validation' and not validation:                
                Filter[i] = False

        self.sn_name = np.array(self.sn_name)[Filter]
        self.sn_name.sort()

        self.spectra = {}
        self.spectra_variance = {}
        self.spectra_wavelength = {}
        self.spectra_phases = {}

        self.spectra_at_max = []
        self.spectra_at_max_variance = []
        self.spectra_at_max_wavelength = []
        self.spectra_at_max_phases = []

        self.spectral_indicators = []
        self.spectral_indicators_error = []

        self.X0 = []
        self.X1 = []
        self.C = []
        self.mb = []
        self.dmz = []
        self.mb_err = []
        self.X1_err = []
        self.C_err = []
        self.C_mb_cov = []
        self.X1_C_cov = []
        self.X1_mb_cov = []

    def load_spectra(self):
        """
        Load time sed of snia.

        will load spectra and all other infomation needed

        -spectra
        -spectra variance
        -spectra wavelength
        -spectra phases respective to salt2.4 phases
        """
        for i in range(len(self.sn_name)):

            self.spectra.update({self.sn_name[i]: {}})
            self.spectra_variance.update({self.sn_name[i]: {}})
            self.spectra_wavelength.update({self.sn_name[i]: {}})
            self.spectra_phases.update({self.sn_name[i]: {}})

            for t in range(len(self.dic[self.sn_name[i]]['spectra'].keys())):
                if self.dic[self.sn_name[i]]['spectra'][t]['salt2_phase'] < 50.:
                    self.spectra[self.sn_name[i]].update({'%i'%t: self.dic[self.sn_name[i]]['spectra'][t]['Y']})
                    self.spectra_variance[self.sn_name[i]].update({'%i'%t: self.dic[self.sn_name[i]]['spectra'][t]['V']})
                    self.spectra_wavelength[self.sn_name[i]].update({'%i'%t: self.dic[self.sn_name[i]]['spectra'][t]['X']})
                    self.spectra_phases[self.sn_name[i]].update({'%i'%t: self.dic[self.sn_name[i]]['spectra'][t]['salt2_phase']})

    def load_spectral_indicator_at_max(self, missing_data=True):
        """
        Load spectral indicators at max of snia.

        will load spectra features at max and all other infomation needed

        -spectral indicator
        -spectral indicator error
        """
        si_list = ['EWCaIIHK', 'EWSiII4128', 'EWMgII',
                   'EWFe4800', 'EWSIIW', 'EWSiII5972',
                   'EWSiII6355', 'EWOI7773', 'EWCaIIIR',
                   'vSiII_4128_lbd', 'vSII_5454_lbd',
                   'vSII_5640_lbd', 'vSiII_6355_lbd']
        number_si = len(si_list)

        indicator_data = np.zeros((len(self.sn_name), number_si))
        indicator_error = np.zeros((len(self.sn_name), number_si))

        for i in range(len(self.sn_name)):
            indicator_data[i] = np.array([self.dic[self.sn_name[i]]['spectral_features'][si] for si in si_list])
            indicator_error[i] = np.array([self.dic[self.sn_name[i]]['spectral_features'][si+'_err'] for si in si_list])

        if missing_data:
            error = indicator_error[:, np.all(np.isfinite(indicator_data), axis=0)]
            data = indicator_data[:, np.all(np.isfinite(indicator_data), axis=0)]

        self.spectral_indicators = indicator_data
        self.spectral_indicators_error = indicator_error

        for sn in range(len(self.sn_name)):
            for si in range(number_si):
                if not np.isfinite(self.spectral_indicators[sn, si]):
                    self.spectral_indicators[sn, si] = np.average(data[:, si], weights=1. / error[:, si]**2)
                    self.spectral_indicators_error[sn, si] = 10**8

    def load_salt2_data(self):


        for i in range(len(self.sn_name)):
            #self.X0.append(meta[self.sn_name[i]]['salt2.X0'])
            self.X1.append(self.dic[self.sn_name[i]]['salt2_info']['X1'])
            self.C.append(self.dic[self.sn_name[i]]['salt2_info']['C'])
            self.mb.append(self.dic[self.sn_name[i]]['salt2_info']['delta_mu'])
            self.mb_err.append(self.dic[self.sn_name[i]]['salt2_info']['delta_mu_err'])
            self.dmz.append(self.dic[self.sn_name[i]]['salt2_info']['dmz'])
            self.X1_err.append(self.dic[self.sn_name[i]]['salt2_info']['X1_err'])
            self.C_err.append(self.dic[self.sn_name[i]]['salt2_info']['C_err'])
            self.C_mb_cov.append(self.dic[self.sn_name[i]]['salt2_info']['delta_mu_C_cov'])
            self.X1_C_cov.append(self.dic[self.sn_name[i]]['salt2_info']['X1_C_cov'])
            self.X1_mb_cov.append(self.dic[self.sn_name[i]]['salt2_info']['delta_mu_X1_cov'])

        #self.X0 = np.array(self.X0)
        self.X1 = np.array(self.X1)
        self.C = np.array(self.C)
        self.mb = np.array(self.mb)
        self.mb_err = np.array(self.mb_err)
        self.dmz = np.array(self.dmz)
        self.X1_err = np.array(self.X1_err)
        self.C_err = np.array(self.C_err)
        self.C_mb_cov = np.array(self.C_mb_cov)
        self.X1_C_cov = np.array(self.X1_C_cov)
        self.X1_mb_cov = np.array(self.X1_mb_cov)
                                                                                        

if __name__=='__main__':

    lds = load_data_sugar()
    lds.load_spectra()
    lds.load_spectral_indicator_at_max()
