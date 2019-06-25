"""compute the sugar model."""

from scipy.sparse import block_diag
from scipy.sparse import coo_matrix

import numpy as np
import sugar_training as sugar
from math_toolbox import passage_plus_plus
import copy
import os
import pickle

class load_data_to_build_sugar:

    def __init__(self, path_output='data_output/', path_output_gp='data_output/gaussian_process/', ncomp=3, filtre=True, fit_Av= False, Rv=None):


        self.path_output = path_output
        self.path_output_gp = path_output_gp
        self.ncomp = ncomp
        self.fit_Av = fit_Av

        dicpca = sugar.load_pickle(os.path.join(self.path_output,'emfa_output.pkl'))
        pca_sn_name = np.array(dicpca['sn_name'])
            
        
        if filtre:
            FILTRE = dicpca['filter']
        else:
            FILTRE = np.array([True]*len(pca_sn_name))

        self.pca_error = dicpca['error'][FILTRE]
        self.pca_data = dicpca['data'][FILTRE]
        self.pca_val = dicpca['val']
        self.pca_vec = dicpca['vec']
        self.pca_norm = dicpca['norm']
        self.pca_Norm_data = dicpca['Norm_data'][FILTRE]
        self.pca_Norm_err = dicpca['Norm_err'][FILTRE]
        self.sn_name = pca_sn_name[FILTRE]

        if self.fit_Av:
            if Rv is None:
                ValueError('I need an Rv Value')
            self.Rv = Rv

        else:
            dic_model = sugar.load_pickle(os.path.join(self.path_output,'model_at_max.pkl'))
            self.sn_name_Av = dic_model['sn_name']
            self.Av = dic_model['Av_cardelli']
            self.Rv = dic_model['RV']
                   
    
        self.rep_GP = self.path_output_gp

        self.N_sn = len(self.sn_name)

        self.number_bin_phase = 0
        self.number_bin_wavelength = 0
        A=np.loadtxt(os.path.join(self.rep_GP,str(self.sn_name[0])+'.predict'))
        phase = A[:,0]
        wavelength = A[:,1]
        self.TX = wavelength
        for i in range(len(wavelength)):
            if wavelength[i]==wavelength[0]:
                self.number_bin_phase += 1

        self.number_bin_wavelength = int(len(wavelength) / self.number_bin_phase)

    def load_lssfr_resize_data(self, File_host, File_names):
        table_host = np.loadtxt(File_host, comments='#', delimiter=',', dtype='str',skiprows=1)
        sn_name_host =  table_host[:,16]
        lssfr = table_host[:,10].astype(float)
        lssfr_err_down = table_host[:,11].astype(float)
        lssfr_err_up = table_host[:,12].astype(float)

        dic = pickle.load(open(File_names))
        sn_real_names = dic.keys()
        sn_false_names = ['oui']*len(sn_real_names)
        for k, sn in enumerate(sn_real_names):            
            sn_false_names[k] = dic[sn]
        join_names = [sn_real_names, sn_false_names]

        sn_name = []
        for sn in sn_name_host:
            if sn in sn_real_names:                
                sn_name.append(sn)
                
        
        lssfr_ordered = []
        lssfr_err = []        
        data_resized = []

        FILTRE = np.array([False]*len(self.sn_name))
        for k, sn in enumerate(self.sn_name):
            sn_real_name_idx = join_names[1].index(sn)
            sn_real_name = join_names[0][sn_real_name_idx]
            if sn_real_name in sn_name :
                sn_name_host_list = sn_name_host.tolist()
                lssfr_idx = sn_name_host_list.index(sn_real_name)
                lssfr_ordered.append(lssfr[lssfr_idx])
                lssfr_err_k = (lssfr_err_up[lssfr_idx] + lssfr_err_down[lssfr_idx])*0.5
                lssfr_err.append(lssfr_err_k)
                FILTRE[k] = True
                
                new_data_k = self.data[k].tolist()
                new_data_k.append(lssfr_ordered[-1])
                data_resized.append(np.array(new_data_k))
                                      
        self.lssfr = lssfr_ordered 
        self.lssfr_err = np.array(lssfr_err)
        final_data = np.array(data_resized)
        self.std = np.zeros(4)
        for k in range(4):  
            self.std[k] = np.std(final_data[:,k])
            final_data[:,k] = (final_data[:,k] - np.mean(final_data[:,k]) ) / self.std[k]
        self.new_data = final_data
        self.sn_name = self.sn_name[FILTRE]
        self.N_sn = len(self.sn_name)
        self.ncomp = 4
        
    
    def compute_EM_PCA_data(self, File_host, File_names):

        # Ce que tu as fais dans test_PCA_p,
        # fais les manipulations ici 

        dat = self.pca_Norm_data
        err = self.pca_Norm_err

        new_base, cov_new_err = sugar.passage_plus_plus(dat, self.pca_vec, err = err, sub_space=self.ncomp)
        
        self.data = new_base
        self.Cov_error = cov_new_err

        self.load_lssfr_resize_data(File_host, File_names)
        dat = self.new_data
        cov = np.zeros((len(self.new_data), 4, 4))
        for sn in range(len(self.new_data)):
            cov[sn,:3,:3] = self.Cov_error[sn]
            cov[sn, 3, 3] = self.lssfr_err[sn]
            # norm error 
        for i in range(4):
            for j in range(4):
                cov[:, i, j] /= self.std[i]*self.std[j]
        
        C = (dat.transpose()).dot(dat) / (len(dat)-1)        
        val, vec = np.linalg.eig(C)        
        
        new_base, cov_new_err = sugar.passage_plus_plus(dat, vec, cov=cov, sub_space=4)
        print('FONCTIONNE')
        self.data = new_base
        self.Cov_error = cov_new_err

    def load_spectra_GP(self,sn_name):
        A = np.loadtxt(os.path.join(self.rep_GP, str(sn_name)+'.predict'))
        Y = A[:,2]
        if not self.fit_Av:
            for j,sn_av in enumerate(self.sn_name_Av):
                if sn_name == sn_av:
                    Y_cardelli_corrected = (Y-(self.Av[j]*sugar.extinctionLaw(A[:,1],Rv=self.Rv)))
            return Y_cardelli_corrected
        else:
            return Y

    def load_phase_wavelength(self,sn_name):
        
        A = np.loadtxt(os.path.join(self.rep_GP, str(sn_name)+'.predict'))
        phase = A[:,0]
        wavelength = A[:,1]
        del A
        return phase,wavelength

    def load_cov_matrix(self,sn_name):

        A = np.loadtxt(os.path.join(self.rep_GP, str(sn_name)+'.predict'))
        size_matrix = self.number_bin_phase*self.number_bin_wavelength
        COV = np.zeros((size_matrix,size_matrix))

        for i in range(self.number_bin_wavelength):
            cov = A[(i*self.number_bin_phase):((i+1)*self.number_bin_phase),3:]
            COV[i*self.number_bin_phase:(i+1)*self.number_bin_phase, i*self.number_bin_phase:(i+1)*self.number_bin_phase] = cov

        return COV
        
    def load_spectra(self):

        self.phase,self.X = self.load_phase_wavelength(self.sn_name[0])
        self.Y_cosmo_corrected = np.zeros((len(self.sn_name),len(self.X)))
        self.CovY = []

        for i,sn in enumerate(self.sn_name):
            print(sn)
            self.Y_cosmo_corrected[i] = self.load_spectra_GP(sn)

            Cov = self.load_cov_matrix(sn)
            COV = []
            for i in range(self.number_bin_wavelength):
                COV.append(coo_matrix(Cov[i*self.number_bin_phase:(i+1)*self.number_bin_phase, i*self.number_bin_phase:(i+1)*self.number_bin_phase]))
            self.CovY.append(block_diag(COV))


class make_sugar(load_data_to_build_sugar):

    def __init__(self, path_output='data_output/', path_output_gp='data_output/gaussian_process/', filtre=True, fit_Av=False, ncomp=3, Rv=2.6, File_host = '../Data/lssfr_paper_full_sntable.csv', File_names = '../Data/sugar_data_release_unblind.pkl'):

        load_data_to_build_sugar.__init__(self, path_output=path_output, path_output_gp=path_output_gp, filtre=filtre, ncomp=ncomp, fit_Av=fit_Av, Rv=Rv)
        self.compute_EM_PCA_data(File_host, File_names)
        self.load_spectra()

    def launch_sed_fitting(self):

        sedfit = sugar.sugar_fitting(self.data, self.Y_cosmo_corrected,
                                     self.Cov_error, self.CovY, self.X,
                                     size_bloc=self.number_bin_phase,
                                     fit_grey=True, fit_Av=self.fit_Av, Rv=self.Rv,
                                     sparse=True)

        del self.data
        del self.Y_cosmo_corrected
        del self.Cov_error
        del self.CovY

        self.sedfit = sedfit        
        self.sedfit.init_fit()
        self.sedfit.run_fit()
        self.sedfit.separate_component()
                                   
    def write_model_output(self):

        if self.ncomp == 3:
            pkl = os.path.join(self.path_output,'sugar_model_%.2f_Rv.pkl'%(self.Rv))
        else:
            pkl = os.path.join(self.path_output,'sugar_model_%i_%.2f_Rv.pkl'%((self.ncomp,self.Rv)))

        dic = {'alpha':self.sedfit.alpha,
               'm0':self.sedfit.m0,
               'delta_m_grey':self.sedfit.delta_m_grey,
               'Rv':self.Rv,
               'gamma_lambda':self.sedfit.gamma_lambda,
               'Av':self.sedfit.a_lambda0,
               'X':self.X,
               'h':self.sedfit._h,
               'sn_name':self.sn_name,
               'chi2':self.sedfit.chi2_save,
               'dof':self.sedfit.dof}
        
        sugar.write_pickle(dic, pkl)

if __name__ == '__main__':
    #ld = make_sugar()
    #ld.launch_sed_fitting()
    #ld.write_model_output()

    test = load_data_to_build_sugar(path_output='../../sugar_plus_plus/test_PCA_p/data_output/', path_output_gp='../../sugar_plus_plus/test_PCA_p/data_output/data_output_gp/', ncomp=3, filtre=True, fit_Av= False, Rv=None)
    test.compute_EM_PCA_data()



