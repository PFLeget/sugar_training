"""compute qi, av, and grey offset from spectral fitting."""

import numpy as np
import sugar_training as st
import scipy.interpolate as inter
import os

class aligne_SED:
    """
    Aligne sugar model on observed sed.
    """
    def __init__(self, path_input, sn, ALPHA, M0, X, Rv):

        self.dic = st.load_pickle(os.path.join(path_input, 'sugar_data_release.pkl'))
        Phase = []
        self.IND = []
        for j in range(len(self.dic[sn]['spectra'].keys())):
            if self.dic[sn]['spectra'][j]['salt2_phase']>-12 and self.dic[sn]['spectra'][j]['salt2_phase']<42:
                Phase.append(self.dic[sn]['spectra'][j]['salt2_phase'])
                self.IND.append(j)
        self.Phase = np.array(Phase)
        self.sn = sn
        self.Alpha = ALPHA
        self.M0 = M0
        self.Rv = Rv
        self.X = X
        
        self.number_bin_phase = 0
        self.number_bin_wavelength = 0
        wavelength = self.X
        for i in range(len(wavelength)):
            if wavelength[i]==wavelength[0]:
                self.number_bin_phase += 1
        self.number_bin_wavelength = len(wavelength)/self.number_bin_phase
                                                                        
        
    def align_SED(self):
        Time = np.linspace(-12,48,21)
        DELTA = len(self.Phase)
        self.SED = np.ones((self.number_bin_wavelength*len(self.Phase),3+len(self.Alpha[0])))
        for Bin in range(self.number_bin_wavelength):
            SPLINE_Mean=inter.InterpolatedUnivariateSpline(Time,self.M0[Bin*21:(Bin+1)*21])
            self.SED[:,0][Bin*DELTA:(Bin+1)*DELTA]=SPLINE_Mean(self.Phase)
            for i in range(len(self.Alpha[0])):
                SPLINE = inter.InterpolatedUnivariateSpline(Time,self.Alpha[:,i][Bin*21:(Bin+1)*21])
                self.SED[:,i+3][Bin*DELTA:(Bin+1)*DELTA] = SPLINE(self.Phase)
            self.SED[:,2][Bin*DELTA:(Bin+1)*DELTA] = st.extinctionLaw(self.dic[self.sn]['spectra'][0]['X'][Bin],Rv=self.Rv)

        reorder = np.arange(self.number_bin_wavelength*DELTA).reshape(self.number_bin_wavelength, DELTA).T.reshape(-1)
        for i in range(len(self.SED[0])):
            self.SED[:,i] = self.SED[:,i][reorder]

    def align_spectra(self):
        self.Y = np.zeros(self.number_bin_wavelength*len(self.Phase))
        self.Y_err = np.zeros(self.number_bin_wavelength*len(self.Phase))
        DELTA = len(self.Phase)
        for Bin in range(DELTA):
            self.Y[Bin*self.number_bin_wavelength:(Bin+1)*self.number_bin_wavelength]=self.dic[self.sn]['spectra'][self.IND[Bin]]['Y']
            self.Y_err[Bin*self.number_bin_wavelength:(Bin+1)*self.number_bin_wavelength]=np.sqrt(self.dic[self.sn]['spectra'][self.IND[Bin]]['V'])


class global_fit:
     """
     Fit h factor on spectra in respect with sugar.
     """
     def __init__(self, Y, SED, dY=None, CovY=None):

         self.Y = Y
         self.A = SED
         if dY is None and CovY is None:
             self.WY = np.eye(len(self.Y))
         else:
             if dY is not None:
                 self.WY = np.eye(len(self.Y))*1./dY**2
             if CovY is not None:
                 self.WY = np.linalg.inv(CovY)

         self.N_comp = len(self.A[0])-1
    
     def separate_alpha_M0(self):
          self.M0=self.A[:,0]
          self.alpha=self.A[:,1:]

     def compute_h(self):
          """
          Compute the true value of the data,
          Av and the grey offset 
          """
          self.separate_alpha_M0()

          A = np.zeros((self.N_comp,self.N_comp))
          B = np.zeros(self.N_comp)
          H = np.zeros(self.N_comp)
          Y = np.zeros(len(self.Y))
          
          Y = self.Y - self.M0
          T = np.dot(self.alpha.T,self.WY.dot(self.alpha))
          A = np.linalg.inv(T)
          B = np.dot(self.alpha.T,self.WY.dot(np.matrix(Y).T))

          H = (np.dot(A, np.matrix(B))).T

          self.h = H
          self.cov_h = A

def comp_sugar_param(path_input= 'data_input/', path_output = 'data_output/', model = 'sugar_model_2.60_Rv.pkl'):

    dic_sed = st.load_pickle(os.path.join(path_output, model))
    ld = st.load_data_sugar(path_input=path_input, training=True, validation=True)
    HH = []
    dic = {}
    SN = ld.sn_name
    for i,SNN in enumerate(SN):
        print(i)
        ASED = aligne_SED(path_input, SNN, dic_sed['alpha'], dic_sed['m0'], dic_sed['X'], dic_sed['Rv'])
        ASED.align_SED()
        ASED.align_spectra()

        GF = global_fit(ASED.Y,ASED.SED,dY=ASED.Y_err,CovY=None)
        GF.compute_h()
        HH.append(GF.h)
        A = np.array(HH[i])
        h = A[0]
        cov_h = GF.cov_h
        dic.update({SNN:{'Av':h[1],
                         'grey':h[0],
                         'cov_q':cov_h}})
        for comp in range(len(h)-2):
            dic[SNN].update({'q%i'%(comp+1):h[2+comp]})

    st.write_pickle(dico, os.path.join(path_output, 'sugar_parameters.pkl'))

if __name__=="__main__":

    comp_sugar_param(path_intput= 'data_input/', path_output = 'data_output/', model = 'sugar_model_2.60_Rv.pkl')
    

