"""plot for extinction law."""

import pylab as plt
import numpy as np
import copy
import matplotlib.gridspec as gridspec
import sugar_training as st
import os

def go_log(ax=None, no_label=False):
    import matplotlib.ticker
    if ax is None:
        ax = plt.gca()
    ax.set_xscale('log')
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.get_xaxis().set_minor_formatter(matplotlib.ticker.ScalarFormatter())
    if no_label:
        plt.minorticks_off()

def go_logxy(ax=None, no_label=False):
    import matplotlib.ticker
    if ax is None:
        ax = plt.gca()
    ax.set_xscale('log')
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.get_xaxis().set_minor_formatter(matplotlib.ticker.ScalarFormatter())
    ax.set_yscale('log')
    ax.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.get_yaxis().set_minor_formatter(matplotlib.ticker.ScalarFormatter())
    if no_label:
        plt.minorticks_off()

def plot_disp_matrix(wlength, matrix, diag, ylabel, title, 
                     cmap=plt.cm.jet, plotpoints=True, VM=[-1,1]):
    
    fig = plt.figure(dpi=150,figsize=(10,10))
    plt.subplots_adjust(left=0.15, bottom=0.03, hspace=0)

    ax2 = fig.add_axes([0.08,0.08,0.73,0.73])
    
    wavex, wavey = np.meshgrid(wlength, wlength)
    im = ax2.pcolor(wavex, wavey, matrix, cmap=cmap, vmin=VM[0],vmax=VM[1])
    go_logxy(ax=ax2)
    plt.gca().invert_yaxis()
 
    ax_cbar = fig.add_axes([0.83, 0.085, 0.05, 0.72])
    cb = plt.colorbar(im, cax = ax_cbar)
    cb.set_label(ylabel,fontsize=25)
    
    ax2.set_xlabel(r'Wavelength [$\AA$]', fontsize=18)
    ax2.set_ylabel(r'Wavelength [$\AA$]', fontsize=18)
    
    ax1 = fig.add_axes([0.08,0.81,0.73,0.18])
    ax1.plot(wlength, diag, 'b', lw=3)
    ax1.set_xlim(ax2.get_xlim())
    go_log(ax=ax1, no_label=True)
    ax1.set_ylim(0,0.22)
    ax1.set_ylabel('$\sqrt{Diag(D)}$ (mag)', fontsize=14)
    

class at_max_plot:

    def __init__(self, path_output='data_output/'):

        # Load the hubble fit data

        dico = st.load_pickle(open(os.path.join(path_output, 'model_at_max.pkl')))
        self.Y_err = dico['Mag_all_sn_err']
        self.Mag_no_corrected = dico['Mag_no_corrected']
        self.alpha = dico['alpha']
        self.M0 = dico['M0']
        self.number_correction = dico['number_correction']
        self.Y_build = dico['Y_build']
        self.Y_build_error = dico['Y_build_error']
        self.data = dico['data']
        self.Cov_error = dico['Cov_error']
        self.X = dico['X']
        self.sn_name = dico['sn_name']
        self.Rv = dico['RV']
        self.Av = dico['Av']
        self.Av_cardel = dico['Av_cardelli']
        self.disp_matrix = dico['disp_matrix']
        self.slope = dico['reddening_law']
        self.CHI2 = dico['CHI2']
        self.trans = dico['delta_M_grey']
        self.corr_matrix = dico['corr_matrix']
        self.dico = dico

        floor_filter = np.array([False]*len(self.X))

        for Bin in range(len(self.X)):
            if self.X[Bin]>6360. and self.X[Bin]<6600.:
                floor_filter[Bin] = True
            else:
                continue
        self.floor_filter = floor_filter

    def plot_spectrum_corrected(self, No_corrected=True):

        Mag_all_sn = copy.deepcopy(self.Mag_no_corrected)
        Mag_all_sn_var = np.zeros_like(Mag_all_sn)
        wRMS = np.zeros(len(self.X))
        wRMS_no_correct = np.zeros(len(self.X))

        for Bin in range(len(self.X)):
            if self.Rv>0:
                Mag_all_sn[:,Bin]-=(np.dot(self.alpha[Bin],self.data.T))+self.trans+(self.Av*st.extinctionLaw(self.X[Bin],Rv=self.Rv))
            else:
                Mag_all_sn[:,Bin]-=(np.dot(self.alpha[Bin],self.data.T))+self.trans

            Mag_all_sn_var[:,Bin] = self.Y_err[:,Bin]**2 + self.Y_build_error[:,Bin]**2 + self.disp_matrix[Bin,Bin]

            wRMS[Bin] = st.comp_rms(Mag_all_sn[:,Bin]-self.M0[Bin], 'francis', err=False, variance=Mag_all_sn_var[:,Bin])
            wRMS_no_correct[Bin] = st.comp_rms(self.Mag_no_corrected[:,Bin] - np.average(self.Mag_no_corrected[:,Bin],weights=1./self.Y_err[:,Bin]**2), 
                                                  'francis', err=False, variance=self.Y_err[:,Bin]**2)
        
        self.MAG_CORRECTED= Mag_all_sn
        self.WRMS_CORR = wRMS
        self.WRMS_NO_CORR = wRMS_no_correct

        indice = np.linspace(0,len(self.Av)-1,len(self.Av)).astype(int)
        Av, indice = zip(*sorted(zip(self.Av, indice)))
        colors = plt.cm.coolwarm(self.Av)
        Grey_scatter = np.zeros(len(self.Av))
        
        fig = plt.figure(figsize=(12,12))
        plt.subplots_adjust(left=0.06, bottom=0.06, right=0.99, top=0.9,hspace=0.001)
        CST_MANU = 0
        for sn in range(len(self.sn_name)):
            plt.subplot(2,1,1)
            if No_corrected:
                if sn==0:
                    plt.plot(self.X,self.Mag_no_corrected[sn]-2.5,color=colors[sn],linewidth=3,zorder=self.Av[sn])
                else:
                    plt.plot(self.X,self.Mag_no_corrected[sn]-2.5,color=colors[sn],linewidth=3,zorder=self.Av[sn])

            if sn==0:
                plt.plot(self.X,Mag_all_sn[sn]-self.M0+np.mean(self.M0)+3.3,color=colors[sn],linewidth=3,alpha=0.5,zorder=self.Av[sn])
            else:
                plt.plot(self.X,Mag_all_sn[sn]-self.M0+np.mean(self.M0)+3.3,color=colors[sn],linewidth=3,alpha=0.5,zorder=self.Av[sn])

        plt.text(6500,-3.7,'Observed spectra',fontsize=20)
        plt.text(5000,2.5,'Corrected residuals ($q_1$, $q_2$, $q_3$, $A_{\lambda_0}$)',fontsize=20)
        go_log(ax=None, no_label=True)
        scat = plt.scatter(self.Av+2500,self.Av+2500,c=self.Av,cmap=plt.cm.coolwarm)
        ax_cbar1 = fig.add_axes([0.08, 0.92, 0.88, 0.025])
        plt.subplot(2,1,1)
        cb = plt.colorbar(scat, cax=ax_cbar1, orientation='horizontal')
        cb.set_label('$A_{\lambda_0}$',fontsize=20, labelpad=-67)
        plt.ylabel('Mag AB + cst',fontsize=20)
        plt.ylim(-5,7)
        plt.xticks([2500.,9500.],['toto','pouet'])
        plt.xlim(self.X[0]-60,self.X[-1]+60)
        plt.gca().invert_yaxis()
        plt.subplot(2,1,2)
        if No_corrected:
            plt.plot(self.X,wRMS_no_correct,'r',linewidth=3,label=r'Observed wRMS, average between $[6360\AA,6600\AA]$ = %.2f mag' %(np.mean(wRMS_no_correct[self.floor_filter])))

        plt.plot(self.X,wRMS,'b',linewidth=3,label=r'Corrected wRMS, average between $[6360\AA,6600\AA]$ = %.2f mag' %(np.mean(wRMS[self.floor_filter])))
                           
        plt.plot(self.X,np.zeros(len(self.X)),'k')
        plt.ylabel('wRMS (mag)',fontsize=20)
        plt.xlabel('wavelength [$\AA$]',fontsize=20)
        plt.xlim(self.X[0]-60,self.X[-1]+60)
        go_log(ax=None, no_label=False)
        plt.ylim(0.0,0.62)
        plt.legend()

    def plot_spectral_variability(self,name_fig=None):
       
        c=['r','b','g','k']
        NUMBER=['first','second','third']

        for correction in range(len(self.alpha[0])):

            plt.figure(figsize=(7,6))
            gs = gridspec.GridSpec(2, 1,height_ratios=[2,1])
            plt.subplots_adjust(left=0.13, bottom=0.11, right=0.99, top=0.95,hspace=0.001)

            plt.subplot(gs[0])

            y_moins = np.zeros(len(self.X))
            y_plus = np.zeros(len(self.X))
            
            for Bin in range(len(self.X)):
                y_moins[Bin]=self.M0[Bin]-self.alpha[Bin,correction]*(np.mean(self.data[:,correction])+np.sqrt(np.var(self.data[:,correction])))
                y_plus[Bin]=self.M0[Bin]+self.alpha[Bin,correction]*(np.mean(self.data[:,correction])+np.sqrt(np.var(self.data[:,correction])))

            plt.fill_between(self.X,self.M0,y_plus,color='k',alpha=0.8 )
            plt.fill_between(self.X,y_moins,self.M0,color='grey',alpha=0.7)
            p3 = plt.plot(self.X,self.M0,'k',linewidth=2,label='toto')
            plt.ylim(-0.45,1.59)
            plt.gca().invert_yaxis()
            plt.ylabel('$M_0(t=0,\lambda) +$ cst. (mag)',fontsize=16)
            plt.title(r'Average spectrum with $\pm$1$\sigma$ variation ($\alpha_{%i}(t=0,\lambda)$)'%(correction+1))
            p1 = plt.Rectangle((0, 0), 1, 1, fc="k")
            p2 = plt.Rectangle((0, 0), 1, 1, fc="grey")
            plt.legend([p1, p2], ['$+1 \sigma$', '$-1 \sigma$'])
            plt.xticks([2500.,9500.],['toto','pouet'])
            plt.xlim(self.X[0]-60,self.X[-1]+60)
            go_log(ax=None, no_label=True)

            plt.subplot(gs[1])

            mean_effect = np.std(self.data[:,correction])*np.mean(abs(self.alpha[:,correction]))
            print('Mean effect correction%i:'%(correction+1), mean_effect)
            plt.plot(self.X,self.alpha[:,correction],'k',linewidth=3)#,label=r'Average effect (mag)=%.3f'%((mean_effect)))
            plt.ylabel(r'$\alpha_{%i}(t=0,\lambda)$'%(correction+1),fontsize=16)
            plt.xlabel('wavelength [$\AA$]',fontsize=16)
            plt.xlim(self.X[0]-60,self.X[-1]+60)
            plt.gca().invert_yaxis()
            go_log(ax=None, no_label=False)

            if name_fig is not None:
                plt.savefig(name_fig[correction])

    def plot_bin_Av_slope(self, Bin):

        if type(Bin) == list:
            for bb in Bin:
                print(bb)
                self.plot_bin_Av_slope(bb)                
            return

        CCM = np.zeros(len(self.X))
        slopes_star = np.zeros(len(self.X))
        slopes_star_err = np.zeros(len(self.X))

        Mag_all_sn = self.Mag_no_corrected
        Mag_all_sn_err = self.Y_err        
       
        slopes = self.slope
        Av = self.Av
        M0 = self.M0
        alpha = self.alpha
        data = self.data
        trans = self.trans
        
        toto = 100.
        BIN = 0

        CCM31 = st.extinctionLaw(self.X,Rv=3.1)
        CCM26 = st.extinctionLaw(self.X,Rv=self.Rv)
        CCMplus = st.extinctionLaw(self.X,Rv=self.Rv+0.5)
        CCMminus = st.extinctionLaw(self.X,Rv=self.Rv-0.5)
        CCM14 = st.extinctionLaw(self.X,Rv=2.0)
        Ind_med = (len(self.X)/2)-1
        CCM31 /= CCM31[Ind_med]
        CCM26 /= CCM26[Ind_med]
        CCMplus /= CCMplus[Ind_med]
        CCMminus /= CCMminus[Ind_med]
        CCM14 /= CCM14[Ind_med]
        AVV = np.linspace(-0.7,1.3,20)       
        
        plt.figure(43,figsize=(11,11))
        plt.subplots_adjust(left=0.12, bottom=0.07, right=0.99, top=0.995)
        MAG = copy.deepcopy(Mag_all_sn)
        
        for sn in range(len(MAG[:,0])):
            for X_Bin in range(len(self.X)):
                for correction in range(len(self.alpha[0])):
                    MAG[sn,X_Bin]-=alpha[X_Bin,correction]*data[sn,correction]

                MAG[sn,X_Bin]-=trans[sn]+M0[X_Bin]
     
        self.MAG=MAG

        plt.subplot(2,1,2)

        plt.errorbar(Av,MAG[:,Bin],linestyle='',xerr=None,yerr=np.sqrt(self.Y_build_error[:,Bin]),ecolor='blue',alpha=0.7,marker='.',zorder=0)
        scat = plt.scatter(Av,MAG[:,Bin],zorder=100,s=50,c='b',edgecolors='k')
        plt.plot(AVV,slopes[Bin]*AVV,'r',label='$\gamma_{%i\AA}$'%(self.X[Bin]),lw=3)
        plt.plot(AVV,CCM26[Bin]*AVV,'b--',linewidth=3,label='CCM $(R_V=%.1f\pm0.5)$'%(self.Rv))
        plt.fill_between(AVV,CCMminus[Bin]*AVV,CCMplus[Bin]*AVV,color='b',alpha=0.3)
        plt.ylabel('$M(t=0,\lambda)-M_0(t=0,\lambda) - \sum_{i=1}^{i=3} \\alpha_i(0,\lambda) q_i$',fontsize=18)
        plt.xlabel('$A_{\lambda_0}$',fontsize=20)
        plt.text(-0.3,1.3,r'$\lambda=%i \AA$'%(self.X[Bin]),fontsize=20)
        plt.ylim(min(MAG[:,Bin])-0.3,max(MAG[:,Bin])+0.3)
        plt.xlim(-0.6,1.19)
        plt.legend(loc=4)

        plt.subplot(2,1,1)
        plt.plot(self.X,slopes,'r',label= '$\gamma_{\lambda}$',lw=3,zorder=0)
        plt.plot(self.X,CCM26,'b--',linewidth=3,label= 'CCM $(R_V=%.1f\pm0.5)$'%(self.Rv),zorder=0)
        plt.fill_between(self.X,CCMminus,CCMplus,color='b',alpha=0.3)
        plt.scatter(self.X[Bin],slopes[Bin],c='k',marker='o',s=200,zorder=10)
        plt.ylabel(r'$\gamma_{\lambda}$',fontsize=20)
        plt.xlabel('wavelength [$\AA$]',fontsize=20)
        plt.ylim(0.35,2.1)
        plt.xlim(self.X[0]-60,self.X[-1]+60)
        go_log(ax=None, no_label=False)
        plt.legend()

    def plot_corr_matrix(self):
        v = self.disp_matrix.diagonal()
        plot_disp_matrix(self.X, self.disp_matrix/np.sqrt(v*v[:,None]), np.sqrt(v),'$\\rho$', '$\\rho$', 
                         cmap=plt.cm.seismic, plotpoints=True, VM=[-1,1])


def run_at_max_plot(path_output='data_output/', path_output_plot='data_output/plot_output/'):
    
    amp = at_max_plot(path_output=path_output)
    name_fig = []
    for i in range(3):
        name_fig.append(os.path.join(path_output_plot, 'alpha%i.pdf'%(i+1)))
    amp.plot_spectral_variability(name_fig=name_fig)
    amp.plot_bin_Av_slope(42)
    plt.savefig(os.path.join(path_output_plot,'CCM_law_bin42.pdf'))
    amp.plot_spectrum_corrected()
    plt.savefig(os.path.join(path_output_plot,'all_spectrum_corrected_without_grey_with_3_eigenvector.pdf'))
    amp.plot_corr_matrix()
    plt.savefig(os.path.join(path_output_plot,'dispersion_matrix.pdf'))

if __name__=='__main__':

    import sugar_training as sugar
    #import os
    #path = os.path.dirname(sugar.__file__)
    
    #lst_dic=[]
    #for i in range(5):
    #    lst_dic.append('../sugar/data_output/sugar_paper_output/model_at_max_%i_eigenvector_without_grey.pkl'%(i+1))
    
    #plot_disp_eig(lst_dic)#,dic = 'disp_eig.pkl')
    #P.savefig('residual_emfa_vectors_at_max.pdf')
    #plot_vec_emfa(lst_dic,4,ALIGN=[-1,1,1,-1,-1])#,dic='vec_emfa_residual.pkl')
    #P.savefig('plot_paper/STD_choice_eigenvector.pdf')
    
    #SP=SUGAR_plot('../sugar_training/data_output/sugar_paper_output/model_at_max_3_eigenvector_without_grey_save_before_PCA.pkl')
    #SP.plot_spectral_variability(name_fig=['fig1.pdf','fig2.pdf','fig3.pdf'],ERROR=False)
    #SP.plot_bin_Av_slope(42)
    #P.savefig('CCM_law_bin42.pdf')#,transparent=True)
    #SP.plot_spectrum_corrected()
    #P.savefig('all_spectrum_corrected_without_grey_with_3_eigenvector.pdf')
    ##SP.plot_spectral_variability(name_fig=None)

