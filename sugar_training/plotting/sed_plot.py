import matplotlib.pyplot as P
import numpy as N
import cPickle
import scipy.interpolate as inter
from ToolBox import Astro
from ToolBox import Statistics 
import copy
from matplotlib.patches import Ellipse
from ToolBox.Signal import loess
from ToolBox.Plots import scatterPlot as SP
import matplotlib.gridspec as gridspec
from matplotlib.gridspec import GridSpec
from ToolBox import MPL
from matplotlib.widgets import Slider, Button, RadioButtons
import sys,os,optparse     
from scipy.stats import norm as NORMAL_LAW
import sugar_training as sugar
import cosmogp
import sncosmo

NBIN = 196
NPHASE = 21
PHASE_MIN = -12
PHASE_MAX = 48

def go_log(ax=None, no_label=False):
    import matplotlib.ticker
    if ax is None:
        ax = P.gca()
    ax.set_xscale('log')
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.get_xaxis().set_minor_formatter(matplotlib.ticker.ScalarFormatter())
    if no_label:
        P.minorticks_off()


def sigma_clip(dist, err, sigma=5):
    init_n = [len(dist)+1, len(dist)]
    count = 0
    while init_n[-1] != init_n[-2]:
        Filtre = (abs(dist)<sigma*N.std(dist))
        dist = dist[Filtre]
        err = err[Filtre]
        init_n.append(len(dist))
        count +=1
        if count> 20:
            break
    return dist, err

class gp_interp:

    def __init__(self,time,y):

        self.time = time
        self.y = y

        self.gp = cosmogp.gaussian_process(self.y, self.time, y_err=N.ones(len(self.y))*0.03,
                                           Mean_Y=self.y, diff=[0], Time_mean=self.time,
                                           kernel='RBF1D')
        self.gp.hyperparameters = [0.5,6.]
        
    def interpolate(self,Time):
        self.gp.get_prediction(new_binning=Time,
                               svd_method=False)
        
        return self.gp.Prediction[0]


def Compare_TO_SUGAR_parameter(emfa_pkl='../sugar/data_output/emfa_output.pkl',
                               SED_max='../sugar/data_output/sugar_model_2.60_Rv.pkl',
                               SUGAR_parameter_pkl='../sugar/data_output/sugar_parameters.pkl'):

    dic = cPickle.load(open(emfa_pkl))

    FILTRE=dic['filter']
    data=sugar.passage(dic['Norm_data'][FILTRE],dic['Norm_err'][FILTRE],dic['vec'],sub_space=3)
    error=sugar.passage_error(dic['Norm_err'][FILTRE],dic['vec'],3,return_std=True)


    dic_SUGAR=cPickle.load(open(SUGAR_parameter_pkl))
    dic_sed_max=cPickle.load(open(SED_max))
    Av_max=dic_sed_max['Av']

    data_SUGAR=N.zeros(N.shape(data))
    error_SUGAR=N.zeros(N.shape(error))
    Av_SUGAR=N.zeros(len(Av_max))
    sn_name=N.array(dic['sn_name'])[dic['filter']]


    for i in range(len(sn_name)):
        data_SUGAR[i,0]=dic_SUGAR[sn_name[i]]['q1']
        data_SUGAR[i,1]=dic_SUGAR[sn_name[i]]['q2']
        data_SUGAR[i,2]=dic_SUGAR[sn_name[i]]['q3']
        Av_SUGAR[i]=dic_SUGAR[sn_name[i]]['Av']
        error_SUGAR[i]=N.sqrt(N.diag(dic_SUGAR[sn_name[i]]['cov_q'][2:,2:]))

    P.figure(figsize=(6,12))
    P.subplots_adjust(left=0.15, bottom=0.05, right=0.99, top=0.98, wspace=0.2, hspace=0.35)
    for i in range(3):
        RHO=N.corrcoef(data[:,i],data_SUGAR[:,i])[0,1]
        P.subplot(4,1,i+1)
        P.scatter(data[:,i],data_SUGAR[:,i],label=r'$\rho=%.2f$'%(RHO),zorder=10)
        P.errorbar(data[:,i],data_SUGAR[:,i], linestyle='', xerr=error[:,i],yerr=error_SUGAR[:,i],ecolor='grey',alpha=0.7,marker='.',zorder=0)
        P.legend(loc=2)
        P.xlabel('$q_{%i}$ from spectral indicators @ max'%(i+1),fontsize=15)
        P.ylabel('$q_{%i}$ from SUGAR fitting'%(i+1),fontsize=15)
        xlim = P.xlim()
        P.plot(xlim,xlim,'r',linewidth=3,zorder=5)
        P.ylim(xlim[0],xlim[1])
        P.xlim(xlim[0],xlim[1])


    RHO=N.corrcoef(Av_max,Av_SUGAR)[0,1]
    P.subplot(4,1,4)
    P.scatter(Av_max,Av_SUGAR,label=r'$\rho=%.2f$'%(RHO),zorder=10)
    P.plot(Av_max,Av_max,'r',linewidth=3)
    P.legend(loc=2)
    P.xlabel('$A_{V}$ from SUGAR training',fontsize=15)
    P.ylabel('$A_{V}$ from SUGAR fitting',fontsize=15)
    xlim = P.xlim()
    P.plot(xlim,xlim,'r',linewidth=3,zorder=5)
    P.ylim(xlim[0],xlim[1])
    P.xlim(xlim[0],xlim[1])


def compare_sel_sucre(path_input = 'data_input/', path_output='data__output/', plot_slopes=False):

    lds = sugar.load_data_sugar(path_input = path_input, training = True, validation = True)
    lds.load_salt2_data()

    Filtre = N.array([True]*len(lds.X0)) 
    
    x0 = lds.X0
    x0_err = lds.X0_err
    x1 = lds.X1
    x1_err = lds.X1_err
    c = lds.C
    c_err = lds.C_err
    grey = N.zeros_like(x0)
    q1 = N.zeros_like(x0)
    q2 = N.zeros_like(x0)
    q3 = N.zeros_like(x0)
    av = N.zeros_like(x0)

    mu = N.zeros_like(x0)

    dico = cPickle.load(open(path_output+'sugar_parameters.pkl'))
    
    for i in range(len(lds.sn_name)):
        sn = lds.sn_name[i]
        if sn in dico.keys():
            grey[i] = dico[sn]['grey']
            q1[i] = dico[sn]['q1']
            q2[i] = dico[sn]['q2']
            q3[i] = dico[sn]['q3']
            av[i] = dico[sn]['Av']
            mu[i] = 5. * N.log10(sugar.cosmology.luminosity_distance(lds.zhelio[i],lds.zcmb[i])) - 5.
        else:
            Filtre[i] = False
    x0 = -2.5 * N.log10(x0[Filtre]) - mu[Filtre]
    x0 -= N.mean(x0)
    x0_err = -2.5 * N.log10(x0_err[Filtre])
    x1_err = x1_err[Filtre]
    c_err = c_err[Filtre]
    x1 = x1[Filtre]
    c = c[Filtre]
    grey = grey[Filtre]
    q1 = q1[Filtre]
    q2 = q2[Filtre]
    q3 = q3[Filtre]
    av = av[Filtre]

    param_lin_fit = N.array([grey,q1,q2,q3,av]).T
    ml_x1 = sugar.multilinearfit(param_lin_fit,x1,xerr=None,yerr=None,covx=None,Beta00=None)
    ml_c = sugar.multilinearfit(param_lin_fit,c,xerr=None,yerr=None,covx=None,Beta00=None)
    ml_x0 = sugar.multilinearfit(param_lin_fit,x0,xerr=None,yerr=None,covx=None,Beta00=None)

    ml_x1.Multilinearfit(adddisp=True)
    ml_c.Multilinearfit(adddisp=True)
    ml_x0.Multilinearfit(adddisp=True)

    hr = sugar.hubble_salt2(path_input=path_input, sn_name=lds.sn_name[Filtre])
    
    param_salt = [c,x1,hr.residu]
    param_salt_err = [c_err,x1_err,x0_err]
    param_salt_name = ['$C$','$X_1$','$\Delta \mu$ SALT2']
    param_sugar = [grey,q1,q2,q3,av]
    param_sugar_name = ['$\Delta M_{grey}$','$q_1$','$q_2$','$q_3$','$A_V$']

    param_slopes = [ml_c,ml_x1,ml_x0]
    x_axis_slopes = [N.linspace(N.min(grey),N.max(grey),10),N.linspace(N.min(q1),N.max(q1),10),
                     N.linspace(N.min(q2),N.max(q2),10),N.linspace(N.min(q3),N.max(q3),10),
                     N.linspace(N.min(av),N.max(av),10)]
    
    ind_salt = [0,0,0,0,0,1,1,1,1,1,2,2,2,2,2]
    ind_sugar = [0,1,2,3,4,0,1,2,3,4,0,1,2,3,4]

    sticks_salt = [N.linspace(-0.1,0.4,6),N.linspace(-2,2,5),N.linspace(-0.5,0.5,3)]
    sticks_sugar = [N.linspace(-0.4,0.4,5),N.linspace(-3,3,7),N.linspace(-4,4,5),N.linspace(-2,2,5),N.linspace(-0.5,1,4)]

    fig = P.figure(figsize=(14,8))
    P.subplots_adjust(left=0.08,top=0.97,right=0.90,wspace=0.,hspace=0.)
    cmap = P.cm.get_cmap('Blues',6)
    cmap.set_over('r')
    bounds = [0, 1, 2, 3, 4, 5]
    import matplotlib
    norm = matplotlib.colors.BoundaryNorm(bounds, cmap.N)
    
    for i in range(15):
        P.subplot(3,5,i+1)
        print(len(param_sugar[ind_sugar[i]]), len(param_salt[ind_salt[i]]))
        rho = N.corrcoef(param_sugar[ind_sugar[i]],param_salt[ind_salt[i]])[0,1]
        signi = Statistics.correlation_significance(abs(rho),len(param_sugar[ind_sugar[i]]),sigma=True)

        scat = P.scatter(param_sugar[ind_sugar[i]],param_salt[ind_salt[i]],s=50,cmap=cmap,
                         c=N.ones_like(param_sugar[ind_sugar[i]])*signi,vmin = 0, vmax = 6, edgecolors='k')
        print(param_sugar_name[ind_sugar[i]], param_salt_name[ind_salt[i]], rho, signi)
        if i+1 not in [1,6,11]:
            P.yticks(sticks_salt[ind_salt[i]],['']*len(sticks_salt[ind_salt[i]]))
        else:
            P.yticks(sticks_salt[ind_salt[i]])
            if i+1 ==11:
                P.ylabel(param_salt_name[ind_salt[i]],fontsize=16)
            else:
                P.ylabel(param_salt_name[ind_salt[i]],fontsize=20)

        if i+1 not in [11,12,13,14,15]:
            P.xticks(sticks_sugar[ind_sugar[i]],['']*len(sticks_sugar[ind_sugar[i]]))
        else:
            P.xticks(sticks_sugar[ind_sugar[i]])
            P.xlabel(param_sugar_name[ind_sugar[i]],fontsize=20)
        if plot_slopes:
            P.plot(x_axis_slopes[ind_sugar[i]],x_axis_slopes[ind_sugar[i]]*param_slopes[ind_salt[i]].alpha[ind_sugar[i]]+param_slopes[ind_salt[i]].M0)
                

    cbar_ax = fig.add_axes([0.93, 0.1, 0.02, 0.87])
    cb = matplotlib.colorbar.ColorbarBase(cbar_ax, cmap=cmap,
                                          norm=norm,
                                          boundaries=bounds+[9],
                                          extend='max',
                                          ticks=bounds,
                                          spacing='proportional')
                                            
    cb.set_label('Pearson correlation coefficient significance ($\sigma$)',fontsize=20)



def compare_host_sucre(SUGAR_parameter_pkl='../sugar/data_output/sugar_parameters.pkl', dic_host='../sugar/data_input/Host.pkl'):

    dico = cPickle.load(open(SUGAR_parameter_pkl))
    dic_host = cPickle.load(open(dic_host))
    sn_name = N.array(dico.keys())
    grey = N.zeros(len(sn_name))
    q1 = N.zeros(len(sn_name))
    q2 = N.zeros(len(sn_name))
    q3 = N.zeros(len(sn_name))
    av = N.zeros(len(sn_name))
    host = N.zeros(len(sn_name))

    for i in range(len(sn_name)):
        sn = sn_name[i]
        if sn in dico.keys():
            grey[i] = dico[sn]['grey']
            q1[i] = dico[sn]['q1']
            q2[i] = dico[sn]['q2']
            q3[i] = dico[sn]['q3']
            av[i] = dico[sn]['Av']
            host[i] = dic_host[sn]['mchost.mass']
    Filtre = (host != 0)
    grey = grey[Filtre]
    q1 = q1[Filtre]
    q2 = q2[Filtre]
    q3 = q3[Filtre]
    av = av[Filtre]
    host = host[Filtre]

    fig = P.figure(figsize=(14,6))
    ax = fig.add_axes([0.06,0.85,0.93,0.05])
    P.subplots_adjust(left=0.06,top=0.77,right=0.99,wspace=0.,hspace=0.)
    cmap = P.cm.get_cmap('Blues',6)
    cmap.set_over('r')
    bounds = [0, 1, 2, 3, 4, 5]
    import matplotlib
    norm = matplotlib.colors.BoundaryNorm(bounds, cmap.N)

    param_sugar = [q1, q2, q3, av]
    label_sugar = ['$q_1$', '$q_2$', '$q_3$', '$A_V$']
    ticks_sugar = [N.linspace(-4,3,8), N.linspace(-4,3,8), N.linspace(-2,2,5), N.linspace(-0.5,1,4)]
    for i in range(4):
        P.subplot(1,4,i+1)
        rho = N.corrcoef(host,param_sugar[i])[0,1]
        signi = Statistics.correlation_significance(abs(rho),len(param_sugar[i]),sigma=True)
        print(label_sugar[i], rho, signi)
        scat = P.scatter(param_sugar[i],host,s=50,cmap=cmap,
                         c=N.ones_like(param_sugar[i])*signi,vmin = 0, vmax = 6)
        if i == 0:
            P.ylabel('$\log(M/M_{\odot})$',fontsize=20)
        else:
            P.yticks([],[])
            
        P.xlabel(label_sugar[i], fontsize=20)
        P.xticks(ticks_sugar[i])
    #cbar_ax = fig.add_axes([0.93, 0.1, 0.02, 0.87])
    #cax, kw = matplotlib.colorbar.make_axes(ax, orientation='horizontal',
    #                                        pad=0.02)
    cb = matplotlib.colorbar.ColorbarBase(ax, cmap=cmap,
                                          norm=norm,
                                          boundaries=bounds+[9],
                                          extend='max',
                                          ticks=bounds,
                                          spacing='proportional',
                                          orientation='horizontal')
                                            
    cb.set_label('Pearson correlation coefficient significance ($\sigma$)',fontsize=20,labelpad=-70)



def plot_corr_sucre(SUGAR_parameter_pkl='../sugar/data_output/sugar_parameters.pkl'):

    dico = cPickle.load(open(SUGAR_parameter_pkl))

    x0 = len(dico.keys())
    sn_name = dico.keys()
    
    grey = N.zeros(x0)
    q1 = N.zeros(x0)
    q2 = N.zeros(x0)
    q3 = N.zeros(x0)
    av = N.zeros(x0)
    
    for i in range(len(sn_name)):
        sn = sn_name[i]
        grey[i] = dico[sn]['grey']
        q1[i] = dico[sn]['q1']
        q2[i] = dico[sn]['q2']
        q3[i] = dico[sn]['q3']
        av[i] = dico[sn]['Av']

    x_param = [grey,q1,q2,q3,av]
    y_param = [av,q3,q2,q1]

    x_ind = [0,1,2,3,4,
             0,1,2,3,4,
             0,1,2,3,4,
             0,1,2,3,4,
             0,1,2,3,4]
    
    y_ind =[None,None,None,None,None,
            3,3,3,3,3,
            2,2,2,2,2,
            1,1,1,1,1,
            0,0,0,0,0]
    
    x_name = ['$\Delta M_{grey}$','$q_1$','$q_2$','$q_3$','$A_V$']
    y_name = ['$A_V$','$q_3$','$q_2$','$q_1$']

    fig = P.figure(figsize=(10,10))
    P.subplots_adjust(left=0.08,hspace=0.,wspace=0.)
    cmap = P.cm.get_cmap('Blues',6)
    cmap.set_over('r')
    bounds = [0, 1, 2, 3, 4, 5]
    import matplotlib
    norm = matplotlib.colors.BoundaryNorm(bounds, cmap.N)


    for i in range(25):
        if i+1 in [2,3,4,5,8,9,10,14,15,20]:
            continue
        else:
            P.subplot(5,5,i+1)
            if i+1 in [1,7,13,19,25]:
                P.yticks([],[])
                P.hist(x_param[x_ind[i]])
                if i+1!=25:
                    P.xticks([],[])
            else:
                rho = N.corrcoef(x_param[x_ind[i]],y_param[y_ind[i]])[0,1]
                signi = Statistics.correlation_significance(abs(rho),len(y_param[y_ind[i]]),sigma=True)
                P.scatter(x_param[x_ind[i]],y_param[y_ind[i]],s=50,cmap=cmap,
                          c=N.ones_like(y_param[y_ind[i]])*signi,vmin = 0, vmax = 6)
                print(x_name[x_ind[i]] + ' ' + y_name[y_ind[i]], ' ', signi, ' ', rho)

            if i+1 not in [21,22,23,24,25]:
                P.xticks([],[])
            else:
                P.xlabel(x_name[x_ind[i]],fontsize=24)

            if i+1 not in [1,6,11,16,21]:
                P.yticks([],[])
            else:
                if i+1 !=1 :
                    P.ylabel(y_name[y_ind[i]],fontsize=24)

    cbar_ax = fig.add_axes([0.93, 0.1, 0.02, 0.87])
    cb = matplotlib.colorbar.ColorbarBase(cbar_ax, cmap=cmap,
                                          norm=norm,
                                          boundaries=bounds+[9],
                                          extend='max',
                                          ticks=bounds,
                                          spacing='proportional')

    cb.set_label('Pearson correlation coefficient significance ($\sigma$)',fontsize=20)
                

                    
class SUGAR_plot:

    def __init__(self,dico_hubble_fit):
                
        dico = cPickle.load(open(dico_hubble_fit))
        #self.key=dico['key']
        self.alpha=dico['alpha']
        self.M0=dico['m0']
        self.X=dico['X']
        self.data = dico['h']


    def plot_spectrophtometric_effec_time(self,comp=0):

        reorder = N.arange(NBIN*NPHASE).reshape(NBIN, NPHASE).T.reshape(-1)
        X=self.X[reorder]
        M0=self.M0[reorder]
        ALPHA=self.alpha[:,comp][reorder]
     
        CST=N.mean(M0)
        fig,ax1=P.subplots(figsize=(10,12))
        P.subplots_adjust(left=0.1, right=0.9,bottom=0.07,top=0.99)
        Time=N.linspace(PHASE_MIN,PHASE_MAX,NPHASE)
        Y2_label=[]
        Y2_pos=[]
        for i in range(NPHASE):
            
            if i%2==0:
                if (PHASE_MIN+(3*i))!=0:
                    Y2_label.append('%i days'%(Time[i]))
                else:
                    Y2_label.append('%i day'%(Time[i]))

                Y2_pos.append(M0[i*NBIN:(i+1)*NBIN][-1]-CST)

                ax1.plot(X[i*NBIN:(i+1)*NBIN],M0[i*NBIN:(i+1)*NBIN]-CST,'k')

                
                y_moins=M0[i*NBIN:(i+1)*NBIN]-ALPHA[i*NBIN:(i+1)*NBIN]*(N.mean(self.data[:,comp])+N.sqrt(N.var(self.data[:,comp])))
                y_plus=M0[i*NBIN:(i+1)*NBIN]+ALPHA[i*NBIN:(i+1)*NBIN]*(N.mean(self.data[:,comp])+N.sqrt(N.var(self.data[:,comp])))

                ax1.fill_between(X[i*NBIN:(i+1)*NBIN],M0[i*NBIN:(i+1)*NBIN]-CST,y_plus-CST,color='k',alpha=0.8 )
                ax1.fill_between(X[i*NBIN:(i+1)*NBIN],M0[i*NBIN:(i+1)*NBIN]-CST,y_moins-CST,color='grey',alpha=0.7)

                if i==0:
                    CST-=1.7
                else:
                    CST-=1


        ax1.set_ylabel('$M_0(t,\lambda)$ + Cst.',fontsize=20)
        ax1.set_xlim(3300,8620)
        ax1.set_ylim(-1,14.8)
        ax1.set_xlabel('wavelength $[\AA]$',fontsize=20)
        p1 = P.Rectangle((0, 0), 1, 1, fc="k")
        p2 = P.Rectangle((0, 0), 1, 1, fc="grey")
        ax1.legend([p1, p2], ['$+1\sigma_{q_%i}$'%(comp+1), '$-1\sigma_{q_%i}$'%(comp+1)],loc=4)
        P.gca().invert_yaxis()
        go_log(ax=ax1, no_label=False)

        ax2=ax1.twinx()
        ax2.set_ylim(-1,14.8)
        ax2.yaxis.set_ticks(Y2_pos)
        ax2.yaxis.set_ticklabels(Y2_label,fontsize=15)
        ax2.set_ylim(ax2.get_ylim()[::-1])

class residual_plot:

    def __init__(self,DIC_spectra,SUGAR_model,Dic_sugar_parameter, dic_salt=None, Rv=None):

        self.dic=cPickle.load(open(DIC_spectra))
        self.sn_name=self.dic.keys()
        self.dicS=cPickle.load(open(Dic_sugar_parameter))
        if dic_salt is not None :
            self.dicSA=cPickle.load(open(dic_salt))
        else:
            self.dicSA=None

	SUGAR=cPickle.load(open(SUGAR_model))
	self.M0=SUGAR['m0']
	self.alpha=SUGAR['alpha']

        if Rv is None:
            self.Rv=SUGAR['Rv']
        else:
            self.Rv=Rv

    def plot_spectra_reconstruct(self,sn,T_min=PHASE_MIN,T_max=PHASE_MAX):
        
        fig,ax1=P.subplots(figsize=(10,8))#, frameon=False)
        P.subplots_adjust(left=0.08, right=0.88,bottom=0.09,top=0.95)
        IND=None
        for i,SN in enumerate(self.sn_name):
            if SN==sn:
                IND=i

        OFFSET=[]
        Off=0
        Phase=[]
        Time=N.linspace(PHASE_MIN,PHASE_MAX,NPHASE)
        DAYS=[-999]
        Y2_pos=[]
        Y2_label=[]
        for j in range(len(self.dic[sn].keys())):
            days=self.dic[sn]['%i'%(j)]['phase_salt2']

            if days>T_min and days<T_max:
                DAYS.append(days)
                if '%10.1f'%DAYS[-2]!='%10.1f'%DAYS[-1]:
                    OFFSET.append(Off+0.2-N.mean(self.dic[sn]['0']['Y']))
                    if Off==0:
                        ax1.plot(self.dic[sn]['%i'%(j)]['X'],self.dic[sn]['%i'%(j)]['Y']+OFFSET[Off],'k',
                                 label='Observed spectra')
                        if self.dicSA:
                            YS = Astro.Coords.flbda2ABmag(self.dicSA[sn]['%i'%(j)]['X'],self.dicSA[sn]['%i'%(j)]['sal2_sed_flux'])
                            YS+=-5.*N.log10(sugar.cosmology.luminosity_distance(self.dicSA[sn]['%i'%(j)]['z_helio'],self.dicSA[sn]['%i'%(j)]['z_cmb']))+5.
                            ax1.plot(self.dic[sn]['%i'%(j)]['X'],YS+OFFSET[Off],'r',lw=2,label='SALT2.4')
                    else:
                        ax1.plot(self.dic[sn]['%i'%(j)]['X'],self.dic[sn]['%i'%(j)]['Y']+OFFSET[Off],'k')
                        if self.dicSA:
                            YS=Astro.Coords.flbda2ABmag(self.dicSA[sn]['%i'%(j)]['X'],self.dicSA[sn]['%i'%(j)]['sal2_sed_flux'])
                            YS+=-5.*N.log10(sugar.cosmology.luminosity_distance(self.dicSA[sn]['%i'%(j)]['z_helio'],self.dicSA[sn]['%i'%(j)]['z_cmb']))+5.
                            ax1.plot(self.dicSA[sn]['%i'%(j)]['X'],YS+OFFSET[Off],'r',lw=2)

                    moins=self.dic[sn]['%i'%(j)]['Y']+OFFSET[Off]-N.sqrt(self.dic[sn]['%i'%(j)]['V'])
                    plus=self.dic[sn]['%i'%(j)]['Y']+OFFSET[Off]+N.sqrt(self.dic[sn]['%i'%(j)]['V'])
                    ax1.fill_between(self.dic[sn]['%i'%(j)]['X'],moins,plus,color='k',alpha=0.5 )
                    Y2_pos.append(self.dic[sn]['%i'%(j)]['Y'][-1]+OFFSET[Off])
                    if abs(self.dic[sn]['%i'%(j)]['phase_salt2'])<2:
                        Y2_label.append('%.1f day'%(self.dic[sn]['%i'%(j)]['phase_salt2']))
                    else:
                        Y2_label.append('%.1f days'%(self.dic[sn]['%i'%(j)]['phase_salt2']))
                    
                    Phase.append(self.dic[sn]['%i'%(j)]['phase_salt2'])
                    Off+=1


        Phase=N.array(Phase)
    

        Reconstruction=N.zeros((len(Phase),NBIN))
        for Bin in range(NBIN):
            SPLINE_Mean=inter.InterpolatedUnivariateSpline(Time,self.M0[Bin*NPHASE:(Bin+1)*NPHASE])
            Reconstruction[:,Bin]+=SPLINE_Mean(Phase)
            for i in range(len(self.alpha[0])):
                SPLINE=inter.InterpolatedUnivariateSpline(Time,self.alpha[:,i][Bin*NPHASE:(Bin+1)*NPHASE])
            
                Reconstruction[:,Bin]+=self.dicS[sn]['q%i'%(i+1)]*SPLINE(Phase)

            Reconstruction[:,Bin]+=self.dicS[sn]['grey']
            Reconstruction[:,Bin]+=self.dicS[sn]['Av']*sugar.extinctionLaw(self.dic[sn]['0']['X'][Bin],
                                                                           Rv=self.Rv)

        MIN=10**25
        MAX=-10**25

        for j in range(len(Reconstruction[:,0])):
            if j == 0:
                ax1.plot(self.dic[sn]['%i'%(j)]['X'],Reconstruction[j]+OFFSET[j],'b',label='SUGAR model',lw=2)
            else:
                ax1.plot(self.dic[sn]['%i'%(j)]['X'],Reconstruction[j]+OFFSET[j],'b',lw=2)

            if N.min(Reconstruction[j]+OFFSET[j])<MIN:
                MIN=N.min(Reconstruction[j]+OFFSET[j])
                
            if N.max(Reconstruction[j]+OFFSET[j])>MAX:
                MAX=N.max(Reconstruction[j]+OFFSET[j])

        P.title('SN-training-4',fontsize=16)
        ax1.set_xlim(3300,8600)
        ax1.set_ylim(MIN-0.5,MAX+2.5)
        ax1.set_ylabel(r'Mag AB $(t,\lambda)$ +Cst.',fontsize=20)
        ax1.set_xlabel('wavelength $[\AA]$',fontsize=20)
        leg = ax1.legend(loc=4)
        go_log(ax=ax1, no_label=False)
        P.gca().invert_yaxis()
        #leg.get_frame().set_alpha(0)
        
        #P.gca().patch.set_alpha(0)
        ax2=ax1.twinx()
        ax2.set_ylim(MIN-0.5,MAX+2.5)
        ax2.yaxis.set_ticks(Y2_pos)
        ax2.yaxis.set_ticklabels(Y2_label)
        ax2.set_ylim(ax2.get_ylim()[::-1])
        #P.savefig('../../Desktop/PTF09.pdf', transparent=True)


    def plot_spectra_movie(self,sn,T_min=PHASE_MIN,T_max=PHASE_MAX):

        def go_to_flux(X,Y,ABmag0=48.59):
            Flux_nu=10**(-0.4*(Y+ABmag0))
            f = X**2 / 299792458. * 1.e-10
            Flux=Flux_nu/f
            return Flux

        Time = N.linspace(PHASE_MIN,PHASE_MAX,NPHASE)

        phaseee = N.array([self.dic[sn]['%i'%(j)]['phase_salt2'] for j in range(len(self.dic[sn].keys()))])
        if N.min(phaseee)<PHASE_MIN:
            minp = PHASE_MIN
        else:
            minp =  N.min(phaseee)
        if N.max(phaseee)>PHASE_MAX:
            maxp = PHASE_MAX
        else:
            maxp = N.max(phaseee)
                            
        time_movie = N.linspace(int(minp),int(maxp),200)

        IND=None
        for i,SN in enumerate(self.sn_name):
            if SN==sn:
                IND=i

        Phase = []
        DAYS = [-999]

        salt = []
        data = []
        
        for j in range(len(self.dic[sn].keys())):
            days=self.dic[sn]['%i'%(j)]['phase_salt2']
            if days>T_min and days<T_max:
                DAYS.append(days)
                if '%10.1f'%DAYS[-2]!='%10.1f'%DAYS[-1]:                    
                    wave_data = self.dic[sn]['%i'%(j)]['X']
                    Phase.append(self.dic[sn]['%i'%(j)]['phase_salt2'])

        data = N.zeros((NPHASE,NBIN))
        gp_data = N.loadtxt('../sugar/data_output/gaussian_process_greg/gp_predict/' + sn + '.predict')
        wave_data = N.zeros(NBIN)
        for Bin in range(NBIN):
            data[:,Bin] = gp_data[:,2][Bin*NPHASE:(Bin+1)*NPHASE]
            wave_data[Bin] = gp_data[:,1][Bin*NPHASE]
                    
        Phase=N.array(Phase)
        salt = N.array(salt)

        SUGAR = N.zeros((len(time_movie),NBIN))
        DATA = N.zeros((len(time_movie),NBIN))
        SALT = N.zeros((len(time_movie),NBIN))

        for Bin in range(NBIN):
            SPLINE_Mean=inter.InterpolatedUnivariateSpline(Time,self.M0[Bin*NPHASE:(Bin+1)*NPHASE])
            SUGAR[:,Bin]+=SPLINE_Mean(time_movie)
            for i in range(len(self.alpha[0])):
                SPLINE=inter.InterpolatedUnivariateSpline(Time,self.alpha[:,i][Bin*NPHASE:(Bin+1)*NPHASE])
            
                SUGAR[:,Bin]+=self.dicS[sn]['q%i'%(i+1)]*SPLINE(time_movie)

            SUGAR[:,Bin]+=self.dicS[sn]['grey']
            SUGAR[:,Bin]+=self.dicS[sn]['Av']*sugar.extinctionLaw(self.dic[sn]['0']['X'][Bin],
                                                                  Rv=self.Rv)
            SUGAR[:,Bin] = go_to_flux(self.dic[sn]['0']['X'][Bin],SUGAR[:,Bin])

        for Bin in range(NBIN):
            SPLINE=inter.InterpolatedUnivariateSpline(Time,data[:,Bin])
            DATA[:,Bin]+=SPLINE(time_movie)
            DATA[:,Bin] = go_to_flux(self.dic[sn]['0']['X'][Bin],DATA[:,Bin])

        source_salt24 = sncosmo.SALT2Source(modeldir='../../2-4-0/data/salt2-4/')
        model = sncosmo.Model(source=source_salt24)
        meta = cPickle.load(open('../sugar/data_input/META_JLA.pkl'))

        for T in range(len(time_movie)):
            model.set(z=meta[sn]['host.zhelio'], t0=0.,
                      x0=meta[sn]['salt2.X0'],
                      x1=meta[sn]['salt2.X1'],
                      c=meta[sn]['salt2.Color'])
            wave = copy.deepcopy(wave_data)*(1+meta[sn]['host.zhelio'])
            YS = model.flux(time_movie[T],wave)*(1+meta[sn]['host.zhelio'])**3
            YS = Astro.Coords.flbda2ABmag(wave_data,YS)
            YS+=-5.*N.log10(sugar.cosmology.luminosity_distance(self.dicSA[sn]['%i'%(j)]['z_helio'],self.dicSA[sn]['%i'%(j)]['z_cmb']))+5.
            SALT[T] = go_to_flux(wave_data,YS)

        import matplotlib
        import matplotlib.pyplot as plt
        import matplotlib.animation as manimation

        fig = plt.figure(figsize=(12,8))
        plt.subplots_adjust(left=0.07,top=0.99,right=0.99,hspace=0,wspace=0)

        FFMpegWriter = manimation.writers['ffmpeg']
        metadata = dict(title=sn, artist='Matplotlib',
                        comment='sugar model')
        writer = FFMpegWriter(fps=15, metadata=metadata, bitrate=5000)
        Name_mp4=sn + ".mp4"

        MIN = N.min(DATA)
        MAX = N.max(DATA)

        MMAX = N.max(DATA-SALT)
        MMIN = N.min(DATA-SALT)

        with writer.saving(fig, Name_mp4, 200):
            for i in range(len(time_movie)):
                print(i)
                plt.subplot(2,1,1)
                plt.cla()
                plt.plot(wave_data,DATA[i],'k',lw=3,label=sn)
                plt.plot(wave_data,SALT[i],'r',lw=3,label='SALT2.4')
                plt.plot(wave_data,SUGAR[i],'b',lw=3,label='SUGAR')
                if abs(time_movie[i])>2:
                    plt.text(5700,0.23,'%.2f days'%(time_movie[i]),fontsize=18)
                else:
                    plt.text(5700,0.23,'%.2f day'%(time_movie[i]),fontsize=18)
                plt.xlim(3500,8500)
                plt.xticks([],[])
                plt.ylabel('Flux [erg s$^{-1}$ cm$^{-2}$ $\AA^{-1}$]',fontsize=18)
                plt.ylim(MIN,MAX)
                plt.legend()
                #plt.gca().invert_yaxis()
                plt.subplot(2,1,2)
                plt.cla()
                plt.plot(wave_data,DATA[i]-DATA[i],'k',lw=3)
                plt.plot(wave_data,abs(DATA[i]-SUGAR[i]),'b',lw=3)
                plt.plot(wave_data,abs(DATA[i]-SALT[i]),'r',lw=3)
                plt.ylim(0.,0.053)
                plt.xlim(3500,8500)
                plt.ylabel('|residual|',fontsize=20)
                plt.xlabel(r'wavelength $[\AA]$', fontsize=20)
                writer.grab_frame()
                

    def plot_spectra_reconstruct_residuals(self,sn,T_min=PHASE_MIN,T_max=PHASE_MAX):

        fig,ax1=P.subplots(figsize=(10,8))
        P.subplots_adjust(left=0.08, right=0.88,bottom=0.09,top=0.95)

        RESIDUAL_SUGAR = []
        RESIDUAL_SUGAR_1 = []
        RESIDUAL_SUGAR_2 = []
        RESIDUAL_SALT2 = []
        ERROR_SPECTRA = []

        IND=None
        for i,SN in enumerate(self.sn_name):
            if SN==sn:
                IND=i

        OFFSET=[]
        Off=0
        Phase=[]
        Time=N.linspace(PHASE_MIN,PHASE_MAX,NPHASE)
        DAYS=[-999]
        JJ=[]
        Y2_pos=[]
        Y2_label=[]
        for j in range(len(self.dic[sn].keys())):
            days=self.dic[sn]['%i'%(j)]['phase_salt2']

            if days>T_min and days<T_max:
                DAYS.append(days)
                if '%10.1f'%DAYS[-2]!='%10.1f'%DAYS[-1]:
                    JJ.append(j)
                    OFFSET.append(Off+0.2)
                    if Off==0:
                        ax1.plot(self.dic[sn]['%i'%(j)]['X'],N.zeros(len(self.dic[sn]['%i'%(j)]['X']))+OFFSET[Off],
                                 'k',label='Observed spectra')
                        if self.dicSA:
                            try:
                                YS=Astro.Coords.flbda2ABmag(self.dicSA[sn]['%i'%(j)]['X'],self.dicSA[sn]['%i'%(j)]['sal2_sed_flux'])
                            except:
                                YS=Astro.Coords.flbda2ABmag(self.dicSA[sn]['%i'%(j)]['X'],self.dicSA[sn]['%i'%(j)]['PF_flux'])
                            YS+=-5.*N.log10(sugar.cosmology.luminosity_distance(self.dicSA[sn]['%i'%(j)]['z_helio'],self.dicSA[sn]['%i'%(j)]['z_cmb']))+5.
                            if not N.isfinite(N.sum(YS)):
                                SPLINE=inter.InterpolatedUnivariateSpline(self.dicSA[sn]['%i'%(j)]['X'][N.isfinite(YS)],
                                                                          YS[N.isfinite(YS)])
                                YS[~N.isfinite(YS)]=SPLINE(self.dicSA[sn]['%i'%(j)]['X'][~N.isfinite(YS)])
                            RESIDUAL_SALT2.append(self.dic[sn]['%i'%(j)]['Y']-YS)
                            ax1.plot(self.dic[sn]['%i'%(j)]['X'],RESIDUAL_SALT2[Off]+OFFSET[Off],'r',lw=2,label='SALT2.4')
                    else:
                        ax1.plot(self.dic[sn]['%i'%(j)]['X'],N.zeros(len(self.dic[sn]['%i'%(j)]['X']))+OFFSET[Off],'k')
                        if self.dicSA:
                            try:
                                YS=Astro.Coords.flbda2ABmag(self.dicSA[sn]['%i'%(j)]['X'],self.dicSA[sn]['%i'%(j)]['sal2_sed_flux'])
                            except:
                                YS=Astro.Coords.flbda2ABmag(self.dicSA[sn]['%i'%(j)]['X'],self.dicSA[sn]['%i'%(j)]['PF_flux'])
                            YS+=-5.*N.log10(sugar.cosmology.luminosity_distance(self.dicSA[sn]['%i'%(j)]['z_helio'],self.dicSA[sn]['%i'%(j)]['z_cmb']))+5.
                            if not N.isfinite(N.sum(YS)):
                                SPLINE=inter.InterpolatedUnivariateSpline(self.dicSA[sn]['%i'%(j)]['X'][N.isfinite(YS)],
                                                                          YS[N.isfinite(YS)])
                                YS[~N.isfinite(YS)]=SPLINE(self.dicSA[sn]['%i'%(j)]['X'][~N.isfinite(YS)])
                            
                            RESIDUAL_SALT2.append(self.dic[sn]['%i'%(j)]['Y']-YS)
                            ax1.plot(self.dic[sn]['%i'%(j)]['X'],RESIDUAL_SALT2[Off]+OFFSET[Off],'r',lw=2)
                            
                    moins=OFFSET[Off]-N.sqrt(self.dic[sn]['%i'%(j)]['V'])
                    plus=OFFSET[Off]+N.sqrt(self.dic[sn]['%i'%(j)]['V'])
                    ax1.fill_between(self.dic[sn]['%i'%(j)]['X'],moins,plus,color='k',alpha=0.5 )
                    Y2_pos.append(OFFSET[Off])
                    if abs(self.dic[sn]['%i'%(j)]['phase_salt2'])<2:
                        Y2_label.append('%.1f day'%(self.dic[sn]['%i'%(j)]['phase_salt2']))
                    else:
                        Y2_label.append('%.1f days'%(self.dic[sn]['%i'%(j)]['phase_salt2']))
                    Phase.append(self.dic[sn]['%i'%(j)]['phase_salt2'])
                    Off+=1
                    

                    ERROR_SPECTRA.append(N.sqrt(self.dic[sn]['%i'%(j)]['V']))

        Phase=N.array(Phase)

        Reconstruction=N.zeros((len(Phase),NBIN))
        rec1 = N.zeros((len(Phase),NBIN))
        rec2 = N.zeros((len(Phase),NBIN))
        for Bin in range(NBIN):
            SPLINE_Mean=inter.InterpolatedUnivariateSpline(Time,self.M0[Bin*NPHASE:(Bin+1)*NPHASE])
            Reconstruction[:,Bin]+=SPLINE_Mean(Phase)
            rec1[:,Bin]+=SPLINE_Mean(Phase)
            rec2[:,Bin]+=SPLINE_Mean(Phase)
            for i in range(len(self.alpha[0])):
                SPLINE=inter.InterpolatedUnivariateSpline(Time,self.alpha[:,i][Bin*NPHASE:(Bin+1)*NPHASE])
                Reconstruction[:,Bin]+=self.dicS[sn]['q%i'%(i+1)]*SPLINE(Phase)
                if i==0:
                    rec1[:,Bin]+=self.dicS[sn]['q%i'%(i+1)]*SPLINE(Phase)
                    rec2[:,Bin]+=self.dicS[sn]['q%i'%(i+1)]*SPLINE(Phase)
                if i==1:
                    rec2[:,Bin]+=self.dicS[sn]['q%i'%(i+1)]*SPLINE(Phase)
                    

            Reconstruction[:,Bin]+=self.dicS[sn]['grey'] - 19.2
            rec1[:,Bin] += self.dicS[sn]['grey'] - 19.2
            rec2[:,Bin] += self.dicS[sn]['grey'] - 19.2
            
            Reconstruction[:,Bin]+=self.dicS[sn]['Av']*sugar.extinctionLaw(self.dic[sn]['0']['X'][Bin],
                                                                           Rv=self.Rv)
            rec1[:,Bin] += self.dicS[sn]['Av']*sugar.extinctionLaw(self.dic[sn]['0']['X'][Bin],
                                                                   Rv=self.Rv)
            rec2[:,Bin] += self.dicS[sn]['Av']*sugar.extinctionLaw(self.dic[sn]['0']['X'][Bin],
                                                            Rv=self.Rv)
        MIN=10**25
        MAX=-10**25
                
        for j in range(len(Reconstruction[:,0])):
            if j == 0:
                ax1.plot(self.dic[sn]['%i'%(j)]['X'],
                         Reconstruction[j]+OFFSET[j]-self.dic[sn]['%i'%(JJ[j])]['Y'],'b',label='SUGAR model',lw=2)
            else:
                ax1.plot(self.dic[sn]['%i'%(j)]['X'],
                         Reconstruction[j]+OFFSET[j]-self.dic[sn]['%i'%(JJ[j])]['Y'],'b',lw=2)

            RESIDUAL_SUGAR.append(Reconstruction[j]-self.dic[sn]['%i'%(JJ[j])]['Y'])
            RESIDUAL_SUGAR_1.append(rec1[j]-self.dic[sn]['%i'%(JJ[j])]['Y'])
            RESIDUAL_SUGAR_2.append(rec2[j]-self.dic[sn]['%i'%(JJ[j])]['Y'])
                            
        
        MIN=N.min(OFFSET[0])
        MAX=N.max(OFFSET[-1])

        P.title('SN-training-4 residual',fontsize=16)
        ax1.set_xlim(3300,8600)
        ax1.set_ylim(MIN-0.6,MAX+3.5)
        ax1.set_ylabel(r'Mag AB $(t,\lambda)$ + Cst.',fontsize=20)
        ax1.set_xlabel('wavelength $[\AA]$',fontsize=20)
        ax1.legend(loc=4)
        go_log(ax=ax1, no_label=False)
        P.gca().invert_yaxis()

        ax2=ax1.twinx()
        ax2.set_ylim(MIN-0.5,MAX+3.5)
        ax2.yaxis.set_ticks(Y2_pos)
        ax2.yaxis.set_ticklabels(Y2_label)
        ax2.set_ylim(ax2.get_ylim()[::-1])

        return RESIDUAL_SUGAR, RESIDUAL_SUGAR_1, RESIDUAL_SUGAR_2, RESIDUAL_SALT2, ERROR_SPECTRA, Phase



def wRMS_sed_sugar_salt(rp, training=True, validation=False, sn_list=[]):

    """
    rp = residual_plot('../sugar/data_input/spectra_snia.pkl',
                       '../sugar/data_output/sugar_model.pkl',
                       '../sugar/data_output/sugar_parameters.pkl',
                       '../sugar/data_output/sugar_paper_output/model_at_max_3_eigenvector_without_grey_save_before_PCA.pkl',
                       dic_salt = '../sugar/data_input/file_pf_bis.pkl')
    """

    residual_sel = []
    residual_sucre = []
    residual_sucre_1 = []
    residual_sucre_2 = []    
    residual_error = []
    nspectra = 0

    SN = N.array(rp.dicS.keys())
    SAMPLE = N.array([rp.dic[sn]['0']['sample'] for sn in SN])

    if training and validation:
        FILTER = N.array([True]*len(SN))
    else:
        if training:
            FILTER = (SAMPLE == 'training')
        if validation:
            FILTER = (SAMPLE == 'validation')

    for sn in range(len(SN)):
        if SN[sn] in sn_list:
            FILTER[sn] = False
    print(len(SN))
    SN = SN[FILTER]
    print(len(SN))
        
    i=0
    for sn in SN:
        sucre, sucre_1, sucre_2, sel, res_error, phase = rp.plot_spectra_reconstruct_residuals(sn,T_min=-12,T_max=48)
        residual_sel.append(sel)
        residual_sucre.append(sucre)
        residual_sucre_1.append(sucre_1)
        residual_sucre_2.append(sucre_2)
        residual_error.append(res_error)
        nspectra += len(N.array(residual_sucre[i])[:,0])
        i+=1
    P.close('all')

    res_sel = N.zeros((nspectra,NBIN))
    res_sucre = N.zeros((nspectra,NBIN))
    res_sucre_1 = N.zeros((nspectra,NBIN))
    res_sucre_2 = N.zeros((nspectra,NBIN))
    res_err = N.zeros((nspectra,NBIN))
    t = 0
    for i in range(len(SN)):
        for j in range(len(residual_sel[i])):
            res_sucre[t] = residual_sucre[i][j]
            res_sucre_1[t] = residual_sucre_1[i][j]
            res_sucre_2[t] = residual_sucre_2[i][j]
            res_sel[t] = residual_sel[i][j]
            res_err[t] = residual_error[i][j]
            t+=1


    wave = rp.dic['PTF09dnl']['0']['X']

    wrms_sel = N.zeros(len(wave))
    wrms_sucre = N.zeros(len(wave))
    wrms_sucre_1 = N.zeros(len(wave))
    wrms_sucre_2 = N.zeros(len(wave))
    
    for Bin in range(len(wave)):
        
        residual_clipped, error_clipped = sigma_clip(res_sel[:, Bin], res_err[:, Bin])
        wrms_sel[Bin] = N.sqrt(N.sum((residual_clipped**2)/(error_clipped**2)) / N.sum(1./(error_clipped**2)))

        residual_clipped, error_clipped = sigma_clip(res_sucre[:, Bin], res_err[:, Bin])
        wrms_sucre[Bin] = N.sqrt(N.sum((residual_clipped**2)/(error_clipped**2)) / N.sum(1./(error_clipped**2)))

        residual_clipped, error_clipped = sigma_clip(res_sucre_1[:, Bin], res_err[:, Bin])
        wrms_sucre_1[Bin] = N.sqrt(N.sum((residual_clipped**2)/(error_clipped**2)) / N.sum(1./(error_clipped**2)))

        residual_clipped, error_clipped = sigma_clip(res_sucre_2[:, Bin], res_err[:, Bin])
        wrms_sucre_2[Bin] = N.sqrt(N.sum((residual_clipped**2)/(error_clipped**2)) / N.sum(1./(error_clipped**2)))

    err_rms = N.zeros_like(wave)
    for i in range(len(wave)):
        loc = sugar.biweight_M(res_err[i])
        err_rms[i] = sugar.biweight_S(res_err[i])
        err_rms[i] = loc
 

    P.figure(figsize=(16,6))#, frameon=False)
    P.subplots_adjust(top=0.9,right=0.65,left=0.07,bottom=0.15)
    P.plot(wave,wrms_sel,'r',linewidth=5,label='SALT2.4 ($X_0$ + $X_1$ + $C$)')
    P.plot(wave,wrms_sucre_1,'b-.',linewidth=3,label='SUGAR ($\Delta M_{grey}$ + $q_1$ + $A_V$)')
    P.plot(wave,wrms_sucre_2,'b--',linewidth=3,label='SUGAR ($\Delta M_{grey}$ +$q_1$ + $q_2$ + $A_V$)')
    P.plot(wave,wrms_sucre,'b',linewidth=5,label='SUGAR ($\Delta M_{grey}$ +$q_1$ + $q_2$ + $q_3$ + $A_V$)')
    #P.plot(wave,N.ones_like(wave)*N.mean(err_rms),'k--',linewidth=1,label='average spectral noise')
    P.xlabel('wavelength $[\AA]$',fontsize=20)
    P.ylabel('wRMS (mag)',fontsize=20)
    P.ylim(0,0.43)
    P.xlim(3200,8700)
    leg = P.legend(bbox_to_anchor=(1.01, 0.7), loc=2, borderaxespad=0.,fontsize=14)
    go_log(ax=None, no_label=False)

    dic = {'X':wave,
           'salt2':wrms_sel,
           'sugar1':wrms_sucre_1,
           'sugar2':wrms_sucre_2,
           'sugar':wrms_sucre}
    FILE = open('for_cnrs.pkl','w')
    cPickle.dump(dic,FILE)
    FILE.close()
    #leg.get_frame().set_alpha(0)
    #P.gca().patch.set_alpha(0)
    #P.savefig('../../Desktop/test.pdf', transparent=True)

def wRMS_sed_time_sugar_salt(rp, training=True, validation=False, sn_list=[]):
    """
    dic = cPickle.load(open('../sugar/data_output/sugar_parameters.pkl'))
    
    rp = residual_plot('../sugar/data_input/spectra_snia.pkl',
                       '../sugar/data_output/sugar_model.pkl',
                       '../sugar/data_output/sugar_parameters.pkl',
                       '../sugar/data_output/sugar_paper_output/model_at_max_3_eigenvector_without_grey_save_before_PCA.pkl',
                       dic_salt = '../sugar/data_input/file_pf_bis.pkl')
    """
       
    residual_sel = []
    residual_sucre = []
    residual_sucre_1 = []
    residual_sucre_2 = []    
    residual_error = []
    phase = [] 
    nspectra = 0

    SN = N.array(rp.dicS.keys())
    try:
        SAMPLE = N.array([rp.dic[sn]['0']['sample'] for sn in SN])
    except:
        print('continue')

    if training and validation:
        FILTER = N.array([True]*len(SN))
    else:
        try:
            if training:
                FILTER = (SAMPLE == 'training')
            if validation:
                FILTER = (SAMPLE == 'validation')
        except:
            print('continue')

    for sn in range(len(SN)):
        if SN[sn] in sn_list:
            FILTER[sn] = False
    print(len(SN))
    SN = SN[FILTER]
    print(len(SN))

    i=0
    for sn in SN:
        sucre, sucre_1, sucre_2, sel, res_error, phas = rp.plot_spectra_reconstruct_residuals(sn,T_min=-12,T_max=48)
        residual_sel.append(sel)
        residual_sucre.append(sucre)
        residual_sucre_1.append(sucre_1)
        residual_sucre_2.append(sucre_2)
        residual_error.append(res_error)
        phase.append(phas)
        nspectra += len(N.array(residual_sucre[i])[:,0])
        i+=1
    P.close('all')
    
    binning_phase = N.linspace(-12,48,21)

    res_sel = []
    res_sucre = []
    res_sucre_1 = []
    res_sucre_2 = []
    res_err = []

    wave = rp.dic['PTF09dnl']['0']['X']
        
    for Bin in range(21):
        print(Bin)
        res_sel.append([])
        res_sucre.append([])
        res_sucre_1.append([])
        res_sucre_2.append([])
        res_err.append([])
        for sn in range(len(phase)):
            for ph in range(len(phase[sn])):
                if abs(phase[sn][ph]-binning_phase[Bin])<1.5:
                    [res_sel[Bin].append(residual_sel[sn][ph][wav]) for wav in range(len(wave))]
                    [res_sucre[Bin].append(residual_sucre[sn][ph][wav]) for wav in range(len(wave))]
                    [res_sucre_1[Bin].append(residual_sucre_1[sn][ph][wav]) for wav in range(len(wave))]
                    [res_sucre_2[Bin].append(residual_sucre_2[sn][ph][wav]) for wav in range(len(wave))]
                    [res_err[Bin].append(residual_error[sn][ph][wav]) for wav in range(len(wave))]

    wrms_sel = N.zeros(21)
    wrms_sucre = N.zeros(21)
    wrms_sucre_1 = N.zeros(21)
    wrms_sucre_2 = N.zeros(21)

    for Bin in range(21):
        
        res_clipped, err_clipped = sigma_clip(N.array(res_sel[Bin]), N.array(res_err[Bin]))
        wrms_sel[Bin] = N.sqrt(N.sum((res_clipped**2)/(err_clipped**2)) / N.sum(1./(err_clipped**2)))

        res_clipped, err_clipped = sigma_clip(N.array(res_sucre[Bin]), N.array(res_err[Bin]))
        wrms_sucre[Bin] = N.sqrt(N.sum((res_clipped**2)/(err_clipped**2)) / N.sum(1./(err_clipped**2)))

        res_clipped, err_clipped = sigma_clip(N.array(res_sucre_1[Bin]), N.array(res_err[Bin]))
        wrms_sucre_1[Bin] = N.sqrt(N.sum((res_clipped**2)/(err_clipped**2)) / N.sum(1./(err_clipped**2)))

        res_clipped, err_clipped = sigma_clip(N.array(res_sucre_2[Bin]), N.array(res_err[Bin]))
        wrms_sucre_2[Bin] = N.sqrt(N.sum((res_clipped**2)/(err_clipped**2)) / N.sum(1./(err_clipped**2)))
        
    P.figure(figsize=(16,6))
    P.subplots_adjust(top=0.9,right=0.65,left=0.05,bottom=0.15)
    P.plot(binning_phase,wrms_sel,'r',linewidth=5,label='SALT2.4 ($X_0$ + $X_1$ + $C$)')
    P.plot(binning_phase,wrms_sucre_1,'b-.',linewidth=3,label='SUGAR ($\Delta M_{grey}$ + $q_1$ + $A_V$)')
    P.plot(binning_phase,wrms_sucre_2,'b--',linewidth=3,label='SUGAR ($\Delta M_{grey}$ +$q_1$ + $q_2$ + $A_V$)')
    P.plot(binning_phase,wrms_sucre,'b',linewidth=5,label='SUGAR ($\Delta M_{grey}$ +$q_1$ + $q_2$ + $q_3$ + $A_V$)')
    P.xlabel('Phase (day)',fontsize=20)
    P.ylabel('wRMS (mag)',fontsize=20)
    P.ylim(0,0.43)
    P.xlim(-13,50)
    P.legend(bbox_to_anchor=(1.01, 0.7), loc=2, borderaxespad=0.,fontsize=16)
    return res_sel
    #P.show()


def wRMS_sed_sugar_sugar_plus(WRMS_pkl=None):

    if WRMS_pkl is None: 
        dic = cPickle.load(open('../sugar/data_output/sugar_parameters.pkl'))
    
        rp3 = residual_plot('../sugar/data_input/spectra_snia.pkl',
                            '../sugar/data_output/sugar_model.pkl',
                            '../sugar/data_output/sugar_parameters.pkl',
                            '../sugar/data_output/sugar_paper_output/model_at_max_3_eigenvector_without_grey_save_before_PCA.pkl',
                            dic_salt = '../sugar/data_input/file_pf_bis.pkl')

        rp4 = residual_plot('../sugar/data_input/spectra_snia.pkl',
                            '../sugar/data_output/sugar_model_4.pkl',
                            '../sugar/data_output/sugar_parameters_4.pkl',
                            '../sugar/data_output/sugar_paper_output/model_at_max_3_eigenvector_without_grey_save_before_PCA.pkl',
                            dic_salt = '../sugar/data_input/file_pf_bis.pkl')

        rp5 = residual_plot('../sugar/data_input/spectra_snia.pkl',
                            '../sugar/data_output/sugar_model_5.pkl',
                            '../sugar/data_output/sugar_parameters_5.pkl',
                            '../sugar/data_output/sugar_paper_output/model_at_max_3_eigenvector_without_grey_save_before_PCA.pkl',
                            dic_salt = '../sugar/data_input/file_pf_bis.pkl')

        residual_sel = []
        residual_sucre = []
        residual_sucre_1 = []
        residual_sucre_2 = []    
        residual_error = []
        nspectra = 0

        i=0
        for sn in dic.keys():
            sucre3, sucre_1, sucre_2, sel, res_error, phase = rp3.plot_spectra_reconstruct_residuals(sn,T_min=-12,T_max=48)
            sucre4, sucre_1, sucre_2, sel, res_error, phase = rp4.plot_spectra_reconstruct_residuals(sn,T_min=-12,T_max=48)
            sucre5, sucre_1, sucre_2, sel, res_error, phase = rp5.plot_spectra_reconstruct_residuals(sn,T_min=-12,T_max=48)
            residual_sucre.append(sucre3)
            residual_sucre_1.append(sucre4)
            residual_sucre_2.append(sucre5)
            residual_error.append(res_error)
            nspectra += len(N.array(residual_sucre[i])[:,0])
            i+=1
        P.close('all')

        res_sucre = N.zeros((nspectra,NBIN))
        res_sucre_1 = N.zeros((nspectra,NBIN))
        res_sucre_2 = N.zeros((nspectra,NBIN))
        res_err = N.zeros((nspectra,NBIN))
        t = 0
        for i in range(len(dic.keys())):
            for j in range(len(residual_sucre[i])):
                res_sucre[t] = residual_sucre[i][j]
                res_sucre_1[t] = residual_sucre_1[i][j]
                res_sucre_2[t] = residual_sucre_2[i][j]
                res_err[t] = residual_error[i][j]
                t+=1

        dic = cPickle.load(open('../sugar/data_input/spectra_snia.pkl'))
        wave = dic['PTF09dnl']['0']['X']

        wrms_sucre = N.sqrt(N.sum((res_sucre**2)/(res_err**2),axis=0) / N.sum(1./(res_err**2),axis=0))
        wrms_sucre_1 = N.sqrt(N.sum((res_sucre_1**2)/(res_err**2),axis=0) / N.sum(1./(res_err**2),axis=0))
        wrms_sucre_2 = N.sqrt(N.sum((res_sucre_2**2)/(res_err**2),axis=0) / N.sum(1./(res_err**2),axis=0))

        dic = {'wave':wave,
               'wrms_sucre':wrms_sucre,
               'wrms_sucre_1':wrms_sucre_1,
               'wrms_sucre_2':wrms_sucre_2}

        File=open('wrms_4_5.pkl','w')
        cPickle.dump(dic,File)
        File.close()

    else:
        dic = cPickle.load(open(WRMS_pkl))
        wave = dic['wave']
        wrms_sucre = dic['wrms_sucre']
        wrms_sucre_1 = dic['wrms_sucre_1']
        wrms_sucre_2 = dic['wrms_sucre_2']
                                                                

    P.figure(figsize=(14,6))
    P.subplots_adjust(top=0.98,right=0.99,left=0.07,bottom=0.12)
    P.plot(wave,wrms_sucre,'b',linewidth=3,label='SUGAR model ($\Delta M_{grey}$ +$q_1$ + $q_2$ + $q_3$ + $A_V$)')
    P.plot(wave,wrms_sucre_1,'b-.',linewidth=3,label='SUGAR model + $q_4$')
    P.plot(wave,wrms_sucre_2,'b--',linewidth=3,label='SUGAR model + $q_4$ + $q_5$')

    P.xlabel('wavelength $[\AA]$',fontsize=20)
    P.ylabel('wRMS (mag)',fontsize=20)
    P.ylim(0,0.23)
    P.xlim(3200,8700)
    #P.legend(bbox_to_anchor=(1.01, 0.7), loc=2, borderaxespad=0.,fontsize=16)
    P.legend(loc=1, fontsize=16)
    go_log()
    P.show()

def wRMS_sed_sugar_sugar_Rv(WRMS_pkl=None):

    if WRMS_pkl is None: 
        dic = cPickle.load(open('../sugar/data_output/sugar_parameters.pkl'))
    
        rp3 = residual_plot('../sugar/data_input/spectra_snia.pkl',
                            '../sugar/data_output/sugar_model.pkl',
                            '../sugar/data_output/sugar_parameters.pkl',
                            '../sugar/data_output/sugar_paper_output/model_at_max_3_eigenvector_without_grey_save_before_PCA.pkl',
                            dic_salt = '../sugar/data_input/file_pf_bis.pkl')

        rp4 = residual_plot('../sugar/data_input/spectra_snia.pkl',
                            '../sugar/data_output/sugar_model_2.10_Rv.pkl',
                            '../sugar/data_output/sugar_parameters_2.10_Rv.pkl',
                            '../sugar/data_output/sugar_paper_output/model_at_max_3_eigenvector_without_grey_save_before_PCA.pkl',
                            dic_salt = '../sugar/data_input/file_pf_bis.pkl', Rv=2.1)

        rp5 = residual_plot('../sugar/data_input/spectra_snia.pkl',
                            '../sugar/data_output/sugar_model_3.10_Rv.pkl',
                            '../sugar/data_output/sugar_parameters_3.10_Rv.pkl',
                            '../sugar/data_output/sugar_paper_output/model_at_max_3_eigenvector_without_grey_save_before_PCA.pkl',
                            dic_salt = '../sugar/data_input/file_pf_bis.pkl', Rv=3.1)

        residual_sel = []
        residual_sucre = []
        residual_sucre_1 = []
        residual_sucre_2 = []    
        residual_error = []
        nspectra = 0

        i=0
        for sn in dic.keys():
            sucre3, sucre_1, sucre_2, sel, res_error, phase = rp3.plot_spectra_reconstruct_residuals(sn,T_min=-12,T_max=48)
            sucre4, sucre_1, sucre_2, sel, res_error, phase = rp4.plot_spectra_reconstruct_residuals(sn,T_min=-12,T_max=48)
            sucre5, sucre_1, sucre_2, sel, res_error, phase = rp5.plot_spectra_reconstruct_residuals(sn,T_min=-12,T_max=48)
            residual_sucre.append(sucre3)
            residual_sucre_1.append(sucre4)
            residual_sucre_2.append(sucre5)
            residual_error.append(res_error)
            nspectra += len(N.array(residual_sucre[i])[:,0])
            i+=1
        P.close('all')

        res_sucre = N.zeros((nspectra,NBIN))
        res_sucre_1 = N.zeros((nspectra,NBIN))
        res_sucre_2 = N.zeros((nspectra,NBIN))
        res_err = N.zeros((nspectra,NBIN))
        t = 0
        for i in range(len(dic.keys())):
            for j in range(len(residual_sucre[i])):
                res_sucre[t] = residual_sucre[i][j]
                res_sucre_1[t] = residual_sucre_1[i][j]
                res_sucre_2[t] = residual_sucre_2[i][j]
                res_err[t] = residual_error[i][j]
                t+=1

        dic = cPickle.load(open('../sugar/data_input/spectra_snia.pkl'))
        wave = dic['PTF09dnl']['0']['X']

        wrms_sucre = N.sqrt(N.sum((res_sucre**2)/(res_err**2),axis=0) / N.sum(1./(res_err**2),axis=0))
        wrms_sucre_1 = N.sqrt(N.sum((res_sucre_1**2)/(res_err**2),axis=0) / N.sum(1./(res_err**2),axis=0))
        wrms_sucre_2 = N.sqrt(N.sum((res_sucre_2**2)/(res_err**2),axis=0) / N.sum(1./(res_err**2),axis=0))

        dic = {'wave':wave,
               'wrms_sucre':wrms_sucre,
               'wrms_sucre_1':wrms_sucre_1,
               'wrms_sucre_2':wrms_sucre_2}

        File=open('wrms_rv.pkl','w')
        cPickle.dump(dic,File)
        File.close()

    else:
        dic = cPickle.load(open(WRMS_pkl))
        wave = dic['wave']
        wrms_sucre = dic['wrms_sucre']
        wrms_sucre_1 = dic['wrms_sucre_1']
        wrms_sucre_2 = dic['wrms_sucre_2']
                                                                

    P.figure(figsize=(8,6))
    P.subplots_adjust(top=0.98,right=0.99,left=0.1,bottom=0.12)
    P.plot(wave,wrms_sucre,'k',linewidth=6,label='SUGAR model ($R_V = 2.6$)')
    P.plot(wave,wrms_sucre_1,'r',linewidth=4,label='SUGAR model ($R_V = 2.1$)')
    P.plot(wave,wrms_sucre_2,'b',linewidth=2,label='SUGAR model ($R_V = 3.1$)')

    P.xlabel('wavelength $[\AA]$',fontsize=20)
    P.ylabel('wRMS (mag)',fontsize=20)
    P.ylim(0,0.23)
    P.xlim(3200,8700)
    #P.legend(bbox_to_anchor=(1.01, 0.7), loc=2, borderaxespad=0.,fontsize=16)
    P.legend(loc=1, fontsize=16)
    P.show()


def plot_M0_time(sugar_asci, comp = 0):

    sugar = N.loadtxt(sugar_asci)
    YLABEL = ['$M_0(\lambda)$ + cst','$\\alpha_1(\lambda)$', '$\\alpha_2(\lambda)$','$\\alpha_3(\lambda)$']
    CST = [19, 0, 0, 0]
    reorder = N.arange(NBIN*21).reshape(NBIN, 21).T.reshape(-1)
    X=sugar[:,1][reorder]
    M0=sugar[:,2+comp][reorder]
    P.figure(figsize=(12,10))
    P.subplots_adjust(right=0.99, top=0.95)
    compt=0
    XLAB=[1,2,3,4]
    YLAB=[2,4,6]

    for i in range(19):
        if (-12+(3*i))%12==0 and compt<7:
            P.subplot(3,2,compt+1)
            print(compt + 1)
            print((-12+(3*i)))
            print('')
            P.plot(X[i*NBIN:(i+1)*NBIN],M0[i*NBIN:(i+1)*NBIN]+CST[comp],'k', lw=3)
            if (-12+(3*i))!=0:
                P.title('Phase = %i days'%((-12+(3*i))),fontsize=20)
            else:
                P.title('Phase = %i days'%((-12+(3*i))),fontsize=20)
            if compt+1 in XLAB:
                P.xticks([2500.,9500.],['toto','pouet'])
            else:
                P.xlabel('wavelength $[\AA]$', fontsize=20)
            if compt+1 in YLAB:
                    print('prout')
            else:
                P.ylabel(YLABEL[comp],fontsize=20)
            P.xlim(3200,8800)
            P.gca().invert_yaxis()
            compt+=1
        if i==18:
            P.subplot(3,2,compt+1)
            P.plot(X[i*NBIN:(i+1)*NBIN],M0[i*NBIN:(i+1)*NBIN]+CST[comp],'k',lw=3)
            P.title('Phase = 42 days',fontsize=20)
            P.xlabel('wavelength $[\AA]$',fontsize=20)
            P.xlim(3200,8800)
            P.gca().invert_yaxis()
            compt+=1

def wRMS_sed_sugar_salt_pedagogique(WRMS):

    dic = cPickle.load(open(WRMS))
    wave = dic['wave']
    wrms_sel = dic['wrms_sel']
    wrms_sucre = dic['wrms_sucre']
    XXX = N.random.normal(size=100)
    
    fig = P.figure(figsize=(16,6))
    P.subplots_adjust(top=0.98,right=0.7,left=0.05,bottom=0.15)
    P.scatter(XXX+200000, XXX+200000, c=XXX)
    #0.125
    ax_cbar1 = fig.add_axes([0.05, 0.1, 0.65, 0.05])
    P.subplot(111)
    cb = P.colorbar(cax = ax_cbar1, orientation='horizontal')
    cb.set_ticks([])
    cb.set_label('supernovae color', fontsize=20)
    P.plot(wave,wrms_sel,'r',linewidth=5,label='SALT model')
    P.plot(wave,wrms_sucre,'b',linewidth=5,label='SUGAR model')
    #P.xlabel('wavelength $[\AA]$',fontsize=20)
    P.xticks([],[])
    P.ylabel('model residuals dispersion (%)',fontsize=20)
    P.yticks([0,0.1,0.2,0.3,0.4],['0%','10%','20%','30%','40%'])
    P.ylim(0,0.43)
    P.xlim(3200,8700)
    P.legend(bbox_to_anchor=(1.01, 0.7), loc=2, borderaxespad=0.,fontsize=16)
    P.show()
    
    

if __name__=='__main__':

    #path = '../../Desktop/post_doc/juillet_aout_2018/sugar_output_tv_like_before_moreval_data/'

    #compare_sel_sucre(path_input = path + 'data_input/', path_output = path + 'data_output/')
    
    #wRMS_sed_sugar_salt_pedagogique('wrms.pkl')
    
    #compare_host_sucre(SUGAR_parameter_pkl='../sugar/data_output/sugar_parameters.pkl', dic_host='../sugar/data_input/Host.pkl')
    wRMS_sed_sugar_sugar_plus(WRMS_pkl='wrms_4_5.pkl')
    #wRMS_sed_sugar_sugar_Rv(WRMS_pkl='wrms_rv.pkl')
    
    #plot_M0_time('../sugar_training/data_output/SUGAR_model_v1.asci', comp = 0)
    #plot_M0_time('../sugar_training/data_output/SUGAR_model_v1.asci', comp = 1)
    #plot_M0_time('../sugar_training/data_output/SUGAR_model_v1.asci', comp = 2)
    #plot_M0_time('../sugar_training/data_output/SUGAR_model_v1.asci', comp = 3)

    #compare_sel_sucre(plot_slopes=False)
    
    ##plot_corr_sucre()
    
    #wRMS_sed_sugar_salt(WRMS='wrms.pkl')
    #P.show()
    #wRMS_sed_time_sugar_salt(WRMS=None)#'wrms_time.pkl')
    
    #SED=SUGAR_plot('../sugar_training/data_output/sugar_model.pkl')
    #SED.plot_spectrophtometric_effec_time(comp=0)
    #P.savefig('alpha1.pdf')
    #SED.plot_spectrophtometric_effec_time(comp=1)
    #P.savefig('alpha2.pdf')
    #SED.plot_spectrophtometric_effec_time(comp=2)
    #P.savefig('alpha3.pdf')

    #Compare_TO_SUGAR_parameter()
    #P.savefig('max_vs_sugar.pdf')
    #dic = cPickle.load(open('../sugar/data_output/sugar_parameters.pkl'))
    
    #rp = residual_plot('../sugar/data_input/spectra_snia.pkl',
    #                   '../sugar/data_output/sugar_model.pkl',
    #                   '../sugar/data_output/sugar_parameters.pkl',
    #                   '../sugar/data_output/sugar_paper_output/model_at_max_3_eigenvector_without_grey_save_before_PCA.pkl',
    #                   dic_salt = '../sugar/data_input/file_pf_bis.pkl')

    
    #sn = 'PTF09dnl'
    #sn = 'SNF20080514-002'
    #sn = 'SN2008ec'
    #sn = 'SN2006X'
    #rp.plot_spectra_movie(sn)
    #sn = 'SN2012cu'
    #for sn in dic.keys():
    #rp.plot_spectra_movie(sn)
    #sn = 'SN2007kk'
    #rp.plot_spectra_reconstruct(sn,T_min=-8,T_max=29.)
    #rp.plot_spectra_reconstruct_residuals
    #P.savefig('plot_paper/reconstruct/'+sn+'.pdf')
    #sucre, sucre1, sucre2,  sel, res_error = rp.plot_spectra_reconstruct_residuals(sn,T_min=-5,T_max=28)
    #P.savefig('plot_paper/reconstruct/'+sn+'_residual.pdf')
