"""plot output from gaussian process for sugar paper."""

import numpy as np
import pylab as plt
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

def plot_line_positon(ylim):
    """
    Draw position of SNIa line at maximum.
    """
    #Ca H&K
    plt.fill_between([3580,3880],[ylim[0],ylim[0]], [ylim[1],ylim[1]],color='k',alpha=0.1)
    #Si II 4131
    plt.fill_between([3940,4060],[ylim[0],ylim[0]], [ylim[1],ylim[1]],color='k',alpha=0.1)
    #Mg II
    plt.fill_between([4120,4420],[ylim[0],ylim[0]], [ylim[1],ylim[1]],color='k',alpha=0.1)
    #Fe 4800
    plt.fill_between([4480,5137],[ylim[0],ylim[0]], [ylim[1],ylim[1]],color='k',alpha=0.1)
    #S II W
    plt.fill_between([5197,5560],[ylim[0],ylim[0]], [ylim[1],ylim[1]],color='k',alpha=0.1)
    #Si II 5972
    plt.fill_between([5620,5902],[ylim[0],ylim[0]], [ylim[1],ylim[1]],color='k',alpha=0.1)
    #Si II 6355
    plt.fill_between([5962,6277],[ylim[0],ylim[0]], [ylim[1],ylim[1]],color='k',alpha=0.1)
    #O I
    plt.fill_between([7205,7840],[ylim[0],ylim[0]], [ylim[1],ylim[1]],color='k',alpha=0.1)
    #Ca II IR
    plt.fill_between([7880,8530],[ylim[0],ylim[0]], [ylim[1],ylim[1]],color='k',alpha=0.1)

def plot_gp_output(plot_mean=False, path_gp = 'data_output/gaussian_process/', path_output_plot= 'data_output/'):
    """
    plot gp output. 
    
    kernel amplitude
    correlation length
    pull mean
    pull std of residual
    """
    try: gp_output = np.loadtxt(path_gp+'gp_info.dat',comments='#')
    except ValueError:
        "%s does not exist"%(path_gp+'gp_info.dat')

    if plot_mean:
        add_subplot = 1
    else:
        add_subplot = 0

    xlim = [3300,8600]
        
    plt.figure(figsize=(12,10))
    plt.subplots_adjust(top=0.85,bottom=0.08,left=0.08,right=0.99,hspace=0.0)
    
    plt.subplot(3+add_subplot,1,1)
    
    ylim = plt.ylim(0,0.7)
    plot_line_positon(ylim)
        
    plt.plot(gp_output[:,0],gp_output[:,1],'b',linewidth=3)


    #plt.xticks([],[])
    nsil = [r'Ca II H&K', r'Si II $\lambda$4131', r'Mg II',
            r'Fe $\lambda$4800', r'S II W', r'Si II $\lambda$5972',
            r'Si II $\lambda$6355', r'O I $\lambda$7773', r'Ca II IR']
    pos = [3730.0,4000.0,4270.0,4808.5,5378.5,
           5761.0,6119.5,7522.5,8205.0]
    
    go_log(ax=None, no_label=True)
    plt.gca().xaxis.tick_top()
    plt.xticks(pos,nsil,fontsize=20,rotation=45)
    plt.yticks(np.linspace(0.1,0.6,6))    
    plt.xlim(xlim[0],xlim[1])
    plt.ylabel('$\sigma(\lambda)$ (mag)',fontsize=20)
    
    plt.subplot(3+add_subplot,1,2)

    ylim=plt.ylim(0,13)
    plot_line_positon(ylim)
    
    plt.plot(gp_output[:,0],gp_output[:,2],'b',linewidth=3)

    plt.xticks([],[])
    plt.yticks(np.linspace(2,12,6))
    go_log(ax=None, no_label=True)
    plt.xlim(xlim[0],xlim[1])
    plt.ylabel('$l(\lambda)$ (days)',fontsize=20)
    
    if plot_mean:
        plt.subplot(3+add_subplot,1,3)

        ylim=plt.ylim(-0.2,0.18)
        plot_line_positon(ylim)
    
        plt.plot(xlim,np.zeros_like(xlim),'k-.')
        plt.plot(gp_output[:,0],gp_output[:,3],'b',linewidth=3)
        err = gp_output[:,4]/np.sqrt(gp_output[:,5])
        plt.fill_between(gp_output[:,0],gp_output[:,3]-err,gp_output[:,3]+err,color='b',alpha=0.3)

        plt.xticks([],[])
        go_log(ax=None, no_label=True)
        plt.yticks(np.linspace(-0.15,0.15,7))
        plt.xlim(xlim[0],xlim[1])
        plt.ylabel('$<\mathrm{pull}(\lambda)>$',fontsize=20)

    plt.subplot(3+add_subplot,1,3+add_subplot)

    ylim=plt.ylim(0.0,1.1)
    plot_line_positon(ylim)
    
    plt.plot(xlim,np.ones_like(xlim),'k-.')
    plt.plot(gp_output[:,0],gp_output[:,4],'b',linewidth=3)
    err = gp_output[:,4]/np.sqrt(2.*(gp_output[:,5]-1.))
    plt.fill_between(gp_output[:,0],gp_output[:,4]-err,gp_output[:,4]+err,color='b',alpha=0.3)
    print('mean STD of PULL : %f'%(np.mean(gp_output[:,4])))
    plt.xlim(xlim[0],xlim[1])
    plt.yticks(np.linspace(0.0,1.0,6))
    go_log(ax=None, no_label=False)
    plt.xlabel('wavelength [$\AA$]',fontsize=20)
    plt.ylabel('$\mathrm{STD}[\mathrm{pull}(\lambda)]$',fontsize=20)
    plt.savefig(os.path.join(path_output_plot, 'gaussian_processes.pdf'))
    
if __name__=='__main__':

    plot_gp_output()

