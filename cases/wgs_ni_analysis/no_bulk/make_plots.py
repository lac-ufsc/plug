import pickle
import numpy as np
import os

#Data folder
path = ('/home/tpcarvalho/carva/python_data/wgs_ni')

#Get all filenames with .pickle extension
filenames = os.listdir(path)
filenames = [k for k in filenames if '.pickle' in k]

#Create empty dict
data_dict = {}

for i in range(len(filenames)):
    #Open each file
    with open(os.path.join(path,filenames[i]), 'rb') as handle:
        #Get pickled data
        pickle_data = pickle.load(handle)
    
    #Fill dictionary with data
    data_dict[filenames[i].replace('.pickle','')] = pickle_data
    
#Export figure? 
exp_fig = 0
figfolder = '/home/tpcarvalho/carva/python_data/wgs_ni/'
figname = figfolder+'wgs_xg.eps'

#Import paper experimental data from CSV file
import csv
exp_data=[]
filename = ('/home/tpcarvalho/carva/python_data/wgs_ni/wheeler_co_conv_ni.csv') 
with open(filename, 'rt') as f:
    reader = csv.reader(f)
    for row in reader:
        exp_data.append(np.array(row,dtype=float))
f.close()
#Convert list to Numpy array  
exp_data = np.array(exp_data)
    
#Get simulation data from dict
wgs_nib_full = data_dict['wgs_nib_full']
wgs_nib_full_sa = data_dict['wgs_nib_full_sa']
wgs_nib_redux = data_dict['wgs_nib_redux']
wgs_nib_redux_sa = data_dict['wgs_nib_redux_sa']
wgs_nib_redux_incomps = data_dict['wgs_nib_redux_incomps']
wgs_nib_redux_mdots = data_dict['wgs_nib_redux_mdots']
wgs_nib_simple_fixed = data_dict['wgs_nib_simple_fixed']
wgs_nib_simple = data_dict['wgs_nib_simple']
wgs_nib_simple_incomps = data_dict['wgs_nib_simple_incomps']
wgs_nib_simple_mdots = data_dict['wgs_nib_simple_mdots']

vis=11

if vis!=0:
    #Plot some results
    import matplotlib.pyplot as plt
    #Set figure size and position   
    fig = plt.figure()
    fig.set_size_inches(11, 8)
    fig.canvas.manager.window.move(200,100)  
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.sans-serif'] = ['Computer Modern Roman']
    plt.rcParams.update({'font.size': 25,'legend.fontsize': 22})  

    if vis==1:
        
        #Get data variables
        CO_conv = wgs_nib_full['co_conv']
        R8_conv = wgs_nib_full['qf_co'][:,0]
        R9_conv = wgs_nib_full['qf_co'][:,1]
        Tout = wgs_nib_full['Tout']
        
        #CO consumption path analysis
        ax = fig.add_subplot(111)  
        ax.plot(Tout, CO_conv, '--k', linewidth=3, label='Conversion')
        ax.plot(Tout, R8_conv, '-r', linewidth=3, label='R8')
        ax.plot(Tout, R9_conv, '-b', linewidth=3, label='R9') 
        ax.set_xlabel('Temperature [$^\circ$C]')
        ax.set_ylabel('$RP_{CO,j}$ [%]')
        ax.legend(loc='best')
        ax.legend(frameon=False)
        
    if vis==2:
        from labellines import labelLines
        
        #Coverages with temperature   
        cov_out = np.array(wgs_nib_full['covs'])[:,-1,:]
     
        ax = fig.add_subplot(111)
        ax.semilogy(wgs_nib_full['Tout'], cov_out[:, 0], '-k', linewidth=3, label='Ni$^*$')
        ax.semilogy(wgs_nib_full['Tout'], cov_out[:, 1], '-k', linewidth=3,  label='H$^*$')
        ax.semilogy(wgs_nib_full['Tout'], cov_out[:, 4], '-k', linewidth=3,  label='H2O$^*$')
        ax.semilogy(wgs_nib_full['Tout'], cov_out[:, 3], '-k', linewidth=3, label='CO$^*$')
        ax.semilogy(wgs_nib_full['Tout'], cov_out[:, 2], '-k', linewidth=3,  label='O$^*$')
        ax.semilogy(wgs_nib_full['Tout'], cov_out[:, 5], '-k', linewidth=3,  label='OH$^*$')
        ax.axis((300,700,1e-06,1))
        ax.set_xlabel('Temperature [$^\circ$C]')
        ax.set_ylabel('$\Theta_k$ [-]')
#        ax.legend(loc='best')
        
        #Label each line 
        labelLines(plt.gca().get_lines(),align=False)
        
    if vis==3:
        from labellines import labelLine
        
        #Axial coordinate [mm]
        zcoord = wgs_nib_full['zcoord']*1e03
        
        #Coverages along reactor
        covs = wgs_nib_full['covs']

        ax = fig.add_subplot(111)  

        ax.semilogy(zcoord, covs[15][:,1], '-k', linewidth=3, label='375 $^\circ$C')
        ax.semilogy(zcoord, covs[15][:,3], '-k', linewidth=3, ) 
        
        ax.semilogy(zcoord, covs[25][:,1], '--k', linewidth=3, label='425 $^\circ$C')
        ax.semilogy(zcoord, covs[25][:,3], '--k', linewidth=3, ) 
        
        ax.semilogy(zcoord, covs[35][:,1], ':k', linewidth=3,  label='475 $^\circ$C')
        ax.semilogy(zcoord, covs[35][:,3], ':k', linewidth=3, ) 
        
        ax.set_xlabel('Length [mm]')
        ax.set_ylabel('$\Theta_k$ [-]')
        ax.axis((0.0,zcoord[-1],1e-3,1))
        ax.legend(loc='best')
        ax.legend(frameon=False)
        
        #Get lines data
        lines = plt.gca().get_lines()
        
        #Label each line 
        labelLine(lines[0],3,label='H$^*$',align=False,fontsize=20)
        labelLine(lines[1],7,label='CO$^*$',align=False,fontsize=20)
        
        labelLine(lines[2],5,label='H$^*$',align=False,fontsize=20)
        labelLine(lines[3],7,label='CO$^*$',align=False,fontsize=20)
        
        labelLine(lines[4],7,label='H$^*$',align=False,fontsize=20)
        labelLine(lines[5],7,label='CO$^*$',align=False,fontsize=20)        

    if vis==4:
        
        #SA results for full mech.
        sa_all = wgs_nib_full_sa['sm_all']
        
        ax = fig.add_subplot(111)  
        ax.plot(wgs_nib_full_sa['Tout'], sa_all[:,7], '-k', linewidth=3, label='R8')
        ax.plot(wgs_nib_full_sa['Tout'], sa_all[:,8], '--k', linewidth=3, label='R9')
        ax.plot(wgs_nib_full_sa['Tout'], sa_all[:,17], ':k', linewidth=3, label='R18') 
        ax.set_xlabel('Temperature [$^\circ$C]')
        ax.set_ylabel('$B_{j}$ [-]')
        ax.legend(loc='best')
        ax.legend(frameon=False)

    if vis==5:
        
        #SA results for redux mech.
        sa_redux = wgs_nib_redux_sa['sm_all']
        
        ax = fig.add_subplot(111)  
        ax.plot(wgs_nib_redux_sa['Tout'], sa_redux[:,6], '-k', linewidth=3, label='R9')
        ax.plot(wgs_nib_redux_sa['Tout'], sa_redux[:,8], '--k', linewidth=3, label='R11')
        ax.set_xlabel('Temperature [$^\circ$C]')
        ax.set_ylabel('$B_{j}$ [-]')
        ax.legend(loc='best')
        ax.legend(frameon=False)
        
    if vis==6:
        
        #CO conversion plot
        ax = fig.add_subplot(111) 
        ax.plot(exp_data[:,0], exp_data[:,1], 'sk', markersize=10,
                markerfacecolor='w', label='Exp. data, Wheeler et al.')
        ax.plot(wgs_nib_full['Tout'], wgs_nib_full['co_conv'], '-k', 
                linewidth=3, label='Full mech. (19-steps)')
        ax.plot(wgs_nib_redux['Tout'], wgs_nib_redux['co_conv'], '--k', 
                linewidth=3, label='Reduced mech. (9-steps)')
        ax.plot(wgs_nib_full['Tout'], wgs_nib_full['co_conv_eq'], ':k', 
                linewidth=3, label='Equilibrium')
        ax.set_xlabel('Temperature [$^\circ$C]')
        ax.set_ylabel('Conversion [%]')
        ax.legend(loc='best')
        ax.legend(frameon=False)
        plt.xlim((300,700))

    if vis==7:
        
        #CO conversion plot w/ one-rate exp.
        ax = fig.add_subplot(111) 
        ax.plot(exp_data[:,0], exp_data[:,1], 'sk', markersize=10,
                markerfacecolor='w', label='Exp. data, Wheeler et al.')

        ax.plot(wgs_nib_redux['Tout'], wgs_nib_redux['co_conv'], '-k', 
                linewidth=3, label='Reduced mech. (10-steps)')
        
        ax.plot(wgs_nib_simple_fixed['Tout'], wgs_nib_simple_fixed['co_conv'], '--k', 
                linewidth=3, label='One-step exp. (fixed $\Theta$)')
        
        ax.plot(wgs_nib_simple['Tout'], wgs_nib_simple['co_conv'], '-.k', 
                linewidth=3, label='One-step exp. (look-up table)')
        
        ax.plot(wgs_nib_full['Tout'], wgs_nib_full['co_conv_eq'], ':k', 
                linewidth=3, label='Equilibrium')
        
        ax.set_xlabel('Temperature [$^\circ$C]')
        ax.set_ylabel('Conversion [%]')
        ax.legend(loc='best',frameon=False,fontsize=20)
        plt.xlim((300,700))

    if vis==8:
        
        from labellines import labelLines
        
        #CO conversion plot w/ one-rate exp. inlet composition variation
        ax = fig.add_subplot(111) 
        
        ratio = wgs_nib_simple_incomps['ratio']
        rratio = np.flip(ratio,axis=0)
        
        for k in range(ratio.size):
            if k == 0:
                labelratio = 'CO:H$_2$O = '+str(int(ratio[k]))+':'+str(int(rratio[k]))
            else:
                labelratio = str(int(ratio[k]))+':'+str(int(rratio[k]))
            
            ax.plot(wgs_nib_redux_incomps['Tout'], 
                    wgs_nib_redux_incomps['co_conv_all'][k,:], '-k', 
                    linewidth=3, label= labelratio )
            
            ax.plot(wgs_nib_simple_incomps['Tout'], 
                    wgs_nib_simple_incomps['co_conv'][k,:], 'ok', 
                    linewidth=3)

            ax.plot(wgs_nib_simple_incomps['Tout'], 
                    wgs_nib_simple_incomps['co_conv_eq'][k,:], ':k', 
                    linewidth=2)
            
        ax.set_xlabel('Temperature [$^\circ$C]')
        ax.set_ylabel('Conversion [%]')
        plt.xlim((300,700))

        #Label each line 
        xvals = [476,555,575,595,610,620,625,630,635,640]
        labelLines(plt.gca().get_lines(),align=False,xvals=xvals,fontsize=24)
       
    if vis==9:
        
        from labellines import labelLines
        
        #CO conversion plot w/ one-rate exp. inlet composition variation
        ax = fig.add_subplot(111) 
        
        vdot0 = np.array([0.03,0.3,3,30,300,3000])
        
        for k in range(vdot0.size):
            
            if k<2:
                ax.plot(wgs_nib_redux_mdots['Tout'], 
                    wgs_nib_redux_mdots['co_conv'][k,:], '-k', 
                    linewidth=3, label=str(vdot0[k])+' SLPM')
            else:
                ax.plot(wgs_nib_redux_mdots['Tout'], 
                    wgs_nib_redux_mdots['co_conv'][k,:], '-k', 
                    linewidth=3, label=str(int(vdot0[k]))+' SLPM')
                
            ax.plot(wgs_nib_simple_mdots['Tout'], 
                    wgs_nib_simple_mdots['co_conv'][k,:], 'ok', 
                    linewidth=3)
            
        ax.plot(wgs_nib_redux_mdots['Tout'], 
                wgs_nib_redux_mdots['co_conv_eq'][0,:], ':k', 
                linewidth=2)
                    
        ax.set_xlabel('Temperature [$^\circ$C]')
        ax.set_ylabel('Conversion [%]')
        plt.xlim((300,700))

        #Label each line 
        xvals = [350,370,450,520,600,640]
        labelLines(plt.gca().get_lines(),align=False,xvals=xvals,fontsize=20)

    if vis==10:
        
        from labellines import labelLines
        
        #CO conversion plot w/ one-rate exp. inlet composition variation
        ax = fig.add_subplot(111) 
        
        ratio = wgs_nib_simple_incomps['ratio']
        rratio = np.flip(ratio,axis=0)
        
        covs_mean = np.mean(wgs_nib_redux_incomps['covs'],axis=2)
        
        for k in range(0,ratio.size,2):
            if k == 0:
                labelratio = 'CO:H$_2$O = '+str(int(ratio[k]))+':'+str(int(rratio[k]))
            else:
                labelratio = str(int(ratio[k]))+':'+str(int(rratio[k]))
                
            ax.semilogy(wgs_nib_redux_incomps['Tout'], 
                        covs_mean[k,:,2], '-k', 
                        linewidth=3, label = labelratio )
            
        ax.set_xlabel('Temperature [$^\circ$C]')
        ax.set_ylabel('$\Theta_\mathrm{CO}$ [-]')
#        plt.ylim((1e-03,0.2))
        plt.xlim((300,700))

        #Label each line 
        xvals = [500,550,550,590,630]
        labelLines(plt.gca().get_lines(),align=False,xvals=xvals,fontsize=24)

    if vis==11:

        from labellines import labelLines
        
        #Coverages with temperature   
        zcoord = wgs_nib_full['zcoord']*1e03
        Xg = np.array(wgs_nib_full['Xg'])[35]
     
        ax = fig.add_subplot(111)
        ax.plot(zcoord, Xg[:, 0], '-b', linewidth=3, label='H$_2$O')
        ax.plot(zcoord, Xg[:, 1], '-g', linewidth=3, label='H$_2$')
        ax.plot(zcoord, Xg[:, 2], '-r', linewidth=3, label='CO')
        ax.plot(zcoord, Xg[:, 3], '-m', linewidth=3, label='CO$_2$')
        
        ax.set_xlabel('Axial coordinate [mm]')
        ax.set_ylabel('Molar fraction [-]')
#        ax.legend(loc='best')
        
        #Label each line 
        labelLines(plt.gca().get_lines(),align=False,fontsize=30)
     
    plt.show()
    
if exp_fig == 1:
    #Export figure as EPS
    plt.savefig(figname,format='eps',dpi=600)