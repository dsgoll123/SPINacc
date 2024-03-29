#!/usr/bin/env python3
##### -*- coding: utf-8 -*-
"""
 MLacc - Machine-Learning-based acceleration of spin-up

 Copyright Laboratoire des Sciences du Climat et de l'Environnement (LSCE)
           Unite mixte CEA-CNRS-UVSQ
 
 Code manager:
 Daniel Goll, LSCE, dsgoll123@gmail.com
 
 This software is developed by Yan Sun, Yilong Wang and Daniel Goll.......
 
 This software is governed by the XXX license
 XXXX <License content>
"""

from Tools import *
#added line
import numpy as np
import subprocess
# print Python version 
print(sys.version)

#
# Read configuration file
#

if len( sys.argv ) < 2:
  print('Missing argument: DEF directory')
  sys.exit()
else:
  dir_def = sys.argv[1]

f=open(dir_def+'MLacc.def','r')
config=f.readlines()
f.close()

# Define standard output file 
thefile=config[1].strip()

# Define task 
itask=config[3].strip().split(',')

logfile=open(thefile,'w',1)
check.display('DEF directory: '+dir_def,logfile)
check.display('running task: %s'%str(itask),logfile)

# Define task 
resultpath=config[5].strip()+'/'
check.display('results are stored at: '+resultpath,logfile)

# Read list of variables
varfile=open(dir_def+'varlist.json','r')
varlist=json.loads(varfile.read())

# load stored results or start from scratch
iprec=int(config[7].strip())
if iprec:
  check.display('read from previous results...',logfile)
  packdata=np.load(resultpath+'packdata.npy',allow_pickle=True).item()
  auxil=np.load(resultpath+'auxil.npy',allow_pickle=True).item()
else:
  check.display('MLacc start from scratch...',logfile)
  # initialize packaged data and auxiliary data
  packdata=pack()
  auxil=auxiliary()
  readvar(packdata,auxil,varlist,config,logfile)
  np.save(resultpath+'packdata.npy',packdata)
  np.save(resultpath+'auxil.npy',auxil)

# range of Ks to be tested, and the final K
maxK=int(config[11].strip())
auxil.Ks=range(2,maxK+1)
auxil.K=int(config[9].strip())

# Define random seed
iseed=int(config[13].strip())
random.seed(iseed)

# added line, randomize with np also because Cluster_all 
np.random.seed(iseed)

check.display('random seed = %i'%iseed,logfile)
# Define do leave-one-out crosee validation (loocv=1) or not (loocv=0)
loocv=int(config[15].strip())

# Read whether to run reproducibility tests for each task
run_repro_test_task1 = int(config[17].strip())
run_repro_test_task2 = int(config[19].strip())
run_repro_test_task3 = int(config[21].strip())
run_repro_test_task4 = int(config[23].strip())


if '1' in itask:
  #
  # test clustering
  dis_all=Cluster.Cluster_test(packdata,auxil,varlist,logfile) 
  #added line 
  np.random.seed(iseed)
  dis_all.dump(resultpath+'dist_all.npy')
  check.display('test clustering done!\nResults have been stored as dist_all.npy',logfile)

  #
  # plot clustering results
  fig,ax=plt.subplots()
  lns=[]
  for ipft in range(dis_all.shape[1]):
    lns+=ax.plot(auxil.Ks,dis_all[:,ipft])
  plt.legend(lns,varlist['clustering']['pfts'],title="PFT")
  ax.set_ylabel('Sum of squared distances of samples to\ntheir closest cluster center')
  ax.set_xlabel('K-value (cluster size)')
  fig.savefig(resultpath+'dist_all.png') 
  plt.close('all')
  check.display('test clustering results plotted!\nResults have been stored as dist_all.png',logfile)
  # Run test of reproducibility for the task if yes
  if run_repro_test_task1:
     subprocess.run(['python', 'tests/task1_log.py'])
     check.display('Task 1 reproducibility test results have been stored in tests_results.txt',logfile)
  check.display('task 1: done',logfile)
if '2' in itask:
  #
  # clustering
  KK=int(config[9].strip())
  check.display('Kmean algorithm, K=%i'%KK,logfile)
  IDx,IDloc,IDsel=Cluster.Cluster_all(packdata,auxil,varlist,KK,logfile)
  np.savetxt(resultpath+'IDx.txt',IDx,fmt='%.2f')
  IDx.dump(resultpath+'IDx.npy')
  IDloc.dump(resultpath+'IDloc.npy')
  IDsel.dump(resultpath+'IDsel.npy')
  check.display('clustering done!\nResults have been stored as IDx.npy',logfile)

  #
  # plot clustering results
  kpfts=varlist['clustering']['pfts']
  for ipft in range(len(kpfts)):
    fig,ax=plt.subplots()
    m=Basemap()
    m.drawcoastlines()
    m.scatter(IDloc[ipft][:,1],IDloc[ipft][:,0],s=10,marker='o',c='gray')
    m.scatter(IDsel[ipft][:,1],IDsel[ipft][:,0],s=10,marker='o',c='red')
    fig.savefig(resultpath+'ClustRes_PFT%i.png'%kpfts[ipft])
    plt.close('all')
  check.display('clustering results plotted!\nResults have been stored as ClustRes_PFT*.png',logfile)
  # Run test of reproducibility for the task if yes 
  if run_repro_test_task2:
     subprocess.run(['python', 'tests/task2_log.py'])
     check.display('Task 2 reproducibility test results have been stored in tests_results.txt',logfile)
  check.display('task 2: done',logfile)
if '3' in itask:
  #
  # build aligned forcing and aligned restart files
  check.check_file(resultpath+'IDx.npy',logfile)
  IDx=np.load(resultpath+'IDx.npy',allow_pickle=True)
  forcing.write(varlist,resultpath,IDx)
  # Run test of reproducibility for the task if yes
  if run_repro_test_task3:
     subprocess.run(['python', 'tests/task3_log.py'])
     subprocess.run(['python', 'tests/task3_2_log.py'])
     check.display('Task 3 reproducibility test results have been stored in tests_results.txt',logfile)
  check.display('task 3: done',logfile)
if '4' in itask:
  #
  # ML extrapolation

  adict=locals()
  var_pred_name1=varlist['pred']['allname']
  var_pred_name2=varlist['pred']['allname_pft']
  var_pred_name=var_pred_name1+var_pred_name2
  auxil.Nv_nopft=len(var_pred_name1)
  auxil.Nv_total=len(var_pred_name)
  auxil.var_pred_name=var_pred_name
  
  # Response variables
  Yvar=varlist['resp']['variables']
  responseY=Dataset(varlist['resp']['sourcefile'],'r')
  
  check.check_file(resultpath+'IDx.npy',logfile)
  IDx=np.load(resultpath+'IDx.npy',allow_pickle=True)
  # generate PFT mask
  PFT_mask,PFT_mask_lai=genMask.PFT(packdata,varlist,varlist['PFTmask']['pred_thres'])

  auxil.Nlat=np.trunc(( 90-IDx[:,0])/auxil.lat_reso).astype(int)
  auxil.Nlon=np.trunc((180+IDx[:,1])/auxil.lon_reso).astype(int)
  labx=['Y']+var_pred_name+['pft']
  
  # copy the restart file to be modified 
  targetfile=varlist['resp']['targetfile'] if 'targetfile' in varlist['resp'] else varlist['resp']['sourcefile']
  restfile=resultpath+targetfile.split('/')[-1]
  os.system('cp -f %s %s'%(targetfile,restfile))
  # add rights to manipulate file:
  os.chmod(restfile, 0o644)
  
  
  for ipool in Yvar.keys():
    #if ipool!="litter":continue
    check.display('processing %s...'%ipool,logfile)
    fx=open(resultpath+ipool+'_R2.txt','w',1)
    fy=open(resultpath+ipool+'_slope.txt','w',1)
    fz=open(resultpath+ipool+'_dNRMSE.txt','w',1)
    fz2=open(resultpath+ipool+'_sNRMSE.txt','w',1)
    fz3=open(resultpath+ipool+'_iNRMSE.txt','w',1)
    f1=open(resultpath+ipool+'_f_SB.txt','w',1)
    f2=open(resultpath+ipool+'_f_SDSD.txt','w',1)
    f3=open(resultpath+ipool+'_f_LSC.txt','w',1)
    fxx=open(resultpath+ipool+'_loocv_R2.txt','w',1)
    fyy=open(resultpath+ipool+'_loocv_slope.txt','w',1)
    fzz=open(resultpath+ipool+'_loocv_dNRMSE.txt','w',1)
    ff1=open(resultpath+ipool+'_loocv_f_SB.txt','w',1)
    ff2=open(resultpath+ipool+'_loocv_f_SDSD.txt','w',1)
    ff3=open(resultpath+ipool+'_loocv_f_LSC.txt','w',1)
    ffz2=open(resultpath+ipool+'_loocv_sNRMSE.txt','w',1)
    ffz3=open(resultpath+ipool+'_loocv_iNRMSE.txt','w',1) 
#    if ipool!='biomass':
#      auxil.pfts=Yvar[ipool][0]['pfts']
#      print(auxil.pfts)
#      ML_som_litter.MLloop(packdata,auxil,ipool,logfile,varlist,labx,resultpath,fx,fy,fz,fz2,fz3,f1,f2,f3,fxx,fyy,fzz,ff1,ff2,ff3,ffz2,ffz3,loocv)
#    else:
#      auxil.pfts=range(2,varlist['npfts']+2)#[1:]#Yvar[ipool][0]['pfts']
#      print(auxil.pfts)
#      ML_biomass.MLloop(packdata,auxil,ipool,logfile,varlist,labx,resultpath,fx,fy,fz,fz2,fz3,f1,f2,f3,fxx,fyy,fzz,ff1,ff2,ff3,ffz2,ffz3,loocv)
#    else:continue
    ML.MLloop(packdata,auxil,ipool,logfile,varlist,labx,resultpath,fx,fy,fz,fz2,fz3,f1,f2,f3,fxx,fyy,fzz,ff1,ff2,ff3,ffz2,ffz3,loocv, restfile)
    fx.close()
    fy.close()
    fz.close()     
    fxx.close()
    fyy.close()
    fzz.close()
    
  # we need to handle additional variables in the restart files but are not state variables of ORCHIDEE  
  
  if 'additional_vars' not in varlist['resp']:
    # Handle the case where 'additional_vars' is not present
    print("We only modify true state variables of ORCHIDEE")  
  else:
    additional_vars = varlist['resp']['additional_vars']

    for var in additional_vars:
        check.display('processing %s...'%var,logfile)
        restnc=Dataset(restfile,'a')
        # all variables derive from npp longterm prediction
        restvar=restnc['npp_longterm']
        restvar1=restnc[var]

        if var == 'gpp_week' or var == 'maxgppweek_lastyear' or var == 'gpp_daily':
          tmpvar=restvar[:]*2.
        else:
          tmpvar=restvar[:]

        restvar1[:]=tmpvar
        restnc.close()     
    # Run test of reproducibility for the task if yes
  if run_repro_test_task4:  
     subprocess.run(['python', 'tests/task4_log.py'])
     subprocess.run(['python', 'tests/task4_2_log.py'])
     check.display('Task 4 reproducibility test results have been stored in tests_results.txt',logfile)  
  check.display('task 4: done',logfile)
if '5' in itask:
  Yvar=varlist['resp']['variables']
  for ipool in Yvar.keys():
   # if ipool!="litter":continue
    subpool_name=varlist['resp']['pool_name_'+ipool]
    npfts=varlist['resp']['npfts']
    subLabel=varlist['resp']['sub_item']
    print(subLabel)
    pp=varlist['resp']['dim'][ipool]
    sect_n=varlist['resp']['sect_n'][ipool]
    if pp[0]=='pft':
      dims=np.array([0,1])
    else:
      dims=np.array([1,0])
    eval_plot_un.plot_metric(resultpath,npfts,ipool,subLabel,dims,sect_n,subpool_name)
    if loocv==1:
      eval_plot_loocv_un.plot_metric(resultpath,npfts,ipool,subLabel,dims,sect_n,subpool_name)
    else:continue
  check.display('task 5: done',logfile)
