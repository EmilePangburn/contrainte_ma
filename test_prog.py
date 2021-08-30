

import numpy as np
import CAMB_base as cb
import camb
import matplotlib.pyplot as plt
import os 


#fraction d'ionisation en escalier
def construction(bins):
    taille = np.shape(bins)[0]+6
    z = np.zeros(taille)
    z[0], z[1],z[2],z[3]= 1,3.8,4.3,6
    z[taille-2],z[taille-1]=30.25,35
    for i in range(4,taille-2):
        z[i]=6+(i-3)*0.25
    xe = np.zeros(np.shape(z)[0])
    xe[0],xe[1],xe[2],xe[3]= 1.08,1.08,1,1
    xe[taille-2],xe[taille-1]=0,0
    for i in range(4,taille-2):
        xe[i] = bins[i-4]
    return z,xe
    
    #z = np.array([1,2.9999,3,3.9999,4,4.9999,5,5.9999,6,6.9999,7,7.9999,8,8.9999,9,9.9999,10,15])
    xe = np.zeros(np.shape(z)[0])
    xe[0],xe[1],xe[2],xe[3]= 1.08,1.08,1,1
    taille = np.shape(z)[0]
    xe[taille-1],xe[taille-2]=0,0
    for i in range(2,taille//2-1):
        xe[2*i],xe[2*i+1]= bins[i-2],bins[i-2]
    return z,xe


#calcul les dérivées des spectres de puissance
def get_deriv_bin_ma(parameter,liste,bins,amplitude,Vec_propre,indice_max,ma_params):
    cmb_params = {"H0": 67.4,"As": 1e-9,"ombh2": 0.02237,"omch2": 0.1200,"ns": 0.9649 }
    if parameter not in liste:
        if parameter == 'As':
            stepsize = 10**(-11)
        if parameter != 'As':
            stepsize = 0.01
        
        cmb_params_l = cmb_params.copy()
        cmb_params_r = cmb_params.copy()
        model = camb.TanhReionization()
        z,xe = construction(bins)
        model.set_xe(z,xe,smooth=0)
        cmb_params_l[parameter] = cmb_params[parameter]-stepsize
        cmb_params_r[parameter] = cmb_params[parameter]+stepsize
        pars_left = camb.set_params(**cmb_params_l,Reion=model)
        pars_right = camb.set_params(**cmb_params_r,Reion=model)
        
    if parameter in liste :
        stepsize=10**(-3)
        bins_l = np.zeros(96)
        bins_r = np.zeros(96)
        ma_params_l = ma_params.copy()
        ma_params_r = ma_params.copy()
        ma_params_l[parameter] = ma_params[parameter]-stepsize
        ma_params_r[parameter] = ma_params[parameter]+stepsize
        #bins_l[amplitude[parameter]-1] =   bins_l[amplitude[parameter]-1]-stepsize
        #bins_r[amplitude[parameter]-1] = bins_r[amplitude[parameter]-1]+ stepsize
        for i in range(96):
            for k in range(1,indice_max+1):
                bins_l[i]+=ma_params_l['m'+str(k)]*Vec_propre[i,k-1]
                bins_r[i]+=ma_params_r['m'+str(k)]*Vec_propre[i,k-1]
        
        model_l = camb.TanhReionization()
        model_r = camb.TanhReionization()
        z_l,xe_l = construction(bins_l)
        z_r,xe_r = construction(bins_r)
        model_l.set_xe(z_l,xe_l,smooth=0)
        model_r.set_xe(z_r,xe_r,smooth=0)
        pars_left = camb.set_params(**cmb_params,Reion=model_l)
        pars_right = camb.set_params(**cmb_params,Reion=model_r)
        
       
    result_left = camb.get_results(pars_left)
    result_right = camb.get_results(pars_right)
    powers_left =result_left.get_cmb_power_spectra(pars_left,raw_cl=True)
    powers_right =result_right.get_cmb_power_spectra(pars_right,raw_cl=True)   
    cl_left = powers_left['total']
    cl_right = powers_right['total']
    cl_tt_left = cl_left[:,0]
    cl_tt_right = cl_right[:,0]
    cl_te_left = cl_left[:,3]
    cl_te_right = cl_right[:,3]
    cl_ee_left = cl_left[:,1]
    cl_ee_right = cl_right[:,1]
    cl_bb_left = cl_left[:,2]
    cl_bb_right = cl_right[:,2]
       
    dCltt_dh = (cl_tt_right - cl_tt_left) / (2 * stepsize)
    dClte_dh = (cl_te_right - cl_te_left) / (2 * stepsize)
    dClee_dh = (cl_ee_right - cl_ee_left) / (2 * stepsize)
    dClbb_dh = (cl_bb_right - cl_bb_left) / (2 * stepsize)
    return dCltt_dh,dClte_dh,dClee_dh,dClbb_dh


#calcul les éléments de la matrice de Fisher
def Fisher_ij_bin_ma(param1,param2,liste,bins,amplitude,Vecteur_propre,nb_ma,ma_params): 
    cmb_params = {"H0": 67.4,"As": 1e-9,"ombh2": 0.02237,"omch2": 0.1200,"ns": 0.9649 }
    model = camb.TanhReionization()
    z,xe = construction(bins)
    model.set_xe(z,xe,smooth=0)
    F_ij = 0
    f_sky =1
    if param1==param2:
        res1 = get_deriv_bin_ma(param1,liste,bins,amplitude,Vecteur_propre,nb_ma,ma_params)
        res2 = res1
    if param1 != param2:
        res1 = get_deriv_bin_ma(param1,liste,bins,amplitude,Vecteur_propre,nb_ma,ma_params)
        res2 = get_deriv_bin_ma(param2,liste,bins,amplitude,Vecteur_propre,nb_ma,ma_params)
    pars = camb.set_params(**cmb_params,Reion=model)
    results = camb.get_results(pars)
    powers =results.get_cmb_power_spectra(pars,raw_cl=True)
    totCL=powers['total']
    for l in range(2,200):
        mat_cl = totCL[l,1]
        matder1 = res1[2][l]
        matder2 = res2[2][l]
        F_ij += (2*l+1)*f_sky/2*matder1*matder2/mat_cl**2
    return F_ij




def Full_Fisher_bin_ma(bins,nb_ma,Vecteur_propre):
    liste = []
    ma_params = {}
    # il faut bien préciser les ma
    #donne la liste des m_a changer le fichier
    result = np.loadtxt('/pbs/home/e/epangbur/stagem1/ma_liste_015')
    for i in range(1,nb_ma+1):
        ma_params["m"+str(i)]=result[i-1]
    for i in range(1,nb_ma+1):
        liste.append('m'+str(i))
    amplitude={}
    for i in range(1,97):
        amplitude["m"+str(i)]=i
    #paramètres cosmologiques
    parameter = ["H0","As","ombh2","omch2","ns"]
    parameter = parameter+ liste
    Fisher = np.zeros((5+nb_ma,5+nb_ma))
    print(ma_params)
    for i in range(5+nb_ma):
        for j in range(i,5+nb_ma):
            Fisher[i,j] = Fisher_ij_bin_ma(parameter[i],parameter[j],liste,bins,amplitude,Vecteur_propre,nb_ma,ma_params)
            print(i,j)
            
    for i in range(5+nb_ma):
        for j in range(i):
            Fisher[i,j]=Fisher[j,i]

    return Fisher


def xe_f(z, tau=0.0544, dz=1, f=1.081884281020483535, with_helium=True,zre=9):
   # cmb_params = {'omch2': 0.120, 'ombh2': 0.0224,'H0': 67.4,'As': 10**-10*np.exp(3.047),'ns': 0.965,'tau': tau}
    cmb_params = {"cosmomc_theta": 0.0104085,"As": 1e-9,"ombh2": 0.02237,"omch2": 0.1200,"ns":0.9649,"tau": tau, "max_l":2002}
    
    He_fraction = 8.1884281020483535E-002
    y = (1+z)**(3/2)
    camb_params = camb.set_params(**cmb_params)
    """
    if zre is not None:
        zr = zre
    if zre is None :
        zr = camb.get_zre_from_tau(camb_params, tau)
    """
    zr=zre
    yre = (1+zr)**(3/2)
    dy = 3/2 * (1 + z)**(1/2) * dz   
    xe = (1+He_fraction)/2 * (1 + np.tanh((yre - y) / dy))  
    # Effect of Helium becoming fully ionized is small so details not important
    
    He_redshift = 3.5
    He_delta = 0.4
    xe += He_fraction / 2 * (1 + np.tanh((He_redshift - z)/He_delta))   
    #xe += He_fraction  * (1 + np.tanh((He_redshift - z)/He_delta))   
    return xe-2*10**(-4)


#calcul les m_a
def ma_z(Vecteur,x_true,x_fid):
    taille = np.shape(Vecteur)
    z = np.linspace(6,30,96)
    vec_true = x_true(z)
    somme = np.sum(Vecteur*(vec_true-x_fid))
    return somme*0.25/(24)


#calcul la fraction d'ionisation à partir des m_a
def xe(z,nb_max,x_true,x_fid,Vecteur_propre):
    coeff = np.zeros(nb_max)
    x_fid = np.zeros(96)
    for i in range(nb_max):
        coeff_a = ma_z(np.real(Vecteur_propre[:,i]),x_true,x_fid)
        coeff[i]= coeff_a
        x_fid += coeff_a*np.real(Vecteur_propre[:,i])
    return x_fid



arrayID = os.environ["SGE_TASK_ID"]
mat = np.loadtxt('/pbs/home/e/epangbur/stagem1/true_mat.txt')
#permet de construire la fraction d'ionisation à partir des m_a
Vecteur_propre = 9.8*np.linalg.eig(mat)[1]
Val_p = np.linalg.eig(mat)[0]


z = np.linspace(6,30,96)
#fraction d'ionisation test
x_fid = np.full(96,0.15)

#on choisit le nombre de m_a à contraindre
nb_ma  =4+int(arrayID)
nb = nb_ma
#on construit la fraction d'ionisation associé
xe = xe(z,nb,xe_f,x_fid,Vecteur_propre)

bins = xe

#matrice de Fisher des m_a
ma = Full_Fisher_bin_ma(bins,nb_ma,Vecteur_propre)

np.savetxt('/pbs/home/e/epangbur/stagem1/error_ma015_'+str(nb_ma)+'.txt',ma)