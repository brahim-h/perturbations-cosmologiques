#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  4 23:13:43 2022

@author: braams92
"""
# PARTIE I : l'UNIVERS HOMOGÈNE 


# Dans cette première partie nous allons résoudre l'univers homogène, c'est-à-dire le comportement de l'univers 
# dans son ensemble en résolvant (entre autres) l'équation de friedmann-lemaitre. 

import numpy as np 
import matplotlib.pyplot as plt 
import scipy.integrate 
from scipy.integrate import odeint 
from scipy import optimize
from matplotlib.colors import *
from matplotlib.colors import LogNorm
import matplotlib.colors as mcolors
from scipy import interpolate

# PARAMÈTRES 

H0 = 2.26e-18                      # constante d'Hubble (seconde)   
omega_m0 =  0.3                    # proportion de matière (baryons + matière noire) aujourd'hui
omega_r0 = 1e-4                    # proportion de rayonnement aujourd'hui
omega_lambda0 = 0.7                # proportion d'énergie noire aujourd'hui
Tcmb = 2.726 # Kelvin              # température du CMB (fond diffus cosmologique)
a1 = 1/119                         # constante apparaissant dans l'expression de la température des baryons
a2 = 1/115                         # constante apparaissant dans l'expression de la température des baryons
gamma = 5/3                        # indice adiabatique d’un gaz parfait monoatomique
kbb = 1.380649e-23                 # constante de boltzmann (exprimée en m**2.kg/k*s**2)
kb = 1.380649e-23*(3.24078e-20)**2 # constante de boltzmann (exprimée en Mpc**2.kg/k*s**2)
u = 1.22                           # le poids moléculaire moyen
Mh = 1.6735575e-27                 # la masse de l’atome d’hydrogène
w_b = 0.15                         # au sein de la matière, proportion de baryons 
w_c = 1-w_b                        # au sein de la matière, proportion de la matière noire


def euler(f,ti,tf,nt,yi):         
    y = np.zeros(nt+1)
    h = (tf-ti)/(nt)
    t=np.linspace(ti,tf,nt+1)
    y[0] = yi
    for i in range (nt): 
        y[i+1] = y[i] + h*f(t[i],y[i])
    return t, y


t0 = -2*np.arcsinh(np.sqrt(0.7/0.3))/(3*H0*np.sqrt(0.7))      # solution analytique de l'âge de l'univers pour une cosmologie 70% énergie noire / 30% matière 
t000 = -2*np.arcsinh(np.sqrt(0.01/0.99))/(3*H0*np.sqrt(0.01)) # solution analytique de l'âge de l'univers pour une cosmologie 1% énergie noire / 99% matière 

plt.figure(figsize=(8,6))

def f(t):
    return (0.7/0.3)**(-1/3)*(np.sinh(3*H0*(t-t0)*np.sqrt(0.7)/(2)))**(2/3) # solution analytique à l'équation de friedmann pour une cosmologie 70% énergie noire / 30% matière
t=np.linspace(-15*3e16,15*3e16,100000)
plt.grid()
plt.plot(t*H0,f(t),'-', label = ("a(t) analytique univers avec 70% d'énergie-noire et 30% de matière"), color = 'blue',linewidth = 2)
plt.legend()
plt.title("évolution du facteur d'échelle dans un univers de matière + énergie noire" )
plt.annotate("âge de l'univers dans cette cosmologie, t0 ="+str("%.2f" %((np.abs(t0))/(3.1536e16)))+" Gyr", xy=(-0.96,0),xytext = (-0.95,0.80),arrowprops=dict(facecolor='black', shrink=0.07) )
plt.xlabel("H0*t")
plt.ylabel("a")


def facteur_échelle(t,a): 
    return  H0*(0.3*a**(-1)+(0)*a**(-2)+0.7*a**(2))**(1/2)  # résolution numérique de l'équation de friedmann 
ti = 0
tf = 15*3e16
a0 = 1                                                      # on fixe a0 = 1 "aujourd'hui"
nt = 10000                                                 

t, a =  euler(facteur_échelle,ti,tf,nt,a0)
plt.grid()
plt.plot(H0*t,a,':', label = ("a(t) numérique univers avec 70% d'énergie-noire et 30% de matière "), color = 'black', linewidth = 6.0)
plt.xlabel("H0*t")
plt.ylabel("facteur d'échelle ")
plt.grid()
plt.title("évolution du facteur d'échelle en fonction du temps a = f(t) en LogLog" )
plt.grid()
plt.legend()
plt.grid()


def f(t):
    return (0.01/0.99)**(-1/3)*(np.sinh(3*H0*(t-t000)*np.sqrt(0.01)/(2)))**(2/3) # solution analytique à l'équation de friedmann pour une cosmologie 1% énergie noire / 99% matière
t=np.linspace(-15*3e16,15*3e16,100000)
plt.grid()
plt.plot(t*H0,f(t),'-', label = ("a(t) analytique univers avec 1% d'énergie-noire et 99% de matière"), color = 'violet',linewidth = 3)
plt.legend()
plt.title("évolution du facteur d'échelle dans un univers de matière + énergie noire" )
plt.annotate("Âge de l'univers dans cette cosmologie, t0 ="+str("%.2f" % ((np.abs(t000))/(3.1536e16)))+" Gyr", xy=(-0.68,0.02),xytext = (-0.25,0.1),arrowprops=dict(facecolor='black', shrink=0.07))
plt.xlabel("H0*t")
plt.ylabel("a")

def facteur_échelle(t,a): 
    return  H0*(0.99*a**(-1)+(0)*a**(-2)+0.01*a**(2))**(1/2)  
ti = 0
tf = 15*3e16
a0 = 1  # on fixe a0 = 1 donc H0 = 1 dans notre modèle et t0 représente " aujourdhui "
nt = 10000

t, a =  euler(facteur_échelle,ti,tf,nt,a0)
plt.plot(H0*t,a,':', label = ("a(t) numérique univers avec 1% d'énergie-noire et 99% de matière "), color = 'black', linewidth = 6.0)
plt.annotate("Aujourd'hui", xy=(0,1),xytext = (-0.5,1.5),arrowprops=dict(facecolor='black',shrink=0.07))
plt.xlabel("H0*t")
plt.ylabel("facteur d'échelle a(t) ")
plt.title("évolution du facteur d'échelle en fonction du temps a = f(H0*t) " )
plt.grid()
plt.legend()
#plt.savefig("scaleFactor.pdf",dpi = 700)
plt.show()

print("La dynamique du facteur d'échelle dépend du contenu de l'univers, elle dépend directement de la proportion des différents fluides cosmologiques (rayonnement, matière, énergie noire), voyons comment évolue la concentrationdes différents fluides afin de mieux comprendre les 2 principaux régimes qui semblent apparaitres dans le graph ci-dessus ")

# La dynamique du facteur d'échelle dépend du contenu de l'univers, elle dépend directement de la proportion des 
# différents fluides cosmologiques (rayonnement, matière, énergie noire), voyons comment évolue la concentration
# des différents fluides afin de mieux comprendre les 2 principaux régimes qui semblent apparaitres dans le graph ci-dessus 


plt.figure(figsize=(6,4))

def rho_m(aa):                   # densité de matière 
    return 0.3*aa**-1
aa = np.linspace(1e-5,5,1000)
plt.plot((aa),(rho_m(aa)),'--',label = r'$\Omega_{m}(a)$', color = "black")

#plt.xscale('log')
#plt.yscale('log')
#plt.show()



def rho_r(aa):                   # desnité de rayonnemennt
    return 1e-4*aa**-2
aa = np.linspace(1e-5,5,1000)
plt.plot((aa),(rho_r(aa)),'--',label = r'$\Omega_{r}(a)$', color = 'violet')
#plt.xscale('log')
#plt.yscale('log')
#plt.show()


def rho_lamb(aa):                # desnité d'énergie noire
    return 0.7*aa**2
aa = np.linspace(1e-5,5,1000)
plt.plot((aa),(rho_lamb(aa)),'--',label = r'$\Omega_{\Lambda}(a)$',color = 'red')
#plt.xscale('log')
#plt.yscale('log')
#plt.show()



plt.title("évolution de la proportion des différents fluides cosmologiques" )
plt.ylabel(" $\Omega_{i}(a)$ ")
plt.xlabel(" facteur d'échelle ")
plt.legend()
plt.text(0.1,50,'matière',horizontalalignment = 'center', verticalalignment = 'center',color = 'black' )
plt.text(0.3,0.00003,'rayonnement',horizontalalignment = 'center', verticalalignment = 'center',color = 'violet')
plt.text(0.010,0.007,'énergie-noire',horizontalalignment = 'center', verticalalignment = 'center',color = 'red')

plt.xscale('log')
plt.yscale('log')
plt.grid()
plt.show()

print("D'après cette figure, on comprend que la proportion des différents fluides dépend de l'âge de l'Univers. En effet, quand l'Univers était jeune ('a' très petit) c'est le fluide de rayonnement qui dominait, un peu plus tard, c'est le fluide de matière qui dominait et puis encore plus tard, par exemple aujourd'hui (pour a = 1), c'est l'énergie noire qui domine. Ces différentes évolutions dans la proportion des différents fluides vont jouer un rôle prépondérant sur la dynamique globale de l'Univers et va en quelque sorte façonner la manière dont le facteur d'échelle évolue, voyons donc cela...")


# On peut ainsi comprendre l'évolution de la taille de l'univers, on comprend alors que l'univers est dynamique et que cette 
# dernière dépend fortement de la cosmologie que l'on considère (proportion des différents fluides cosmo)
# (il y a plus de détail sur la physique mise en jeu dans le rapport)

# FIN PARTIE I 



# PARTIE II : PERTURBATIONS COSMOLOGIQUES 

# maintenant que nous avons résolu l'univers homogène et que nous avons une description de l'évolution globale de 
# l'univers, nous pouvons étudier la manière dont évolue le fluide de matière (baryon + matière noire froide) dans un univers 
# en expansion.. Nous allons donc résoudre les équations hydrodynamiques (légèrement modifiée car univers en expansion)
# qui régissent la dynamique du champs scalaire de densité pour les baryons et la matière noire froide. 
# Nous mettrons (notamment) en évidence les BAO (Oscillations Acoustiques des Baryons) caractéristique du couplage entre 
# la matière et le rayonnement émanant de la compétition entre l'effondrement gravitationel et la pression de radiation
# des surdensités de matière. 
# (toute la physique est détaillée dans le rapport)



# 1: évolution de la densité de baryon et matière noire froide en fonction du facteur d'échelle. 

# On va résourdre les équations hydrodynamiques dans l'espace de fourier, le vecteur d'onde 'k' va alors apparaitre
# homogène à l'inverse d'une distance. Pour le moment, nous garderons tout en mètre^-1 

k = 1e-19     #m^-1 

plt.figure(figsize=(8,6))

def F(X,a,k):                                               
    h_t = H0*((omega_m0)*a**(-3)+(omega_r0)*a**(-4)+(omega_lambda0))**(1/2)    # paramètre de Hubble H(a(t))
    Temp_t = (Tcmb/a)*(1+(a/a1)/(1+(a2/a)**(3/2)))**(-1)                       # description de la température des baryons     
    Vs_t = np.sqrt(gamma*kbb*Temp_t/(u*Mh))                                    # description de la vitesse des baryons (comme un gaz parfait)
    delta_c, theta_c, delta_b, theta_b = X
    delta_c_dot = - theta_c / (h_t*a)
    theta_c_dot = -1.5*(h_t/a)*(w_c*delta_c+w_b*delta_b)-2*theta_c/(a)
    delta_b_dot = -theta_b / (h_t*a)
    theta_b_dot = -1.5*(h_t/a)*(w_c*delta_c+w_b*delta_b)-2*theta_b/(a)+(delta_b*(Vs_t*k)**2)/(h_t*(a)**3)
    res = np.array([delta_c_dot, theta_c_dot, delta_b_dot, theta_b_dot])
    return res
    
a = np.linspace(1e-6,1,1000000)                              
delta_c0 = 1                                               # condition initiale sur le champ de desnité en matière noire (DM)
theta_c0 = 0                                               # condition initiale sur la divergence du champ de vitesse pour la DM 
delta_b0 = 1                                               # condition initiale sur le champ de desnité en baryons                                              
theta_b0 = 0                                               # condition initiale sur la divergence du champ de vitesse pout les baryons 
X0=np.array([delta_c0,theta_c0,delta_b0,theta_b0])

 
solu0 = scipy.integrate.odeint(F,X0,a,args=(k,))
delta_c_dot, theta_c_dot, delta_b_dot, theta_b_dot  =solu0.T
       
plt.xlabel(" facteur d'échelle ")
plt.ylabel(" $\delta_b(k)$ ")
plt.plot(a,solu0[:,2], label=r'$\delta_b(a)$ baryons, k = 1e$^{-19}$ m$^{-1}$', color='black') # On plot seulement l'évolution du champ de densité des baryons et DM, la divergence du champ de vitesse ne nous intérèsse pas
                                                                                                
plt.title("évolution du champ de densité des baryons" )
plt.xlabel(" facteur d'échelle")
plt.ylabel("$\delta_b(a)$")
plt.xscale('log')                                          # il est judicieux de tout mettre en échelle Log-Log afin de bien voir les ocsillations (BAO)
plt.yscale('log')
#plt.savefig("champ_de_densité.pdf",dpi = 300)
plt.legend()
plt.grid()
plt.show()

# on observe alors les BAO dans le champ de densité des baryons, il y a plusieurs choses à décrire sur ce graphique
# nous verrons ceci en détail dans le rapport 

# on peut aussi tracer l'évolution du champ de densité de la DM pour bien comprendre que la DM ne se couple pas
# avec le rayonnement. 

plt.figure(figsize=(8,6))
plt.plot(a,solu0[:,2], label=r'$\delta_b(a)$ baryons, k = 1e$^{-19}$ m$^{-1}$', color='black') 
plt.plot(a,solu0[:,0],label=r'$\delta_c(a)$ matière-noire froide, k = 1e$^{-19}$ m$^{-1}$', color='red')  
plt.xscale('log')               # il est judicieux de tout mettre en échelle Log-Log afin de bien voir les ocsillations (BAO)
plt.yscale('log')
plt.title("évolution du champ de densité des baryons et de la DM" )
plt.xlabel(" facteur d'échelle ")
plt.ylabel("$\delta_b(a)$, $\delta_c(a)$ ")
plt.legend()
plt.text(1e-5,2e4, '$\Omega_{m}$='+str(w_b),horizontalalignment = 'center', verticalalignment = 'center',color = 'black',fontsize = 15)
plt.text(1e-5,6e4, '$\Omega_{c}$='+str(w_c),horizontalalignment = 'center', verticalalignment = 'center',color = 'red',fontsize = 15)
plt.text(0.007,3e3,'pas de couplage de la CDM avec les photons',horizontalalignment = 'center', verticalalignment = 'center',color = 'red')
plt.grid()
plt.show()

print("Nous venons donc de mettre en évidence le phénomène de BAO qui permet de comprendre énormément de choses (Puit gravitationel de la CDM, compétitions entre gravité et pression de radiation, corrélation entre BAO et distribution des galaxies)... Il y a beaucoup de choses a dire sur ces graphiques, referez vous au rapport pour davantage de détails.")
# Il y a pleins de choses à dire sur ces différents graphiques, tout sera détaillé dans le rapport



# Un des objectifs du projet est de simuler le fond diffus cosmologique, pour cela, nous aurons besoin de calculer 
# le spectre de puissance qui est une fonction qui dépend de k et du champ de densité des baryons. 
# En conséquence nous allons re calculer l'évolution du champ de densité des baryons
# non pas en fonction du facteur d'échelle mais directement en fonction de k. 


plt.figure(figsize=(8,6))
def F(X,a,k):
    h_t = H0*((omega_m0)*a**(-3)+(omega_r0)*a**(-4)+(omega_lambda0))**(1/2)
    Temp_t = (Tcmb/a)*(1+(a/a1)/(1+(a2/a)**(3/2)))**(-1)
    Vs_t = np.sqrt(gamma*kb*Temp_t/(u*Mh))
    delta_c, theta_c, delta_b, theta_b = X
    delta_c_dot = - theta_c / (h_t*a)
    theta_c_dot = -1.5*(h_t/a)*(w_c*delta_c+w_b*delta_b)-2*theta_c/(a)
    delta_b_dot = -theta_b / (h_t*a)
    theta_b_dot = -1.5*(h_t/a)*(w_c*delta_c+w_b*delta_b)-2*theta_b/(a)+(delta_b*(Vs_t*k)**2)/(h_t*(a)**3)
    res = np.array([delta_c_dot, theta_c_dot, delta_b_dot, theta_b_dot])
    return res
    
a = np.linspace(1e-4,1,1000)
delta_c0 = 1
theta_c0 = 0
delta_b0 = 1
theta_b0 = 0
X0=np.array([delta_c0,theta_c0,delta_b0,theta_b0])

kk = np.geomspace(1e-4,1,1000)                                                 # vecteur d'onde [Mpc-1]    

SOL = []                                                                         
for i in kk :  
    solu0 = scipy.integrate.odeint(F,X0,a,args=(i,))                   
    delta_c_dot, theta_c_dot, delta_b_dot, theta_b_dot  =solu0.T
    SOL.append(solu0[:,2])
    
SOL = np.array(SOL)
plt.plot(kk,SOL[:,1],label=r'$\delta_b(k)$, a = 1e$^{-4}$', color='black')     # solution pour a = 1e-4 
plt.title("évolution du champ de densité des baryons en fonction de k" )
plt.xlabel(" k [Mpc $^{-1}$] ")
plt.ylabel("$\delta_b(k)$")
plt.legend()
plt.xscale('log')
#plt.yscale('log')
plt.legend()
plt.grid()
plt.show()
print("Le spectre de puissance étant défini dans l'espace de Fourier, il est nécéssaire de faire passer la surdensité des Baryons dans cet espace également")

# maintenant que nous avons calculer l'évolution du champ de densité en fonction de k, nous allons calculer le 
# spectre de puissance qui sera l'outil de base pour simuler le fond diffus cosmologique. 
# Le spectre de puissance suit une loi d'échelle de la forme : P(k) = k^(n-4)*delta^2(k)

plt.figure(figsize=(6,4))
def P(k):
    n = 1.2
    return kk**(n-4)*(SOL[:,1])**2                   # nous pouvons choisir n'importe qu'elle valeur de 'n' dans l'intervalle [0,4]

def J(k):
    n = 3.5
    return kk**(n-4)*(SOL[:,1])**2  
def q(k):
    n = 2
    return kk**(n-4)*(SOL[:,1])**2
def o(k):
    n = 2.9
    return kk**(n-4)*(SOL[:,1])**2                   # par la suite, nous verrons l'importance de ce paramètre 'n'
plt.plot(kk,P(k),label = "n = 1.2",color = 'orange')
plt.plot(kk,q(k),label = "n = 2",color = 'magenta')
plt.plot(kk,o(k),label = "n = 2.9",color = 'green')
plt.plot(kk,J(k),label = "n = 3.5",color = 'red')
plt.legend()
plt.xscale('log')
plt.yscale('log')                         
plt.grid()
plt.title("Spectre de puissance de la matière")
plt.xlabel(" k [$Mpc^{-1}$] ")
plt.ylabel(" $P(k)$ ")
#plt.savefig("SpectreDe.pdf",dpi = 1000)
plt.show()
print("Le spectre de puissance est un outil très très puissant qui va permettre de comprendre en détail les simulations du CMB qui arrivent juste en dessous. Pour plus de détail sur le lien Spectre de puissance - CMB, voir le rapport à partir de la page 15")

# Nous pouvons désormais simuler le fond diffus cosmologique à partir d'un 'bruit blanc' gaussien 


plt.figure(figsize=(8,7))
N = 500                                              # taille de la simulation 
def bruit_blanc(N):
    
    return np.random.normal(0,1,(N,N))               # tirage gaussien sur chaque pixel 2d 
plt.imshow(bruit_blanc(N)) 
plt.colorbar()
plt.title("Bruit Blanc")
#plt.savefig("B.pdf",dpi = 300)
plt.show()


s = np.fft.rfft2(bruit_blanc(N))                     # transformer de fourier du bruit blanc

kx = np.fft.fftfreq(500,500/500)                     # système de coordonnée permettant de repérer la position de chaque pixel 
ky = np.fft.rfftfreq(500,500/500)                    # c'est aussi ici que l'on détermine la taille des structures que nous allons observer en déterminant la distance entre 2 pixels de la simulation 

delta = interpolate.interp1d(kk,SOL[:,1],'linear')   # interpolation du champ de densité afin de connaitre sa valeur pour tout k

n = 2.5                                              # valeur de 'n' dans [0,4]
z = np.zeros((500,250))
for i in range(1,500):
    for j in range(1,250):
        z[i,j] = ((s[i,j])*np.sqrt(((kx[i])**2+(ky[j])**2)**(n-4)))*delta(np.sqrt((kx[i])**2+(ky[j])**2)) # on multiplie chaque élément de 's' par la racine carrée du spectre de puissance 

l = z 
def make_colormap(seq):
    """Return a LinearSegmentedColormap
    seq: a sequence of floats and RGB-tuples. The floats should be increasing
    and in the interval (0,1).
    """
    seq = [(None,) * 3, 0.0] + list(seq) + [1.0, (None,) * 3]
    cdict = {'red': [], 'green': [], 'blue': []}
    for i, item in enumerate(seq):
        if isinstance(item, float):
            r1, g1, b1 = seq[i - 1]
            r2, g2, b2 = seq[i + 1]
            cdict['red'].append([item, r1, r2])
            cdict['green'].append([item, g1, g2])
            cdict['blue'].append([item, b1, b2])
    return mcolors.LinearSegmentedColormap('CustomMap', cdict)


c = mcolors.ColorConverter().to_rgb
rvb = make_colormap(
    [c('blue'), c('cyan'),0.44, c('white'), c('orange'),0.7,c('orange'),c('indianred'),c('orangered'),c('red')])



fond_diffus = np.fft.irfft2(l)                       # puis on repasse dans l'espace réel  
plt.figure(figsize=(8,7))
plt.imshow((fond_diffus) , cmap = rvb) 
plt.colorbar()
plt.title("Simulation du CMB en P(k) ∝ k^"+str("%.2f" %(n-4)))
plt.xlabel(" x [Mpc] ")
plt.ylabel(" y [Mpc] ")
plt.show()

print("Et voici !!, nous pouvons tenter de comparer ces simulations aux relevés du satellite Planck par exemple. Ceci est fait dans le rapport. Il est également intéréssant d'étudier la dépendance en 'n' qui est le paramètre cosmologique dont dépend principalement les simulations")



 


