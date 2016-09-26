# -*- coding: utf-8 -*-
"""
Created on Thu Jan  7 12:10:40 2016
Calculates the solubility of B in alpha matrix when varying C concentration
using Wagner formalism for dilute solutions. An approximation is made that
the solvent follows Raoult law for low additions of solutes. This might not
respect Gibbs-Duhem relation.
@author: johannha
"""

import numpy as np
from scipy.optimize import fsolve, root
"""
   Some global parameters
"""
# Constants and parameters
RG = 8.314
T = 1323.
# Formation energy of A3B
dgf = -71152.

"""
   Defines thermodynamic functions and relations
"""

def Parameters_FCCA1(T):
    
    if (T < 1100):
        GHSERBB = -7735.284+107.111864*T-15.6641*T*np.log(T)-6.864515E-3*T**2+0.618878E-6*T**3+370843*T**(-1)
    elif (T >= 1100 ) & (T < 2348):
        GHSERBB = -16649.474+184.801744*T-26.6047*T*np.log(T)-0.79809E-3*T**2-0.02556E-6*T**3+1748270*T**(-1)
    elif ( T >= 2348) & (T < 3000):
        GHSERBB = -36667.582+231.336244*T-31.5957527*T*np.log(T)-1.59488E-3*T**2+0.134719E-6*T**3+11205883*T**(-1)
    elif (T >= 3000):
        GHSERBB = -21530.653+222.396264*T-31.4*T*np.log(T)
    
    if (T < 1728):
        GHSERNI = -5179.159+117.854*T-22.096*T*np.log(T)-4.8407E-3*T**2
    elif (T >= 1728):
        GHSERNI = -27840.620+279.134977*T-43.1*T*np.log(T)+1127.54E28*T**(-9)
    
    if ( T < 1687):
        GHSERSI = -8162.609+137.236859*T-22.8317533*T*np.log(T)-1.912904E-3*T**2-0.003552E-6*T**3+176667*T**(-1)
    elif (T >= 1687):
        GHSERSI = -9457.642+167.281367*T-27.196*T*np.log(T)-420.369E28*T**(-9)

    # Surface of reference
    G0NI = GHSERNI;
    
    # From Dinsdale (SGTE) PURE5: Lattice stability DIAMOND -> FCC
    G0SI = 51000-21.8*T+GHSERSI;
    
    # From Dinsdale (SGTE) PURE5 : Lattice stability RHOMBO->FCC
    #th_param.G0NiB = 43516.0-11.6070*T+GHSERBB;
    # th_param.G0NiB = 27763-27.136*T+GHSERBB;
    G0NIB = 43514.0-12.217*T+GHSERBB+GHSERNI;
    
    # Devrait être à optimiser
    G0SIB = GHSERSI+51000-21.8*T+GHSERBB+43514.0-12.217*T;
    #th_param.G0SiB = 1306.8-9.2035*T+GHSERBB;
    
    # Interaction parameters on substitutional sublattice
    # Sundman and Agren (Assessment of Ni-Si-B phase diagram).
    # th_param.L0NiSiVa = -201292+39.31*T;
    # th_param.L0NiSiVa = -204564.52 + 38.99204*T;
    L0NISIVA = -208234.46+44.14177*T;
    
    L0NISIB = L0NISIVA;
    # th_param.L0NiSiB = 0.0;

    # th_param.L1NiSiVa = -63053+20*T;
    # th_param.L1NiSiVa = -82289.61;
    L1NISIVA = -108533.44;

    L1NISIB = L1NISIVA;
    # th_param.L1NiSiB = 0.0;
    
    # interaction parameters on interstitial sublattice
    # Campbell and Kattner (optimized from Teppo and Taskinen value)
    # th_param.L0NiBVa = -90402.33+36*T;
    L0NIBVA = -46908.8587+14.7084193*T;
    # th_param.L0NiBVa = -41420;
    
    L0SIBVA = 5144.67;
    
    # N/A parameter is not used
    
    #th_param.G0B = 43516.0-11.6070*T+GHSERBB;
    # We pack all data in a dictionnary.
    HS = dict()
    # Lattice stabilities
    HS["G0NI"] = G0NI
    HS["G0SI"] = G0SI
    HS["G0NIB"] = G0NIB
    HS["G0SIB"] = G0SIB
    # secondary pair parameters
    HS["L0NISIVA"] = L0NISIVA
    HS["L0NISIB"] = L0NISIB
    HS["L1NISIVA"] = L1NISIVA
    HS["L1NISIB"] = L1NISIB
    HS["L0NIBVA"] = L0NIBVA
    HS["L0SIBVA"] = L0SIBVA
    
    return HS

def WagnerParameters_FCCA1(T):

    # Get sublattice coefficients
    HS = Parameters_FCCA1(T)

    # Calculates DGX (some sort of fictitious standard state for boron)
    DGX = HS["G0SIB"]-HS["G0SI"]-HS["G0NIB"]+HS["G0NI"]

    eBB = 2-2/RG/T*HS["L0NIBVA"]
    eXB = 1/RG/T*(DGX-HS["L0NISIVA"]+HS["L0NISIB"]+HS["L0SIBVA"]-HS["L0NIBVA"])
    eXX = -2/RG/T*(HS["L0NISIVA"]+3*HS["L1NISIVA"])
    
    return eBB, eXB, eXX

def SolubilityProductEquation(xb, xc, dgf, ebb, ebc, T):
    """
    The system of equations being solved, equations of solubility product.
    """
    return np.exp(dgf/RG/T-ebb*xb-ebc*xc) - xb*(1.-xb-xc)**3

def BoronSolubility(x, x0b, x0si, dgf, ebb, ebc, T):
    xb, xc, xni, falp = x
    fppt = 1 - falp
    x0ni = 1 - x0si - x0b
    
    Ksp = np.exp(dgf/RG/T-ebb*xb-ebc*xc) 
    
    eqn1 = xni**3*xb - Ksp              # Solubility product equation
    eqn2 = x0b - xb*falp - 0.25*fppt    # boron balance
    eqn3 = x0si - xc*falp               # silicon balance (does not precipitate in Ni3B)
    eqn4 = x0ni - xni*falp - 0.75*fppt  # nickel balance
    
    return np.array([eqn1, eqn2, eqn3, eqn4])

def BSolInSuperSaturatedFCC(x_cell, dgf, ebb, ebc, T):
  
    sol = np.empty((0,4))
    for local in x_cell:
        x0 = np.array([0.025, local[1], 1-local[1]-0.025, 0.5])
        args = (local[0], local[1], dgf, ebb, ebc, T)
        opts = { 'xtol' : 1E-8 }
        sol_b = root(BoronSolubility, x0, args=args, options=opts)
        sol = np.append(sol, [sol_b.x], axis = 0)
    
    return sol
	  
def BSolubilityInFCC(xsi, dgf, ebb, ebc, T):
    """
    Calculate the equilibrium solubility of B in a solution of A when C is
    present with concentration xsi. xsi can be a vector of concentration
    values. (numpy array, or any iterable type)
    """
    xb = np.empty(0)
    xb_0 = 0.025
    for xc in xsi:
        args = (xc, dgf, ebb, ebc, T)
        sol_b = fsolve(SolubilityProductEquation, xb_0, args=args)
        xb = np.append(xb, sol_b)
    
    return xb

if __name__ == "__main__":
    
    ebb, ebc, ecc = WagnerParameters_FCCA1(T)
    args = (0.15, dgf, ebb, 12.0, ecc)
    xb_0 = 0.0015
    #eqn = SolubilityProductEquation(0.0015, *args)
    
    # Trouver xb pour une valeur donnée de xsi
    xsi = np.linspace(0.0, 0.15, 100)
    xb = BSolubilityInFCC(xsi, dgf, ebb, 12.)
    
    import matplotlib.pyplot as plt
    
    plt.figure()
    plt.plot(xb, xsi, 'o-r')
    plt.xlim(0, 0.0015)
    plt.ylim(0, 0.15)
    plt.xlabel("x$_{B}$", fontsize='large')
    plt.ylabel("x$_{Si}$", fontsize='large')
    plt.show()
