# -*- coding: utf-8 -*-
"""
Created on Tue Sep 16 20:36:33 2014

@author: johannha
"""

import numpy as np
import SolProduct as SP
from fipy import *

def TracerDiffCoeffs(T):
    
    R_g = 8.314
    DB = 0.37e-4*np.exp(-166105/R_g/T)
    DSi = 10.6e-4*np.exp(-271.1e3/R_g/T)
    
    return DB, DSi
    

def equilibriumStep(x_cell, dgf, ebb, ebc, T):
    """ Compute the precipitated fraction and matrix equilibrium fractions
	    along the concentration profiles 
	    x_cell = [CBT(), CSIT(), CB(), CSI()] """
    R_g = 8.314
    sol = np.empty((0,4))
    for cell in x_cell:
        ctot_b, ctot_si = cell[0], cell[1]
        ctot_ni = 1. - ctot_b - ctot_si
        cmat_b, cmat_si = cell[2], cell[3]
        cmat_ni = 1. - cmat_b - cmat_si
        fm_old = cell[4]
        
        # Compute solubility product
        csol_si = cmat_si
        csol_b = SP.BSolubilityInFCC([csol_si], dgf, ebb, ebc, T)
        Ksp = numerix.exp( dgf / R_g / T - ebb * csol_b - ebc * csol_si )
        
        # Is the volume supersaturated? If not, will precipitates dissolve?
        is_growing_or_dissolving = ( cmat_b * cmat_ni**3  > Ksp ) | ( fm_old < 1. )
        if is_growing_or_dissolving:
            # We already know csol, but we need to estimate the precipitate fraction formed or dissolved
            vol_contents = SP.BSolInSuperSaturatedFCC( [[ctot_b, ctot_si]] , dgf, ebb, ebc, T)
            csol_b, csol_si, csol_ni, fm_new = vol_contents[0]
            # if there are no precipitate left after dissolution
            is_rapidly_dissolving = csol_b > ctot_b
            if is_rapidly_dissolving:
                sol = np.append( sol, [[ctot_b, ctot_si, ctot_ni, 1.]], axis = 0 )
                # print ("fm ? = %.8f (should be > 1) fm_old ? = %.8f " ) % (fm_new, fm_old)
            # Otherwise, the matrix fraction will be the solubility limit.
            else:
                sol = np.append(sol, [[ csol_b, csol_si, csol_ni, fm_new ]], axis = 0)
        # Not supersaturated, nothing to dissolve...
        else:
            sol = np.append(sol, [[ctot_b, ctot_si, ctot_ni, fm_old]], axis = 0)
            
    return sol[:,0], sol[:,1], sol[:,2], sol[:,3]


def ModelFun( t_max, DGf=-71152., epBB=0.0, epXB=-5.0, epXX=50.0, beBB=0., beXB=0., beXX=0.0 ):
	## Geometric parameters
    ## --------------------
    
    L = 0.0002
    nx = 500
    time_to_gap_completion = 1800.
    gap_width = 50e-6 # meters
    
    
    ## Time stepping parameters
    ## ------------------------
    
    tmax = t_max
    dt = 3.
    
    ## Physical parameters
    ## -------------------
    
    T = 1323.  # Temperature in K.
    
    ## Ni3B free energy of formation
    dgf = DGf # in J/mol
    print ("Dgf_Ni3B = %f" ) % ( dgf )
    
    ## Thermodynamic parameters
    eBB = epBB
    eXB = epXB
    eXX = epXX
    print ("epsBB = %f, epsXB = %f, epsXX = %f") % (eBB, eXB, eXX)
    
    ## Kinetic parameters
    bSiB = beXB
    bSiSi = beXX
    bBB = beBB
    print ("betaBB = %f, betaXB = %f, betaXX = %f") % (bBB, bSiB, bSiSi)
    
    ## Precipitate composition
    xP_b = 0.25
    xP_ni = 0.75
    
    ## X concentration at the interface
    IntSi = 0.10
    
    IntB = SP.BSolubilityInFCC(np.array([IntSi]), dgf, eBB, eXB, T)
    IntB = IntB.item()
    
    ## Set up the solver
    ## -----------------
    
    mesh = Grid1D(nx=nx, Lx=1.)
    
    YB = CellVariable(mesh=mesh, name='YB', hasOld=True)
    YSI = CellVariable(mesh=mesh, name='YSi', hasOld=True)
    
    # Boundary conditions
    # Input flux given by a set of Dirichlet condition at the left hand side
    # of the computation volume. The right hand side is dealt with using
    # outflux conditions (a source term in the equations...)
    YB.constrain(IntB/(1.-IntB), mesh.facesLeft)
    YSI.constrain(IntSi/(1.-IntB), mesh.facesLeft)
    
    YNI = 1. - YSI - YB
    
    CB = YB / ( 1. + YB )
    CSI = YSI / ( 1. + YB )
    
    falp = CellVariable(mesh=mesh, name='pptFr', value=1.)
    
    CBT = CB * falp + xP_b * (1. - falp)
    CSIT = CSI * falp
    CNIT = 1 - CBT - CSIT
    
    ## Interdiffusion matrix
    """
        Calculated composition dependence
        ---------------------------------
        
        D11 = uB*(1.-uB)*DB*(1.-xB)**2*(1./xB+eBB)
        D12 = uB*(1.-uB)*DB*(1.-xB)*eXB
        D21 = uSi*DSi*(1.-xB)**2*((1.-xSi)*eXB - eBB*xB)
        D22 = uSi*DSi*(1.-xB)*(1./xSi+(1.-xSi)*eXX-eXB*xB)
    """
    
    D0B, D0SI = TracerDiffCoeffs(T)
    
    DB = D0B*numerix.exp(bSiB*CSI+bBB*CB)
    DSI = D0SI*numerix.exp(bSiSi*CSI+bSiB*CB)
    
    D11 = DB*(1.-YB)*(1-CB)*(1.+eBB*CB)
    D12 = DB*CB*(1.-YB)*eXB
    D21 = DSI*CSI*(1.-CB)*((1.-CSI)*eXB - eBB*CB)
    D22 = DSI*(1.+CSI*((1.-CSI)*eXX-eXB*CB))
    
    ## Time dependent variables (for the moving boundary)
    
    lbda = gap_width / numerix.sqrt( time_to_gap_completion )
    
    t = Variable(value = 1E-6)
    v = Variable(value = -lbda/2/numerix.sqrt(t()), name='intVel')
    xst = Variable(value = -lbda * numerix.sqrt(t()), name='intPos')
    u = CellVariable(mesh=mesh, value = v * (1. - mesh.cellCenters) / (L - xst), name='u')
    
    # Sets outflow boundary up (to the right of the mesh)
    exteriorCoeff = FaceVariable(mesh, value=u.faceValue, rank=1)
    exteriorCoeff.setValue( 0., where = ~mesh.facesRight )
    
    ## The set of terms ( transient == diffusion + convection + source of stretch + outflow source )
    
    # Calculates diffusives terms (flux depends on concentration in alpha only)
    DT11 = ImplicitDiffusionTerm(coeff=(falp**2*D11/(L-xst)**2).faceValue, var=YB)
    DT12 = ImplicitDiffusionTerm(coeff=(falp**2*D12/(L-xst)**2).faceValue, var=YSI)
    DT21 = ImplicitDiffusionTerm(coeff=(falp**2*D21/(L-xst)**2).faceValue, var=YB)
    DT22 = ImplicitDiffusionTerm(coeff=(falp**2*D22/(L-xst)**2).faceValue, var=YSI)
        
    # advection term (due to the moving interface) and outflow boundary condition
    AdT1 = ImplicitSourceTerm(coeff=v/(L-xst), var=YB) + PowerLawConvectionTerm(coeff=u, var=YB) + ImplicitSourceTerm(coeff=exteriorCoeff.divergence, var=YB)
    AdT2 = ImplicitSourceTerm(coeff=v/(L-xst), var=YSI) + PowerLawConvectionTerm(coeff=u, var=YSI) + ImplicitSourceTerm(coeff=exteriorCoeff.divergence, var=YSI)
        
    eqn0 = TransientTerm(var=YB) ==  AdT1 + DT11 + DT12
    eqn1 = TransientTerm(var=YSI) == AdT2 + DT21 + DT22
    eqn = eqn0 & eqn1
    
    fppt = 0.
    
    while t() < tmax:
        
        ## Update time-dependent values    
        t.setValue(t() + dt)
        v.setValue(-lbda/2/numerix.sqrt(t()))
        xst.setValue(-lbda * numerix.sqrt(t()))
        u.setValue(v * (1. - mesh.cellCenters) / (L - xst))
        
        # update old values of diffusion profile to current ones
        YB.updateOld()
        YSI.updateOld()
        
        ## Precipitation step
        
        # Equilibrium calculation using the solubility product equation 
        # modified for non-dilute solutions.
        concentrations = zip( CBT(), CSIT(), CB(), CSI(), falp() )
        cb, csi, cni, fmat = equilibriumStep( concentrations, dgf, eBB, eXB, T)
        # Distribute the solute between matrix and precipitates
        YB.setValue( value = cb / ( 1. - cb ) )
        YSI.setValue( value = csi / ( 1. - cb ) )
        # Compute the new precipitate and matrix fractions
        fppt += np.array( (falp() - fmat) * dt, dtype=float )
        fppt = np.where( fppt < 0., 0., fppt) # Just truncation errors?
        falp.setValue( value = 1 - fppt )
        
        ToSave = CBT, CSIT, CB, CSI, falp
        ## Diffusion step
        
        res = 1e+10
        while res > 1e-6:	    
            res = eqn.sweep(dt = dt)
        
    return ToSave
