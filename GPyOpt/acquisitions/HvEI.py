# Copyright (c) 2016, the GPyOpt Authors
# Licensed under the BSD 3-clause license (see LICENSE.txt)

import numpy as np
from .base import AcquisitionBase
from scipy.special import erf
from ..util.general import get_quantiles

class AcquisitionHvEI(AcquisitionBase):
    """
    Expected improvement acquisition function 
      with Probability of Feasibility  black-box constraint handling

    Based on Gardner et. al. 2014, "Bayesian Optimization with Inequality Constraints"
      also on Gelbart et. al 2014, Gelbart 2015 and Schonlau 1997

    :param model: GPyOpt class of model
    :param model_c: list of GPyOpt class of model 
    :param space: GPyOpt class of domain
    :param optimizer: optimizer of the acquisition. Should be a GPyOpt optimizer
    :param cost_withGradients: function
    :param jitter: positive value to make the acquisition more explorative.
    :param jitter_c: list of positive values to force higher constraint compliance
    :param void_min: positive value to use in case no valid (non-constraint violating) value is available.

    .. Note:: allows to compute the Improvement per unit of cost

    """

    analytical_gradient_prediction = True

    def __init__(self, model, space, optimizer=None, cost_withGradients=None, jitter=0.01, model_c=[], jitter_c=None, void_min = 1e5, P= None, r = None):
        self.optimizer = optimizer
        super(AcquisitionHvEI, self).__init__(model, space, optimizer, cost_withGradients=cost_withGradients)
        self.jitter = jitter

        self.model_c = model_c
        self.void_min = void_min
        if jitter_c is not None:
            self.jitter_c = jitter_c
        else:
            self.jitter_c = 0.03*np.ones(len(self.model_c))

        if P is None:
            print("There are no solutions in Pareto Front to calculate acquisition function")
        else:
            self.P = P

        if r = None:
            print("There is no reference point to calculate acquisition function")
        else:
            self.r = r        

    @staticmethod
    def fromConfig(model, space, optimizer, cost_withGradients, config, model_c = []):
        return AcquisitionHvEI(model, space, model_c, optimizer, cost_withGradients, 
                               jitter=config['jitter'],jitter_c=config['jitter_c'],void_min=config['void_min'])

    def _compute_acq(self, x):
        """
        Computes the Constrained Expected Improvement per unit of cost
        """
        ########################################################################
        ########################################################################
        S = np.array(sorted(self.P, key=lambda x: x[0]))
        k = len(S)

        c2 = np.sort(S[:,1])
        c1 = np.sort(S[:,0])
        c = np.zeros((k+1,k+1))

        m1, s1 = self.model[0].predict(x)
        m2, s2 = self.model[1].predict(x)

        for i in range(k+1):
            for j in range(k-i+1):
                if (j == 0):
                    fMax2 = self.r[1]
                else:
                    fMax2 = c2[k-j]

                if (i == 0):
                    fMax1 = self.r[0]
                else:
                    fMax1 = c1[k-i]

                if (j == 0):
                    cL1 = -np.inf
                else:
                    cL1 = c1[j-1]

                if (i == 0):
                    cL2 = -np.inf
                else:
                    cL2 = c2[i-1]

                if (j == k):
                    cU1 = self.r[0]
                else:
                    cU1 = c1[j]

                if (i == k):
                    cU2 = self.r[1]
                else:
                    cU2 = c2[i]

                SM = []
                for m in range(k):
                    if (cU1 <= S[m,0]) and (cU2 <= S[m,1]):
                        SM.insert(0,[S[m,0],S[m,1]])

                phi1U, Phi1U, u1U = get_quantiles(self.jitter, cU1, m1, s1)
                phi1L, Phi1L, u1L = get_quantiles(self.jitter, cL1, m1, s1)
                phi2U, Phi2U, u2U = get_quantiles(self.jitter, cU2, m2, s2)
                phi2L, Phi2L, u2L = get_quantiles(self.jitter, cL2, m2, s2)

                ## Calculate HV 2D ##
                fMax = [fMax1, fMax2]
                M = np.array(sorted(np.array(SM), key=lambda x: x[0]))
                h = 0
                if (not(len(self.P))==0):
                    n = len(M)
                    for l in range(n):
                        if (l == 0):
                            h = h + (fMax[0]-M[l,0])*(fMax[1] - M[l,1])
                        else:
                            h = h + (fMax[0]-M[l,0]) * (M[l-1,1] - M[l,1])
                 ####################

                exipsi1U = s1* phi1U + (fMax1-m1)*Phi1U
                exipsi1L = s1* phi1L + (fMax1-m1)*Phi1L
                exipsi2U = s2* phi2U + (fMax2-m2)*Phi2U
                exipsi2L = s2* phi2L + (fMax2-m2)*Phi2L

                sPlus = np.copy(h)
                Psi1 = exipsi1U - exipsi1L
                Psi2 = exipsi2U - exipsi2L
                GaussCDF1 = Phi1U - Phi1L
                GaussCDF2 = Phi2U - Phi2L
                c[i,j] = Psi1*Psi2 - sPlus*GaussCDF1*GaussCDF2

        f_acqu = np.sum(np.sum(np.maximum(c,0)))

        ########################### END #######################################

        ########################################################################
        
        for ic,mdl_c in enumerate(self.model_c):
            m_c, s_c = mdl_c.predict(x)
            
            if isinstance(s_c, np.ndarray):
                s_c[s_c<1e-10] = 1e-10
            elif s_c< 1e-10:
                s_c = 1e-10
            
            z_c = (m_c-self.jitter_c[ic])/s_c    # Implement constraint of type c(x) >= 0
            Phi_c = 0.5*(1+erf(z_c/np.sqrt(2.))) # contrained cdf from erf
            
            f_acqu[...] = f_acqu[...] * Phi_c[...]
        
        ########################################################################

        return f_acqu
    
    def _compute_acq_withGradients(self, x):
        """
        Computes the Constrained Expected Improvement and its derivative
        """
        ########################################################################
        ########################################################################
        S = np.array(sorted(self.P, key=lambda x: x[0]))
        k = len(S)

        c2 = np.sort(S[:,1])
        c1 = np.sort(S[:,0])
        c = np.zeros((k+1,k+1))
        
        m1, s1, dmdx1, dsdx1 = self.model[0].predict_withGradients(x)
        m2, s2, dmdx2, dsdx2 = self.model[1].predict_withGradients(x)
        

        for i in range(k+1):
            for j in range(k-i+1):
                if (j == 0):
                    fMax2 = self.r[1]
                else:
                    fMax2 = c2[k-j]

                if (i == 0):
                    fMax1 = self.r[0]
                else:
                    fMax1 = c1[k-i]

                if (j == 0):
                    cL1 = -np.inf
                else:
                    cL1 = c1[j-1]

                if (i == 0):
                    cL2 = -np.inf
                else:
                    cL2 = c2[i-1]

                if (j == k):
                    cU1 = self.r[0]
                else:
                    cU1 = c1[j]

                if (i == k):
                    cU2 = self.r[1]
                else:
                    cU2 = c2[i]

                SM = []
                for m in range(k):
                    if (cU1 <= S[m,0]) and (cU2 <= S[m,1]):
                        SM.insert(0,[S[m,0],S[m,1]])

                phi1U, Phi1U, u1U = get_quantiles(self.jitter, cU1, m1, s1)
                phi1L, Phi1L, u1L = get_quantiles(self.jitter, cL1, m1, s1)
                phi2U, Phi2U, u2U = get_quantiles(self.jitter, cU2, m2, s2)
                phi2L, Phi2L, u2L = get_quantiles(self.jitter, cL2, m2, s2)

                ## Calculate HV 2D ##
                fMax = [fMax1, fMax2]
                M = np.array(sorted(np.array(SM), key=lambda x: x[0]))
                h = 0
                if (not(len(self.P))==0):
                    n = len(M)
                    for l in range(n):
                        if (l == 0):
                            h = h + (fMax[0]-M[l,0])*(fMax[1] - M[l,1])
                        else:
                            h = h + (fMax[0]-M[l,0]) * (M[l-1,1] - M[l,1])
                ####################

                exipsi1U = s1* phi1U + (fMax1-m1)*Phi1U
                exipsi1L = s1* phi1L + (fMax1-m1)*Phi1L
                exipsi2U = s2* phi2U + (fMax2-m2)*Phi2U
                exipsi2L = s2* phi2L + (fMax2-m2)*Phi2L

                sPlus = np.copy(h)
                Psi1 = exipsi1U - exipsi1L
                Psi2 = exipsi2U - exipsi2L
                GaussCDF1 = Phi1U - Phi1L
                GaussCDF2 = Phi2U - Phi2L
                c[i,j] = Psi1*Psi2 - sPlus*GaussCDF1*GaussCDF2
        f_acqu = np.sum(np.sum(np.maximum(c,0)))

            ## HVEI Gradient ##
        mu = np.array([[m1, m2]])
        s = np.array([[s1, s2]])
        dmu = np.array([dmdx1, dmdx2])
        ds = np.array([dsdx1, dsdx2])
        
        c1_g = np.sort(S[:,0])[::-1]
        
        # Define the lower and upper bound for distribution
        
        a = -np.inf
        b = np.inf
        cL2 = a
        
        n = np.size(mu[0][0])
        n_s = np.size(ds[1])
        
        dehvi = np.zeros((1,n_s))
        
        c_g = np.zeros((n, k))
        Px_L = np.zeros((n,1))
        
        for i in range(k+1):
            if i == 0:
                cU1 = self.r[0]
            else:
                cU1 = c1_g[i-1]
            
            if i != (k):
                cU2 = c2[i]
                cL1 = c1_g[i]
            else:
                cU2 = self.r[1]
                cL1 = a
        
            for j in range(n):
                mu1 = mu[j,0]
                mu2 = mu[j,1]
                s1 = s[j,0]
                s2 = s[j,1]
                
                phi1U, Phi1U, u1U = get_quantiles(self.jitter, cU1, mu1, s1)
                phi1L, Phi1L, u1L = get_quantiles(self.jitter, cL1, mu1, s1)
                phi2U, Phi2U, u2U = get_quantiles(self.jitter, cU2, mu2, s2)
                
                exipsi1U_g = s1* phi1U + (cU1-mu1)*Phi1U
                exipsi1UL_g = s1* phi1L + (cU1-mu1)*Phi1L
                exipsi2U_g = s2* phi2U + (cU2-mu2)*Phi2U
                
                Psi1 = exipsi1U_g - exipsi1UL_g
                Psi2 = exipsi2U_g
                
                pdfny1i, cdfny1i, u1i = get_quantiles(self.jitter, cL1, mu1, s1)
                pdfny2i, cdfny2i, u21 = get_quantiles(self.jitter, cU2, mu2, s1)
                pdfny1i1, cdfny1i1, u1i1 = get_quantiles(self.jitter,cU1, mu1,s1)
                
                if i != k:
                    Px = pdfny1i
                    dist = cU1 - cL1
                    R = dist * (pdfny1i*((mu1-cL1)/(s1**2)*ds[0,:] - dmu[0,:]/s1) * Psi2 + (pdfny2i * ds[1,:] - cdfny2i * dmu[1,:]) * cdfny1i)
                    eq2_2 = -(( -dist/s1 * pdfny1i - cdfny1i) * dmu[0,:] +(pdfny1i - (cL1-mu1) * dist/s1**2 * pdfny1i) * ds[0,:]) *Psi2
                    
                else:
                    R = dist * (pdfny2i * ds[1,:] - cdfny2i * dmu[1,:]) * cdfny1i
                    eq2_2 = -((-dist/s1 * pdfny1i - cdfny1i) * dmu[0,:] + (pdfny1i)) * Psi2
            
                eq2_1 = (pdfny1i1 * dmu[0,:] - cdfny1i1 * dmu[0,:]) * Psi2
                eq2_3 = (pdfny2i * ds[1,:] - cdfny2i * dmu[1,:]) * Psi1
                
                
                df_acqu = R + eq2_1 + eq2_2 + eq2_3 + df_acqu

            ########################### END #######################################

        Phis_c   = []
        dPFsdx_c = []
        for ic,mdl_c in enumerate(self.model_c):
            m_c, s_c, dmdx_c, dsdx_c = mdl_c.predict_withGradients(x)
            
            if isinstance(s_c, np.ndarray):
                s_c[s_c<1e-10] = 1e-10
            elif s_c< 1e-10:
                s_c = 1e-10
                
            z_c = (m_c-self.jitter_c[ic])/s_c    # Implement constraint of type c(x) >= 0
            phi_c = np.exp(-0.5*(z_c**2))/(np.sqrt(2.*np.pi)*s_c)
            Phi_c = 0.5*(1+erf(z_c/np.sqrt(2.))) # contrained cdf from erf
            dPFsdx = phi_c*(dmdx_c-((m_c-self.jitter_c[ic])/s_c)*dsdx_c)
            
            Phis_c.append(np.copy(Phi_c))
            dPFsdx_c.append(np.copy(dPFsdx))
        
        ########################################################################

        f_acqu_c = np.copy(f_acqu)
        t1 = np.copy(df_acqu)
        t2 = np.zeros(df_acqu.shape)
        for i in range(len(self.model_c)):
            f_acqu_c = f_acqu_c * Phis_c[i]
            t1 = t1 * Phis_c[i]
            
            g2 = np.copy(dPFsdx_c[i])
            for j in range(len(self.model_c)):
                if(j==i):
                    continue
                g2 = g2 * Phis_c[j]
            
            t2 += f_acqu * g2
            
        df_acqu_c = t1+t2
        
        ########################################################################
        
        return f_acqu_c, df_acqu_c
