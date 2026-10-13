"""Unified EP baseline comparison on the manuscript's 2-D benchmark.
Floating-point diagnostics only; not a validated interval certificate.
Requires numpy, scipy.
"""
import json
import numpy as np
from scipy.optimize import root


def benchmark():
    # F(x,y;theta) = (x-x^3+y/5+theta, -y+(x-x^3)/10)
    z0=np.array([1.0,0.0]); theta=0.0
    A=np.array([[-2.0,0.2],[-0.2,-1.0]]) # Jacobian at (1,0)
    B=np.array([[1.0],[0.0]])
    q=np.array([0.7,0.2]) # gradient C at branch: x-0.3, y+0.2
    exact=float((B.T@np.linalg.solve(A.T,q))[0])
    vf=float((B.T@np.linalg.solve(A,q))[0])
    Aa=(A-A.T)/2
    def F(z):
        x,y=z; return np.array([x-x**3+y/5, -y+(x-x**3)/10])
    def gradC(z):
        x,y=z; return np.array([x-.3,y+.2])
    def asym_g(z,beta): return F(z)-2*Aa@(z-z0)-beta*gradC(z)
    rows=[]
    for beta in [0.08,0.04,0.02,0.01,0.005]:
        roots=[]; residuals=[]
        for b in [beta,-beta]:
            sol=root(lambda z:asym_g(z,b),z0,tol=1e-12)
            roots.append(sol.x); residuals.append(float(np.linalg.norm(asym_g(sol.x,b))))
        # For this benchmark, parameter update uses first component of state difference.
        est=float((roots[0][0]-roots[1][0])/(2*beta))
        rows.append({'beta':beta,'asymep_estimate':est,'asymep_abs_error':abs(est-exact),
                     'root_residual_plus':residuals[0],'root_residual_minus':residuals[1]})
    # Dyadic ideal solution: d=beta A^{-T}q, algebraic stationary identity.
    dy=[]
    for beta in [0.1,0.01,0.001,0.0001]:
        d=beta*np.linalg.solve(A.T,q)
        dy.append({'beta':beta,'ideal_signal':float((B.T@(d/beta))[0]),'abs_error':abs(float((B.T@(d/beta))[0])-exact)})
    return {'branch':[1.0,0.0], 'A':A.tolist(),'B':B.tolist(),'q':q.tolist(),
            'implicit_gradient_exact':exact,'standard_vf_ep_infinitesimal':vf,
            'standard_vf_abs_bias':abs(vf-exact),'asymep_finite_beta':rows,'dyadic_ideal_finite_beta':dy}

if __name__=='__main__':
    print(json.dumps(benchmark(),indent=2))
