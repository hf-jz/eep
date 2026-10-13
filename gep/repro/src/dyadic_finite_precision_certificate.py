"""Finite-precision certificate for the Dyadic EP difference solve on the paper benchmark.

The numerical candidate is produced by ordinary float64 linear algebra. Every
float is then interpreted as its exact binary rational value and the residual
is recomputed using Fraction arithmetic. This certifies the stationary linear
solve's finite-precision residual contribution. The midpoint is the exactly
representable branch root (1,0); this does NOT certify an ODE trajectory or a
noisy/inexact midpoint computation.
Python standard library + NumPy only.
"""
from fractions import Fraction as Q
import json
import numpy as np

Aq = ((Q(-2), Q(1,5)), (Q(-1,5), Q(-1)))
q_exact = (Q(7,10), Q(1,5))
B = (Q(1), Q(0))

def inv2(M):
    a,b=M[0]; c,d=M[1]; det=a*d-b*c
    return ((d/det,-b/det),(-c/det,a/det))
def transpose(M): return tuple(tuple(M[j][i] for j in range(len(M)) ) for i in range(len(M[0])))
def mv(M,v): return tuple(sum(M[i][j]*v[j] for j in range(len(v))) for i in range(len(M)))
def norm_inf(M): return max(sum(abs(v) for v in row) for row in M)
def fq_float(x): return Q.from_float(float(x))

AT=transpose(Aq); ATinv=inv2(AT)
truth_state=mv(ATinv,q_exact)
truth=sum(B[i]*truth_state[i] for i in range(2))
inv_norm=norm_inf(ATinv)
A_float=np.array([[-2.0, 0.2],[-0.2,-1.0]],dtype=np.float64)
q_float=np.array([0.7,0.2],dtype=np.float64)
rows=[]
for beta in (0.1,0.01,0.001,0.0001):
    d_float=float(beta)*np.linalg.solve(A_float.T,q_float)
    # Interpret the actual float outputs exactly as rational numbers.
    d=tuple(fq_float(v) for v in d_float)
    bq=fq_float(beta)
    # Exact residual r = A^T d - beta q, computed with rational arithmetic.
    Ad=mv(AT,d)
    r=tuple(Ad[i]-bq*q_exact[i] for i in range(2))
    r_inf=max(abs(v) for v in r)
    err_bound=inv_norm*r_inf/abs(bq)
    # Parameter direction is first component because B=(1,0).
    observed=d[0]/bq
    actual_error=abs(observed-truth)
    assert actual_error <= err_bound
    rows.append({
        'beta_float': beta,
        'd_float': [float(v) for v in d_float],
        'residual_exact': [str(v) for v in r],
        'residual_inf_exact': str(r_inf),
        'residual_inf_decimal': float(r_inf),
        'inverse_transpose_inf_norm_exact': str(inv_norm),
        'gradient_estimate_exact': str(observed),
        'true_gradient_exact': str(truth),
        'actual_gradient_error_exact': str(actual_error),
        'actual_gradient_error_decimal': float(actual_error),
        'certified_error_upper_exact': str(err_bound),
        'certified_error_upper_decimal': float(err_bound),
        'certificate_passed': actual_error <= err_bound
    })
result={
 'certificate_scope':'finite-precision stationary linear solve; exact branch midpoint (1,0); no ODE trajectory, midpoint uncertainty, or model uncertainty included',
 'arithmetic':'each float64 output converted to its exact binary rational, then residual and norm bound computed exactly with Fraction',
 'A_exact':[[str(v) for v in row] for row in Aq],
 'q_exact':[str(v) for v in q_exact],
 'true_adjoint_state_exact':[str(v) for v in truth_state],
 'true_gradient_exact':str(truth),
 'inverse_transpose_inf_norm_exact':str(inv_norm),
 'rows':rows,
 'all_certificates_pass':all(x['certificate_passed'] for x in rows)
}
print(json.dumps(result,indent=2))
