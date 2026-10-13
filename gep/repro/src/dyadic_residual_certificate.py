"""Exact-rational residual-only certificate for Dyadic EP's linear benchmark.
This certifies the algebraic residual contribution only, with midpoint fixed exactly.
It is NOT an end-to-end finite-time ODE/integrator certificate.
Python standard library only.
"""
from fractions import Fraction as Q
import json

# At the right branch: A = [[-2, 1/5],[-1/5,-1]], q=(7/10,1/5), B=(1,0).
A=((Q(-2),Q(1,5)),(Q(-1,5),Q(-1)))
q=(Q(7,10),Q(1,5))

def inv2(M):
    a,b=M[0]; c,d=M[1]
    det=a*d-b*c
    return ((d/det,-b/det),(-c/det,a/det))

def transpose(M): return tuple(tuple(M[j][i] for j in range(len(M)) ) for i in range(len(M[0])))
def matvec(M,v): return tuple(sum(M[i][j]*v[j] for j in range(len(v))) for i in range(len(M)))
def norm_inf_matrix(M): return max(sum(abs(v) for v in row) for row in M)

AT=transpose(A)
AT_inv=inv2(AT)
exact=matvec(AT_inv,q)
B=(Q(1),Q(0))
exact_signal=sum(B[i]*exact[i] for i in range(2))
inv_norm=norm_inf_matrix(AT_inv)
residual_inf=Q(1,10**10)
rows=[]
for beta in (Q(1,10),Q(1,100),Q(1,1000),Q(1,10000)):
    # Exact norm inequality: ||A^{-T} r_d / beta||_inf <= ||A^{-T}||_inf ||r_d||_inf / |beta|
    bound=inv_norm*residual_inf/abs(beta)
    rows.append({'beta':str(beta),'beta_decimal':float(beta),'residual_inf_bound':str(residual_inf),
                 'inverse_transpose_inf_norm_exact':str(inv_norm),
                 'certified_gradient_error_bound_exact':str(bound),
                 'certified_gradient_error_bound_decimal':float(bound),
                 'true_gradient_exact':str(exact_signal)})
print(json.dumps({'scope':'residual-only; midpoint is exact; no ODE integration error included',
                  'A':[[str(v) for v in row] for row in A],
                  'q':[str(v) for v in q], 'A_transpose_inverse':[[str(v) for v in row] for row in AT_inv],
                  'true_state_adjoint':[str(v) for v in exact], 'true_gradient':str(exact_signal),
                  'inverse_transpose_inf_norm':str(inv_norm), 'rows':rows},indent=2))
