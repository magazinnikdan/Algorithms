from sympy import roots
from sympy import symbols, simplify

alpha, eta, v_norm, c1, c2, beta = symbols('alpha eta v_norm c1 c2 beta')

eq = alpha*eta**2*(eta-beta*v_norm**2)**2-c1*(eta-beta*v_norm**2)**2-c2*(2*eta-beta*v_norm**2)
eq_simp=simplify(eq)
eta_roots=roots(eq,eta)
roots_array=[]
for k, v in eta_roots.items():
    ev_k = k.subs(beta,1)
    ev_k = ev_k.subs(v_norm,1)
    ev_k = ev_k.subs(c1, 1)
    ev_k = ev_k.subs(c2, 1)
    ev_k = ev_k.subs(alpha, 1)
    roots_array.append(float(ev_k))
print(roots_array)