function C=GPcovSIG(s,t,ppar)

Z1=(ppar(2)+ppar(3)*s*s+1);

Z2=(ppar(2)+ppar(3)*t*t+1);

Z3=ppar(2)+ppar(3)*s*t;

Zn=sqrt(Z1*Z2);

Z=Z3/Zn;

C=ppar(1)*asin(Z);

