function C=dGPcovSIG(s,t,ppar)

Z1=(ppar(2)+ppar(3)*s*s+1);
Z2=(ppar(2)+ppar(3)*t*t+1);
Z3=ppar(2)+ppar(3)*s*t;
Zn=sqrt(Z1*Z2);
Z=Z3/Zn;

dz=ppar(3)*(t/Zn-s*Z/Z1);
zd=ppar(3)*(s/Zn-t*Z/Z2);
C=dz*ppar(1)/sqrt(1-Z^2);

