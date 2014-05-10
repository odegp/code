function C=GPcov(s,t,ppar)
C=ppar(1)*exp(-ppar(2)*(s-t)^2);