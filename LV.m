function dx=LV(t,x,theta)

dx=zeros(2,1);

alpha=theta(1); 
beta=theta(2); 
gamma=theta(3); 
delta=theta(4);

dx(1) = x(1)*(alpha-beta*x(2));
dx(2) = -x(2)*(gamma-delta*x(1));

