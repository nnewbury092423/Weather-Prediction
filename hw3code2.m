close all

C1 = 0:.00000001:10^-4;
C2 = 10^-7 ;
T = 298;
R = 8.31;
epsilon = 4
epsilon0 = 8.85*10^-12
F = 96485.3321;
q = 1.6*10^(-19);
Clcon = -10^(-4);
% volume in liters
r = .5*10^(-6);
n = 6.022*10^23
C = epsilon*epsilon0*4*pi*r^2/(8*10^-9);

Vol = (4/3)*pi*r^3* 1000;
figure
dphi = Vol.*q.*n*((-log(C1) - 5).*10^-6 + C1 + Clcon)/C;
title('membrane voltage')
plot(C1,dphi);
figure
f =C2.*exp(-F.*dphi/(R*T)) - C1;
plot(C1,f)
title('Concentration Function (Zero is Concentration Solution)')
disp(f)

%H+ Concentration = 5.1 * 10^-5

 Ce = 5.1*10^-5
 dphi = -log(Ce/10^-7)*R*T/F 
 %dphi = Vol*q*n*((-log(Ce) - 5)*10^-6 + Ce + Clcon)/C