dphi = -.08:.0001:.08
%dphi = 0:.03:.03
% cosntants
C1 = .1;
C2 = .01;
Pc = 10^-12;
T = 298;
R = 8.31;
F = 96485.3321;
Z = 1;
%
XA = R*T*log(C2/C1) - F*.03
%beta determination
Beta = -Z.*F.*dphi./(R*T);
%Jc determination 
plot(dphi,Beta);
Jc = -Pc.*Beta.*(C1-C2.*exp(Beta))./(1-exp(Beta));
plot(dphi,Jc);
title('J-V Relation')
xlabel('dphi (V)')
ylabel('Flux(Moles/(S-M^2)')

Powercom = -8.6*10^3*.1642*10^-13