function [ s ] = randSig(u,b,n)
%Create random signal on the sphere
%============================================%
%  D = diffusion tensor                      %
%  u = unit vector                           %
%  n = # of signal components
%
% lambda1 = 1700e-6 mm^2/s --typical #s
% lambda2 =  300e-6  "
% lambda3 =  300e-6  "
%       b = 3000 s/mm^2
%============================================%

%Diffusion tensor parameters
lambda1 = 1700e-6;
lambda2 =  300e-6;
lambda3 =  300e-6;


D1 = diag([lambda1 lambda2 lambda3]);   %diagonal diffusion tensor for "prolate white matter"
D2 = diag([lambda2 lambda1 lambda3]);     

V1 = [1 0 0; 0 0 0; 0 0 0];             %orthonormal e-vectors of diffusion tensor
V2 = [1 0 0; 0 1 0; 0 0 0] ./ sqrt(2);  %   ""

u1p = transpose(V1)*u; %Change basis to diagonalize diffusion tensor
u2p = transpose(V2)*u;


if(n==1)
  
    s = exp(-b * dot(u1p, D1*u1p) );   %Single mode
   
elseif(n==2)
    coef = rand([2,1]);          %Take random strengths
    coef = coef./norm(coef,1);   %Normalize to unity, i.e., taking convex combination
    
    s = coef(1)*exp(-b * dot(u1p, D1*u1p) ) + coef(2)*exp(-b * dot(u2p, D2*u2p) );
    
else
    
    exp(-200 * (1 - dot([0 0 1],u))); %

end

end
