function [ s ] = randSig(u,b,n,theta)
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
lambda2 = 300e-6;
lambda3 = 300e-6;


rotationMatrix = rotation3Dz(theta);

D1 = diag([lambda1 lambda2 lambda3]);   %diagonal diffusion tensor for "prolate white matter"
D2 = diag([lambda1 lambda2 lambda3]);
D3 = D1;

V1 = [1 0 0; 0 1 0; 0 0 1];  %orthonormal e-vectors of diffusion tensor
V2 = [1 0 0; 0 1 0; 0 0 1];  %   ""
V3 = V1;

V2 = rotationMatrix*V2;
V3 = rotationMatrix*rotationMatrix*V3;

u1p = transpose(V1)*u; %Change basis to diagonalize diffusion tensor
u2p = transpose(V2)*u;
u3p = transpose(V3)*u;


if(n==1)
  
    s = exp(-b * dot(u1p, D1*u1p) );   %Single mode
   
elseif(n==2)
    
    s = 0.5*exp(-b * dot(u1p, D1*u1p) ) + 0.5*exp(-b * dot(u2p, D2*u2p) );
       
elseif(n==3)
    
    s = ((1/3)*exp(-b * dot(u1p, D1*u1p) ) + (1/3)*exp(-b * dot(u2p, D2*u2p) ) + (1/3)*exp(-b * dot(u3p, D3*u3p) ));
    
end

end
