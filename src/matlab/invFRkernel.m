function [value] = invFRkernel(mu, N)
%Reproducing kernel
%==========================================================================
%Purpose: Calculate the inverse Funk-Radon transform of reproducing kernel  
%         for the space of spherical harmonics of maximum degree N.
%
%  Usage: k = invFRkernel(0.1, 10);
%
%  Input:
%    mu = cos(theta)   (a scaler)
%     N = maximum degree of subspace
%
%  Output: value of function
%
%==========================================================================

% filter=[ 1.0 
%    1.000000000000000
%    1.000000000000000
%    1.000000000000000
%    1.000000000000000
%    1.000000000000000
%    1.000000000000000
%    1.000000000000000
%    1.000000000000000
%    0.789584670088749
%    0.689234601547581
%    0.600674720610018
%    0.440095247861013
%    0.330187867483138
%    0.202073539095921
%    0.141876799249721
%    0.082849296264968];

%filter = ones(size(filter));

%Check that -1 <= mu <= 1
if(abs(mu)>1.0)
    mu = sign(mu)*1;
end


%Need Legendre polynomials
legPolys = legP(mu,N);

sum = 0.0;
for i=0:2:N
    %sum = sum +  (2*i+1) * filter(i+1) * legPolys(i+1) / pnAtzero(i) ;
    sum = sum +  (2*i+1) * legPolys(i+1) / pnAtzero(i) ;
end
value = sum / (8*pi);

end





