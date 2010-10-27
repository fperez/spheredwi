function [k] = kernel(mu, N)
%Reproducing kernel
%==========================================================================
%Purpose: Calculate the reproducing kernel for the space of spherical
%         harmonics of maximum degree N.
%
%  Usage: k = kernel(0.1, 10);
%
%  Input:
%    mu = cos(theta)   (a scaler)
%     N = maximum degree of subspace
%
%  Output: value of sum( diag((1/(4*pi)) * (2 * (0:1:N) + 1)) * 
%                        legendreP(mu, (0:1:N))', 1)
%          based on the Christoffel-Darboux formula.
%
%==========================================================================

%Check that -1 <= mu <= 1
if(abs(mu)>1.0)
    mu = sign(mu)*1;
end

%Based on GB and Brandon Jones' notes
legPolys = legP(mu,N);
gOld = 1;
for i=1:N
    gNew  = (2*i + 1) * legPolys(i+1) / (i + 1) + i * gOld / (i + 1);
    gOld  = gNew;
end
k = (N+1) * gNew / (4*pi);

end
