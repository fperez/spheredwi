function [val] = kernelRep(evalPnt, quadPnts, coefs, maxDegree)
%Reproducing kernel representation
%==========================================================================
%Purpose: Calculate the reproducing kernel for the space of spherical
%         harmonics of maximum degree N.
%
%  Usage: k = kernel(0.1, 10);
%
%  Input:
%     evalPnt -- point on the sphere to evaluate representation
%    quadPnts -- quadrature points
%       coefs -- coefficients of the representation 
%   maxDegree -- maximum degree of subspace
%
%  Output: value of representation based on reproducing kernel
%
%==========================================================================

n  = length(quadPnts);
val = 0.0;

for i=1:n
    mu = dot(evalPnt,quadPnts(i,1:3));
    val = val + coefs(i) * kernel(mu,maxDegree);
end

end







