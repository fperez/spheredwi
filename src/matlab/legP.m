function [ p ] = legP( mu, N )
%Legendre polynomials: calculation of Legendre polynomials up degree N
%==========================================================================
%Purpose: 
%
%  Usage: [p] = legP(mu, N);
%
%  Calculates: Legendre polynomials up to and including degree N evaluated
%              at mu = cos(theta)
%
%  Input:
%       mu = cos(theta)
%        N = highest degree
%
%  Output: vector of polynomial evaluations p(1) = p0(mu), p(2) = p1(mu),
%          p(3) = p2(mu), ... p(N) = pN-1(mu), p(N+1) = pN(mu).
%==========================================================================
p = zeros([N+1,1]);
p(1) = 1;
p(2) = mu;
for i=2:N
    p(i+1) = (2*(i-1) + 1) * mu * p(i) / ((i-1) + 1) - (i-1) * p(i-1) / ((i-1)+1);
end

end

