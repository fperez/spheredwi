function [ A ] = matrix(qpnts,spnts,np,ns,N)
%Create matrix associated with kernel interpolation
%qpnts = quadrature points
%spnts = sample points
%np    = number of points in grid
%ns    = number of sample points
%N     = maximum degree of spherical harmonic subspace

%Initialize
A = zeros(ns,np);

%Create matrix
for i=1:ns
    
    for j=1:np
        
        cosTheta = dot(spnts(i,:),qpnts(j,:));
        
        if(abs(cosTheta)>1) 
   
            cosTheta = 1.0 * sign(cosTheta);
            
        end
        
        A(i,j) = invFRkernel(cosTheta, N);
        
    end
    
end

end

