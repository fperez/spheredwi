function [ R ] = rotation3Dz(theta)
%3dRotation: creates 3D rotation matrix for rotation about z-axis
%   
R = zeros(3,3);

R = [ cos(theta) sin(theta) 0.0;
     -sin(theta) cos(theta) 0.0;
      0.0        0.0        1.0];

%--below is general rotation matrix
%R = [cos(theta)*cos(psi) -cos(phi)*sin(psi)+sin(phi)*sin(theta)*cos(psi)    sin(phi)*sin(psi)+cos(phi)*sin(theta)*cos(psi);
%     cos(theta)*sin(psi)  cos(phi)*cos(psi) + sin(phi)*sin(theta)*sin(psi) -sin(phi)*cos(psi) + cos(phi)*sin(theta)*sin(psi);
%    -sin(theta)           sin(phi)*cos(theta)                                    cos(phi)cos(theta)];


end

