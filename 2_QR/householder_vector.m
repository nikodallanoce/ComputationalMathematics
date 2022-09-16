function [u, s] = householder_vector(x)
% Householder vector
% Inputs:
%       x           input array    
%
% Output:
%       u           householder vector
%       s           norm of input array
%
% Reference:
%       Algorithm 3 from our report.
%
% Created by Niko Dalla Noce, Alessandro Ristori and Simone Rizzo

s = norm(x);
if x(1) >= 0, s = -s; end
v = x;
v(1) = v(1) - s;
u = v / norm(v);
end

