function [u, s] = householder_vector(x)
% Householder vector
% Inputs:
%       x           
%
% Output:
%       Q1y         product between the orthogonal matrix Q1 and y, if y is not
%                passed then only Q1 is returned
%       R1          upper triangular matrix
%
% Reference:
%       Algorithm 4 from our report.
%
% Created by Niko Dalla Noce, Alessandro Ristori and Simone Rizzo
s = norm(x);
if x(1) >= 0, s = -s; end
v = x;
v(1) = v(1) - s;
u = v / norm(v);
end

