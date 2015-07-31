function R = rhomatrix(C)
%RHOMATRIX normalized correlation coefficents.
%   R = RHOMATRIX(C) returns a matrix of correlation coefficients, R,
%   given a covariance matrix, C.
%
%   If C is the covariance matrix, C = COV(X), then RHOMATRIX(C) is
%   the matrix whose (i,j)'th element is
%  
%        R(i,j) =  C(i,j)/SQRT(C(i,i)*C(j,j)).
%
%-----------------------------------------------------------------
%    History:
%    Date            Who         What
%    -----------     -------     -----------------------------
%    03-10-2004      rme         Created and written.

d = sqrt(diag(C));

R = C ./ (d*d');
