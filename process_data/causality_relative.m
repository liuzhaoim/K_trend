% Purpose: Estimate normalized Liang-Kleeman information flow between two time series.
% Main inputs: Two aligned time series and a differencing interval.
% Main outputs: Information flow and uncertainty estimates.
% Notes: Update placeholder paths and product/version switches before running.

function [T21, t21, err90, err95, err99] = causality_relative(xx1, xx2, np)
% 
% function [T21, err90, err95, err99] = causality_est(x1, x2, np)
% e.g, [T21, err90, err95, err99] = causality_est(x1, x2, 1)
%
%
% Estimate T21, the information transfer from series X2 to series X1 
% dt is taken to be 1.
%
% On input:
%    X1, X2: the series
%    np:  # of time steps for differencing
%	  (usually  1, but if highly chaotic and densely sampled, use 2)
%
% On output:
%    T21:  info flow from X2 to X1	(Note: Not X1 -> X2!)
%    t21:  percent of T21 (provided by Yineng Rong)***********
%    err90: standard error at 90% significance level
%    err95: standard error at 95% significance level
%    err99: standard error at 99% significance level
%


dt = 1;	


[nm, one] = size(xx1);


dx1(:,1) = (xx1(1+np:nm, 1) - xx1(1:nm-np, 1)) / (np*dt);
 x1(:,1) = xx1(1:nm-np, 1);

dx2(:,1) = (xx2(1+np:nm, 1) - xx2(1:nm-np, 1)) / (np*dt);
 x2(:,1) = xx2(1:nm-np, 1);

clear xx1 xx2;

N = nm-np;



C = cov(x1, x2);

dC(1,1) = sum((x1 - mean(x1)) .* (dx1 - mean(dx1))); 
dC(1,2) = sum((x1 - mean(x1)) .* (dx2 - mean(dx2))); 
dC(2,1) = sum((x2 - mean(x2)) .* (dx1 - mean(dx1))); 
dC(2,2) = sum((x2 - mean(x2)) .* (dx2 - mean(dx2))); 
dC = dC / (N-1);

dCC=sum((dx1 - mean(dx1)) .* (dx1 - mean(dx1)))/(N-1);
% C_infty = cov(x1, x2);
C_infty = C;

detc = det(C);

a11 = C(2,2) * dC(1,1) - C(1,2) * dC(2,1);
a12 = -C(1,2) * dC(1,1) + C(1,1) * dC(2,1);
% a21 = -C(1,2) * dC(1,2) + C(1,1) * dC(2,2);
% a22 = C(2,2) * dC(1,2) - C(1,2) * dC(2,2);

a11 = a11 / detc;
a12 = a12 / detc;
% a21 = a21 / detc;
% a22 = a22 / detc;

f1 = mean(dx1) - a11 * mean(x1) - a12 * mean(x2);
% f2 = mean(dx2) - a21 * mean(x1) - a22 * mean(x2);

R1 = dx1 - (f1 + a11*x1 + a12*x2);
% R2 = dx2 - (f2 + a21*x1 + a22*x2);

Q1 = sum(R1 .* R1);
% Q2 = sum(R2 .* R2);

b1 = sqrt(Q1 * dt / N);
% b2 = sqrt(Q2 * dt / N);

%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% covariance matrix of the estimation of (f1, a11, a12, b1)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
NI(1,1) = N * dt / b1/b1;
NI(2,2) = dt/b1/b1 * sum(x1 .* x1);
NI(3,3) = dt/b1/b1 * sum(x2 .* x2);
NI(4,4) = 3*dt/b1^4 * sum(R1 .* R1) - N/b1/b1;
NI(1,2) = dt/b1/b1 * sum(x1);
NI(1,3) = dt/b1/b1 * sum(x2);
NI(1,4) = 2*dt/b1^3 * sum(R1);
NI(2,3) = dt/b1/b1 * sum(x1 .* x2);
NI(2,4) = 2*dt/b1^3 * sum(R1 .* x1);
NI(3,4) = 2*dt/b1^3 * sum(R1 .* x2);

NI(2,1) = NI(1,2);
NI(3,1) = NI(1,3);    NI(3,2) = NI(2,3);
NI(4,1) = NI(1,4);    NI(4,2) = NI(2,4);   NI(4,3) = NI(3,4);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

invNI = inv(NI);
var_a12 = invNI(3,3);		%% ???? The marginal?
%
% approx. variance of a12, corr. to variance of T21
%


%
% Information transfer: T21 = C12/C11 * a12
%			T12 = C21/C22 * a21
%
 T21 = C_infty(1,2)/C_infty(1,1) * (-C(2,1)*dC(1,1) + C(1,1)*dC(2,1)) / detc;
% T12 = C_infty(2,1)/C_infty(2,2) * (-C(1,2)*dC(2,2) + C(2,2)*dC(1,2)) / detc;
%

var_T21 = (C_infty(1,2)/C_infty(1,1))^2 * var_a12;

% Local Entropy Generation
% H1 = p

p=(C(2,2)*dC(1,1)-C(1,2)*dC(2,1))/detc;
H1=p;

% Noise
q=(-C(1,2)*dC(1,1)+C(1,1)*dC(2,1))/detc;
Hn=(np*dt)/(2*C(1,1))*( dCC+p^2*C(1,1)+q^2*C(2,2)-2*p*dC(1,1)-2*q*dC(2,1)+2*p*q*C(1,2));

Z21=abs(T21)+abs(H1)+abs(Hn);
t21=T21/Z21;
%
% From the standard normal distribution table, 
% significance level alpha=95%, z=1.96
%		           99%, z=2.56
%			   90%, z=1.65
%
	z99 = 2.56;
	z95 = 1.96;
	z90 = 1.65;

	err90 = sqrt(var_T21) * z90;
	err95 = sqrt(var_T21) * z95;
	err99 = sqrt(var_T21) * z99;
