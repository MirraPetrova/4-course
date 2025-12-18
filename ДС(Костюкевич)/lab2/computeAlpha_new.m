function computeAlpha

% ==========================================================

clear all,
close all,
clc
 
warning off


A = [0 1; 1 0];
B = [0.5; 0.5];
Q = [0.5 0;0 0.5];
R = 1.0;

n = length(A(1,:));
m = length(B(1,:));

%% 1: Local Controller
disp('Step 1')

% Riccati
P_LQR = care(A,B,Q,R);

K = -inv(R)*B.'*P_LQR

AK = A+B*K;
disp('eig(AK)')
eig(AK) % eigenvalues of AK

disp('______________________________')

%% 2: Searching P
disp('Step 2')

kappa = 0.95 % from condition

if kappa >= -max(real(eig(AK)))
    error('kappa >= -max(real(eig(AK)))')
end

% Equation Lyapunova
P=lyap((A+B*K+kappa*eye(n)), Q+K'*R*K); % from theory


disp('______________________________')

%% 3: Searching alpha_1
disp('Step 3')

u_min = -2*ones(m,1); % from condition
u_max = 2*ones(m,1);

uu = sdpvar(m,1);

MU = [ uu >= u_min, uu <= u_max ];

U = Polyhedron(MU);

U.minHRep();

clear('uu','MU')

options = sdpsettings('solver','fmincon','verbose',0);

mu = length(U.A(:,1));

alpha_1_vec=[];

for k = 1:mu
    
    x_opt = sdpvar(n,1);
   
    H = x_opt' * P * x_opt;
    
    constraints = [U.A(k,:) * K * x_opt == U.b(k,:)];

    sol = optimize(constraints,-H,options);
    alpha_1_vec = [alpha_1_vec, double(H)];
    
end
clear('k','constraints','x_opt','H','sol','mu','mx')

alpha_1 = min(alpha_1_vec)


disp('______________________________')

%%  4: Searching alpha
disp('Step 4')
% alpha<=alpha_l,
%L_Phi<=L_Phi_max.

L_Phi_max = kappa*min(eig(P))/ norm(P,2); % from theory (1)

alpha_ub = alpha_1;
alpha_lb = 0;
L_Phi = FcnL_phi(AK,K,P,alpha_1);
alpha = alpha_1;
exitflag = 1;
nn = 1;

n_max = 100;

while exitflag == 1 && nn <= n_max
    
    alpha_old = alpha;
    
    if L_Phi > L_Phi_max
        alpha_ub = 0.5*(alpha_ub + alpha_lb);
    elseif L_Phi <= L_Phi_max && L_Phi ~= 0
        alpha_lb = 0.5*(alpha_ub + alpha_lb);
    else
        error('error')
    end
    
    alpha = 0.5*(alpha_ub + alpha_lb);
    L_Phi = FcnL_phi(AK,K,P,alpha);
    
   
    if abs(alpha - alpha_old)/abs(alpha_old) <= 10^-12 && L_Phi <= L_Phi_max && L_Phi ~= 0
        exitflag = 0;
    end
    nn = nn + 1;
    
end
clear('alpha_old','alpha_lb','alpha_ub','nn')

alpha

end



function [c, ceq] = nonlinConsAlpha(x, P, alpha)

    c = x'*P*x - alpha;
    ceq = [];

end

function xdot = system(t, x, u)

    xdot = zeros(2,1);
    
    mu = 0.5;
    xdot(1) = x(2) + u*(mu + (1-mu)*x(1)); % from condition
    xdot(2) = x(1) + u*(mu - 4*(1-mu)*x(2));
    
end

function phi = FcnPhi(x,AK,K)

    f = system(0,x,K*x);
    phi = f - AK*x;

end

function L_Phi = FcnL_phi(AK,K,P,alpha)

    opt = optimset('MaxFunEvals',10000,'MaxIter',10000,'Display','off');

    [x1,L_Phi_tilde] = fmincon(@(x) -sqrt(FcnPhi(x,AK,K)' * FcnPhi(x,AK,K))/sqrt(x'*x) ,...
        [10;10],[],[],[],[],[],[],@(x)nonlinConsAlpha(x,P,alpha),opt);
    
    L_Phi = -L_Phi_tilde;

end


