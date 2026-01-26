function Exercise3

close all
clear all
clc

A = [1 1; 0 1];
B = [0.5; 1];

n = length(A(1,:)); % 2
m = length(B(1,:)); % 1

% X = {[0 1]*x <= 2}
X = Polyhedron('A',[0 1],'b',2);

% U = {|u| <= 1}
U = Polyhedron('A',[1; -1],'b',[1; 1]);

% W = {w: ||w||_inf <= 0.1}
W = Polyhedron('A',[1 0; 0 1; -1 0; 0 -1],'b',[0.1; 0.1; 0.1; 0.1]);
Hw = W.A;
Kw = W.b;

Q = diag([1, 1]);
R = 0.01;

[P,~,~] = dare(A,B,Q,R);
K = -1*inv(R + B.'*P*B)*B.'*P*A;

A_K = A+B*K;

x0 = [-5; -2];

MPCIterations = 20;

N = 9;

epsilon = 1e-3;
[S_alpha,~,~] = InvariantApprox_mRPIset(A_K,W,epsilon);
S = S_alpha;
S.minHRep();

Xbar = X - S;
Xbar.minHRep();
KS = S.affineMap(K);
Ubar = U - KS;
Ubar.minHRep();

% Y = {y \in R^n | H*y <= K}
H_x = Xbar.A;
k_x = Xbar.b;
H_u = Ubar.A;
k_u = Ubar.b;

HxXf = [H_x; H_u*K]; % y - 2*1
KxXf = [k_x; k_u];

[Zf, HXf, KXf] = maxInvSet(A_K, HxXf, KxXf);

%% Optimization task
% ==============================================
% v,z
%
% quadprog

tol_opt = 1e-8;
options = optimset('Display','off', ...
    'TolFun', tol_opt, ...
    'MaxIter', 10000, ...
    'Algorithm', 'interior-point-convex', ...
    'TolConSQP', 1e-6);

% ineq
Ax = [];
bx = [];
Au = [];
bu = [];

% ineq for U
A_u = H_u;
b_u = k_u;

% ineq for X
A_x = H_x;
b_x = k_x;

for k = 1:N
    Ax = blkdiag(Ax, A_x);
    bx = [bx; b_x];
end
for k = 1:N
    Au = blkdiag(Au, A_u);
    bu = [bu; b_u];
end

% terminal ineq
Ax=blkdiag(Ax,HXf);
bx=[bx;KXf];

A_ineq = blkdiag(Ax, Au);
b_ineq = [bx; bu];

% eq
Aeq = zeros(n*(N), n*(N+1)+m*N);
beq = zeros(n*(N),1);
for k=0:N-1
    Aeq(n*k+1:n*(k+1), 1:n*(N+1)) = [zeros(n,n*k),A,-eye(n),zeros(n,n*(N-1-k))];
    Aeq(n*k+1:n*(k+1), n*(N+1)+1:end) = [zeros(n,m*k),B,zeros(n,m*(N-1-k))];
end

Qstack = [];
Rstack = [];
for k=1:N
    Qstack = blkdiag(Qstack, 0.5*Q);
    Rstack = blkdiag(Rstack, 0.5*R);
end
Qstack = blkdiag(Qstack, 0.5*P);

H = blkdiag(Qstack, Rstack);
f = zeros(size(H,1),1); 

% memory
x = zeros(n,MPCIterations);
u = zeros(m,MPCIterations);
v_MPC = zeros(m,MPCIterations);
z_MPC = zeros(n,MPCIterations);

x(:,1) = x0;

u0 = zeros(m*N,1);
x0 = repmat(x0,N+1,1);

for ii=1:MPCIterations-1
    
    x_measure = x(:,ii);
    Hs = S.A;
    ks = S.b;
    A_init = zeros(size(Hs,1), n*(N+1)+m*N);
    A_init(:, 1:n) = -Hs;
    b_init = ks - Hs * x_measure;
    A_ineq_iter = [A_ineq; A_init];
    b_ineq_iter = [b_ineq; b_init];
    % H = (H + H')/2;
    
    solutionOL = quadprog(H,f,A_ineq_iter,b_ineq_iter,Aeq,beq,[],[],[x0; u0],options);
    
    x_OL = solutionOL(1:n*(N+1),1);
    u_OL = solutionOL(n*(N+1):end,1);
    
    v_MPC(:,ii) = u_OL(1);   
    z_MPC(:,ii) = x_OL(1:n);   
    
    u(:,ii) = v_MPC(:,ii) + K*( x(:,ii) - z_MPC(:,ii) );
   
    w_min = -0.1;
    w_max = 0.1;
    w(:,ii) = w_min + (w_max - w_min) .* rand(n,1);
    
    % update x
    x(:,ii+1) = A*x(:,ii) + B*u(:,ii) + w(:,ii);
    
    x0 = x(:,ii+1);
    u0 = zeros(m*N,1);
end

%% Graphics
figure(1), hold on,
plot(Zf+S,'color','magenta'), plot(Zf)
for ii=1:(MPCIterations-1)
    plot(S+z_MPC(:,ii),'color','purple')
end
clear('ii')
plot(x(1,:),x(2,:))
plot(z_MPC(1,:),z_MPC(2,:),'g-.')
plot(0,0,'.','color','black')
title('state space')
xlabel('x_1')
ylabel('x_2')

figure(2), stairs(u), hold on, stairs(v_MPC,'red')
title('input')
xlabel('iteration')
ylabel('u')

end

function [F_alpha_s, alpha, s] = InvariantApprox_mRPIset(A, W, epsilon)

% system dimension
n = W.Dim;

% initialization
alpha = 0;
logicalVar = 1;

s = 0;

while logicalVar == 1;
    
    s = s + 1;
    
    % alpha_0(s)
    
    % inequality representation of the set W: f_i*w <= g_i , i=1,...,I_max
    f_i = (W.A)';
    g_i = W.b;
    I_max = length(W.b);
    
    % call of the support function h_W
    h_W = zeros(I_max,1);
    for k = 1:I_max
        
        a = (A^s)' * f_i(:,k);
        
        h_W(k) = fkt_h_W(a, W);
        
    end
    clear('k')
    
    % output
    alpha_opt_s = max( h_W ./ g_i );  
    alpha = alpha_opt_s;
    
    %  M(s)
    ej = eye(n);
    sum_vec_A = zeros(n,1);
    sum_vec_B = zeros(n,1);
    updt_A = zeros(n,1);
    updt_B = zeros(n,1);
    
    for k = 1:s
        for j = 1:n
            a = (A^(k-1))' * ej(:,j);
            updt_A(j) = fkt_h_W(a, W);
            updt_B(j) = fkt_h_W(-a, W);
        end
        sum_vec_A = sum_vec_A + updt_A;
        sum_vec_B = sum_vec_B + updt_B;
    end
    clear('k')
    
    Ms = max(max(sum_vec_A, sum_vec_B));
    
    % Interrupt criterion
    if alpha <= epsilon/(epsilon + Ms)
        logicalVar = 0;
    end
    
end

% Fs
Fs = Polyhedron('A', [], 'b', [], 'Ae', eye(n), 'be', zeros(n,1));

for k = 1:s
    
    Fs = Fs + (A^(k-1)) * W;
    
end

% F_Inf approx
F_alpha_s = 1/(1 - alpha) * Fs;



% support function h_W
function [h_W, diagnostics] = fkt_h_W(a, W)

% dimension of w
nn = W.Dim;

% optimization variable
w = sdpvar(nn,1);

% cost function
Objective = -a' * w;

% constraints
Constraints = [ W.A * w <= W.b ];

% optimization
Options = sdpsettings('solver','quadprog','verbose',0);
diagnostics = optimize(Constraints,Objective,Options);

% output
w_opt = value(w);
h_W = a' * w_opt;

end


end


function [O_Inf,G,g]=maxInvSet(A,H,h)

options = optimset('Display','off');

m=length(h(:,1));


notFinished=1;
fmax=-inf;


h_new=h;
H_new=H;

while(notFinished)
      
 for i=1:m 
    [~,fval]=linprog(-H_new(end-m+i,:)*A,H_new,h_new,[],[],[],[],options);
    fmax=max(-fval-h(i),fmax);
 end
 
 if(fmax<=0)
     notFinished=0;
 else
     fmax=-inf;
    
     H_new=[H_new; H_new(end-m+1:end,:)*A];
     h_new=[h_new;h_new(end-m+1:end)];   
 end
end

G=H_new;
g=h_new;
O_Inf = Polyhedron(G,g);
O_Inf.minHRep();
O_Inf.minVRep();

end
