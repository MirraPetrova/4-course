function MPC_Exercise2

clear all
close all
clc

tol_opt       = 1e-8;
options = optimset('Display','off',...
                'TolFun', tol_opt,...
                'MaxIter', 10000,...
                'Algorithm', 'active-set',...
                'FinDiffType', 'forward',...
                'RelLineSrchBnd', [],...
                'RelLineSrchBndDuration', 1,...
                'TolConSQP', 1e-6);

n = 2; % from condition  - x
m = 1; % for u

x_eq = zeros(n,1);
u_eq = zeros(m,1);

mpciterations = 15;

T = 1.5; % from condition

delta = 0.1; % from condition

N = T/delta; % =15

t_0 = 0.0;
x_init = [-0.7; -0.8]; % from condition

Q = [0.5 0;0 0.5]; % from condition
R = 1.0;

K_loc = [-2.1180 -2.1180]; % from computeAlpha_new
P = [16.5926 11.5926;11.5926 16.5926] % from computeAlpha_new
alpha = 0.7; % from computeAlpha_new

u0 = 0.5*ones(m,N);

% Calculation of initial predictions of the state sequence
x0 = zeros(n*(N+1),1); % first - x_init, next - N counts
x0(1:n)=x_init;
for k = 1:N
    u_k = u0(:,k);                          
    x_k = x0((k-1)*n+1:k*n); % 1,2 - 3,4 - ...
    x0(k*n+1:(k+1)*n) = dynamic(delta, x_k, u_k); % changing 3,4 - 5,6 - ... by discretization
end


% Linear constraints
% Inequalities:
A=[];
b=[];
% Initial state constraint
Aeq_x = zeros(n, n*(N+1));
Aeq_x(:,1:n) = eye(n);
Aeq = [Aeq_x, zeros(n, m*N)];
beq = x_init;
% Control constraints
lb = [-Inf*ones(n*(N+1),1); -2*ones(m*N,1)];
ub = [ Inf*ones(n*(N+1),1);  2*ones(m*N,1)];

t = [];
x = [];
u = [];

E = ellipsoid(x_eq, alpha*inv(P));

f1 = figure(1); hold on
set(f1,'PaperPositionMode','auto')
set(f1,'Units','pixels')
% set(f1,'Position', [0, 0,640-1, 480]); 
plot(E,'r'), axis equal, grid on


fprintf('   k  |      u(k)        x(1)        x(2)     Time \n');
fprintf('---------------------------------------------------\n');

tmeasure = t_0;
xmeasure = x_init;

for ii = 1:mpciterations 

   % Initial prediction and initial constraints
    beq=xmeasure;
    y_init=[x0;u0(:)];

    t_Start = tic;
    
    % structure: y_OL=[x_OL,u_OL];
    [y_OL, V, exitflag, output]=fmincon(@(y) costfunction( N, y, x_eq, u_eq, Q, R, P,n,m,delta),...
        y_init,A,b,Aeq,beq,lb,ub,...
        @(y) nonlinearconstraints(N, delta, y, x_eq, P, alpha,n,m), options);

    t_Elapsed = toc( t_Start );
    
    x_OL=y_OL(1:n*(N+1));
    u_OL=y_OL(n*(N+1)+1:end);
    
    % Feedback 
    t = [t, tmeasure];
    x = [x, xmeasure]; % like previous lab
    u = [u, u_OL(1)]; 
    
    % Updating
    xmeasure = x_OL(n+1:2*n); % 3,4
    tmeasure = tmeasure + delta; % like previous lab
    
    u0 = repmat(K_loc*xmeasure, 1, N);
    x0 = [xmeasure; x0(1:end-n)];
    
    fprintf(' %3d  | %+11.6f %+11.6f %+11.6f  %+6.3f\n', ii, u(end),...
            x(1,end), x(2,end),t_Elapsed);
     
    f1 = figure(1);
    plot(x(1,:),x(2,:),'b'), grid on, hold on,
    plot(x_OL(1:n:n*(N+1)),x_OL(n:n:n*(N+1)),'g')
    plot(x(1,:),x(2,:),'ob')
    xlabel('x(1)')
    ylabel('x(2)')
    drawnow
  
end

figure(2)
stairs(t,u);

end

function xdot = system(t, x, u, delta)
    xdot = zeros(2,1);

    mu = 0.5;
    xdot(1) = x(2) + u(1)*(mu + (1-mu)*x(1));
    xdot(2) = x(1) + u(1)*(mu - 4*(1-mu)*x(2));
    
end

function cost = costfunction(N, y, x_eq, u_eq, Q, R, P,n,m,delta)
    
    cost = 0;
    x=y(1:n*(N+1));
    u=y(n*(N+1)+1:end);
   
    for k=1:N
        x_k=x(n*(k-1)+1:n*k);
        u_k=u(m*(k-1)+1:m*k);
        cost = cost + delta*runningcosts(x_k, u_k, x_eq, u_eq, Q, R);
    end
    cost = cost + terminalcosts( x(n*N+1:n*(N+1)), x_eq, P);
    
end


function cost = runningcosts(x, u, x_eq, u_eq, Q, R)
    cost = (x - x_eq)' * Q * (x - x_eq) + (u - u_eq)' * R * (u - u_eq);
    
end



  function [c, ceq] = nonlinearconstraints(N, delta, y, x_eq, P, alpha,n,m)     
   
   x=y(1:n*(N+1)); 
   u=y(n*(N+1)+1:end);
   c = [];
   ceq = [];

    for k=1:N
        x_k=x((k-1)*n+1:k*n);
        x_new=x(k*n+1:(k+1)*n);        
        u_k=u((k-1)*m+1:k*m);
        % Dynamic constraints
        ceqnew=x_new - dynamic(delta, x_k, u_k);
        ceq = [ceq ceqnew];
       
    end
   
   [cnew, ceqnew] = terminalconstraints( x(n*N+1:n*(N+1)), x_eq, P, alpha);
    c = [c cnew];
    ceq = [ceq ceqnew];
    
end

function cost = terminalcosts(x, x_eq, P)
    cost = (x - x_eq)' * P * (x - x_eq);
end



function [c, ceq] = terminalconstraints(x, x_eq, P, alpha)
    c   =  (x - x_eq)' * P * (x - x_eq) - alpha;
    ceq = [];
end




function [x] = dynamic(delta, x0, u)
    % Discretization by Runge-Kutta (ode45)
   
    atol_ode  = 1e-4;
    rtol_ode  = 1e-4;
    options = odeset('AbsTol', atol_ode, 'RelTol', rtol_ode);
    
    [t_intermediate, x_intermediate] = ode45(@system, [0,delta], x0, options, u);
        
    x = x_intermediate(end,:)';

end