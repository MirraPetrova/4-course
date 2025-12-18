function MPC_Exercise1()

% ������� ������� �������, �������� �����, ������� ���������� ����
clear all
close all
clc

% ��������� ����������
tol_opt       = 1e-8;
options = optimset('Display','off',...
    'TolFun', tol_opt,...
    'MaxIter', 10000,...
    'Algorithm', 'active-set',...
    'TolConSQP', 1e-6);

warning off

% ��������� ������
mpciterations = 50;

% ����� ������������ (����������� �������� ������������)
T = 4; % ��������, 4 �������

% ��� �������������
delta = 0.1; % ��������, 0.1 �������

% �������� ������������
N = T/delta; % ����� ����� ������������

% �������� ������� (�����������)
Ac = [0 1; -2 -3];
Bc = [0; 1];

n = length(Ac(1,:)); % ����������� ���������
m = length(Bc(1,:)); % ����������� ����������

sysc = ss(Ac,Bc,eye(n),zeros(n,m));

sysd = c2d(sysc, delta, 'zoh');
Ad = sysd.A;
Bd = sysd.B;
% �������� ��������� (��������, ��������)
xTerm = [0; 0];

% ��������� ���������
Q = [1 0; 0 1];  % ���� �� ���������
R = 1;           % ��� �� ����������

% ��������� �������
tmeasure = 0.0; % ��������� �����
xmeasure = [0.6; 0.8]; % ��������� ���������

% ��������� ������������ ����������
u0 = zeros(N,1); % ��������� ���������� (��������, �������)
% ��������� ������������ ��������� (�����������, ��� ��������� ���������� �������)
x0 = repmat(xmeasure, 1, N+1); % ������� ��������� (������ n x N+1)

% ����������� �� ��������� � ����������
H_x = [1; -1; 0; 0; 0; 0]; % ������, ����� ����� ������ ���� �����������
k_x = [1; 1; 0; 0; 0; 0]; % ��������, x(1) � x(2) ���������� [-1,1]

H_u = [1; -1]; % ���������� ���������� [-1, 1]
k_u = [1; 1];

% ���������� ������ ��� ������������ �����
% ���������� ������� ������� ��� ���� ���������������� ������

% ������� ��� ����������
U = zeros(N,m);
X = zeros(n,N+1);

% ������������ ������ ������������
% ������ ����� ��� ������������
A_block = zeros(n*(N+1), n);
B_block = zeros(n*(N+1), m*N);

A_block(1:n, :) = eye(n);
for i=1:N
    A_block((i)*n+1:(i+1)*n, :) = Ad^i;
    for j=1:i
        B_block(i*n+1:(i+1)*n, (j-1)*m+1:j*m) = Ad^(i-j)*Bd;
    end
end

% ������������ �����
H = 2*(B_block'*kron(eye(N),Q)*B_block + kron(eye(N),R));
f = zeros(N,1); % �������� ����� (���� ����, ����� ��������)

% �����������
% ����������
A_u = [eye(N*m); -eye(N*m)];
b_u = [k_u*ones(N,1); k_u*ones(N,1)];

% ����������� �� ��������� (��������, x(1) � x(2) � [-1,1])
% � ���� H_x * x <= k_x
% �� � ��� ���� ������������ ���������, ��������� � ����������� ����� A � B
% ������� ����������� �� ��������� ����� �������� ����� ����������

% ��� ����� �������� ������� ��� ���������
A_state = A_block;
b_state_upper = repmat([1; 1], N+1, 1); % ������� �������
b_state_lower = -b_state_upper; % ������ �������

% ������������ ����
% ������� ��� �������� �����������
t = [];
x = [];
u = [];

fprintf('   k  |      u(k)        x(1)        x(2)     Time \n');
fprintf('---------------------------------------------------\n');

for ii=1:mpciterations
    
    % ��������� ����������� �� ��������� (���� ���� ����������� �� x)
    % � ������ ������, ��������, ��������� ��������� [-1,1]
    % ��������� ����������� ����������� �� ����������
    % ��������������, ��� x = A_state * x0 + B * u, ��� x0 - ��������� ���������
    
    % ��������� ������� ��� ������������ ���������
    % (����� ��� ���� A_block, B_block)
    % ����������� ��� ���������
    A_x_total = A_state;
    b_x_upper_total = b_state_upper;
    b_x_lower_total = -b_state_lower;
    
    % ������� ������ ������������� ����������������
    tic;
    % ������ ������� quadprog
    % ������� �������: 0.5*u'*H*u + f'*u
    % �������� ���������
    solutionOL = quadprog(H, f, A_u, b_u, A_x_total, b_x_upper_total, [], [], u0, options);
    t_Elapsed = toc;
    
    % ��������� ����������� ���������� � ���������
    u_OL = reshape(solutionOL, N, m);
    x_OL = A_block * xmeasure + B_block * solutionOL;
    
    % ���������� ��������� �������
    xmeasure = x_OL(:,2); % ��������� ����� ������� ����
    tmeasure = tmeasure + delta;
    
    % ��������� ����������
    t = [t, tmeasure];
    x = [x, xmeasure];
    u = [u, u_OL(1)];
    
    % ���������� ������������ ��� ���������� ����
    x0 = x_OL(:,2:end); % ����� � ���������� ���������� ���������
    u0 = [u_OL(2:end,:); u_OL(end,:)]; % ����� ����������
    
    % �����
    fprintf(' %3d  | %+11.6f %+11.6f %+11.6f  %+6.3f\n', ii, u(end), x(1,end), x(2,end), t_Elapsed);
    
    % ���������� ��������
    figure(1);
    plot(x(1,:),x(2,:),'b'); grid on; hold on;
    plot(x_OL(1,:),x_OL(2,:),'g');
    plot(x(1,end),x(2,end),'ob');
    xlabel('x(1)');
    ylabel('x(2)');
    title('state space');
    drawnow;
    
    figure(2);
    stairs(t(end)+delta*(0:N-1), u_OL);
    grid on; hold on;
    plot(t(end), u(end),'bo');
    xlabel('prediction time');
    ylabel('uOL');
    drawnow;
    
end

% �������� �������
figure(3);
stairs(t, u);
xlabel('t');
ylabel('u');
title('closed-loop input');

end


Aeq_block = zeros(n*N, m*N + n*N);
beq_block = zeros(n*N,1);

for k = 1:N
    row_idx = (k-1)*n + (1:n);
 
    col_u_idx = ( (k-1)*m + 1 : (k)*m );
    Aeq_block(row_idx, col_u_idx) = -Bd;
    
    col_x_idx = m*N + ( (k-1)*n + 1 : k*n );
    Aeq_block(row_idx, col_x_idx) = eye(n);
    
    if k > 1
        col_x_prev_idx = m*N + ( (k-2)*n + 1 : (k-1)*n );
        Aeq_block(row_idx, col_x_prev_idx) = -Ad;
    else
        beq_block(row_idx) = Ad * xmeasure;
    end
end

Aeq_t = zeros(n, m*N + n*N);
Aeq_t(:, m*N + ( (N-1)*n + 1 : N*n )) = eye(n);
beq_t = xTerm;

A_eq = [Aeq_block; Aeq_t];
b_eq = [beq_block; beq_t];