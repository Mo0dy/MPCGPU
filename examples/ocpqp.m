% this file generates a random set of {A, B, b, Q, R, q, r} & x0 for LQR
% solves the LQR in two ways 
% 1. Riccati
% 2. Schur complement with PCG
% the kkt matrix {G_dense, C_dense, g, c} are saved to the data/ directory
% as txt files 

% the qp_solve.cu will read in txt files and perform OCP QP solving with
% Schur complement + PCG on GPU

close all
clear
digits(16)
N = 10;
nx = 20;
nu = 1;

[A, B, b, Q, R, q, r, x0] = generateRandomLQR(nx, nu, N);
[x, u] = riccatiSolve(A, B, b, Q, R, q, r, N, x0, nx, nu);
[G_dense, C_dense, g, c, xu_schur, x_schur, u_schur] = schurSolve(A, B, b, Q, R, q, r, N, x0, nx, nu);
writeKKTMatrixToFile(G_dense, C_dense, g, c)

function writeKKTMatrixToFile(G_dense, C_dense, g, c)
    if ~exist('data', 'dir')
       mkdir('data')
    end
    writematrix(G_dense, './data/G_dense.txt')
    writematrix(C_dense, './data/C_dense.txt')
    writematrix(g, './data/g.txt')
    writematrix(-c, './data/c.txt')
end

function [A, B, b, Q, R, q, r, x0] = generateRandomLQR(nx, nu, N)
    % generate a set of random {A_k, B_k}
    A = cell(1, N-1);
    B = cell(1, N-1);
    b = cell(1, N-1);
    for i=1:N-1
        A{i} = rand(nx, nx);
        B{i} = rand(nx, nu);
        b{i} = rand(nx, 1);
    end

    % generate a set of random {Q_k, R_k, q_k, r_k}
    Q = cell(1, N);
    q = cell(1, N);
    R = cell(1, N-1);
    r = cell(1, N-1);
    for i=1:N-1
        Q{i} = diag(rand(nx,1));
        q{i} = rand(nx, 1);
        R{i} = diag(rand(nu,1));
        r{i} = rand(nu, 1);
    end
    Q{N} = diag(rand(nx,1));
    q{N} = rand(nx, 1);

    x0 = rand(nx, 1);
end

function cost = LQRcost(x, u, Q, R, q, r, N)
    cost = 0;
    for i=1:N-1
        cost = cost + 0.5*x(:, i)'*Q{i}*x(:, i);
        cost = cost + 0.5*u(:, i)'*R{i}*u(:, i);

        cost = cost + x(:, i)'*q{i};
        cost = cost + u(:, i)'*r{i};
    end
    cost = cost + 0.5*x(:, N)'*Q{N}*x(:, N);
    cost = cost + x(:, N)'*q{N};
end

function [x, u] = riccatiSolve(A, B, b, Q, R, q, r, N, x0, nx, nu)
    P = Q{N};
    p = q{N};
    K = cell(1, N-1);
    k = cell(1, N-1);
    for i=N-1:-1:1
        % pay attention to the update order of P & p
        Re = R{i} + B{i}'*P*B{i};
        K{i} = -inv(Re)*B{i}'*P*A{i};
        k{i} = -inv(Re)*(r{i} + B{i}'*(P*b{i} + p));
        p = q{i} + A{i}'*(P*b{i} + p) - K{i}'*Re*k{i};
        P = Q{i} + A{i}'*P*A{i} - K{i}'*Re*K{i};
    end

    x = zeros(nx, N);
    x(:,1)= x0;
    u = zeros(nu, N-1);
    for i=1:N-1
        u(:,i) = K{i}*x(:,i) + k{i};
        x(:,i+1) = A{i}*x(:,i) + B{i}*u(:,i) + b{i};
    end
end

function [G_dense, C_dense, g, c, xu_gt, x, u] = schurSolve(A, B, b, Q, R, q, r, N, x0, nx, nu)
    x = zeros(nx, N);
    u = zeros(nu, N-1);

    [D, O, S] = formKKTSchur(A, B, Q, R, N);
    P = formPreconditionerSS(D, O, N, nx);
%     disp('S original')
%     disp(S)
    % plotSpectrum(P, S)

    [D_t_ldl, O_t_ldl, S_t_ldl, T_ldl] = preprocessSLDL(D, O, N, nx);
    [D_t_chol, O_t_chol, S_t_chol, T_chol] = preprocessSChol(D, O, N, nx);
%     disp('S_t_ldl')
%     disp(S_t_ldl)
%     for i=1:N
%        disp(T_ldl{i})
%     end
    P_t_ldl = formPreconditionerSS(D_t_ldl, O_t_ldl, N, nx);
    P_t_chol = formPreconditionerSS(D_t_chol, O_t_chol, N, nx);
%     disp('P_t_ldl')
%     disp(P_t_ldl)
    % plotSpectrum(P_t, S_t)
    
    N = N-1;
    n = nx + nu;
    G = zeros(N*n+nx);
    g = zeros(N*n+nx, 1);
    C = zeros(nx*(N+1), N*n+nx);
    c = zeros(nx*(N+1),1);

    nG = nx^2 + nu^2;
    G_dense = zeros(N*nG+nx^2, 1);
    nC = nx^2 + nx*nu;
    C_dense = zeros(N*nC, 1);
    
    for i=0:N-1
        G(1+i*n:(i+1)*n, 1+i*n:(i+1)*n) = blkdiag(Q{i+1}, R{i+1});
        g(1+i*n:(i+1)*n, 1) = [q{i+1};r{i+1}];

        C((i+1)*nx+1:(i+2)*nx, i*n+1:(i+1)*n+nx) = [-A{i+1}, -B{i+1}, eye(nx)];
        c((i+1)*nx+1:(i+2)*nx, 1) = b{i+1};

        G_dense(1+i*nG:(i+1)*nG, 1) = [Q{i+1}(:); R{i+1}(:)];
        C_dense(1+i*nC:(i+1)*nC, 1) = [-A{i+1}(:); -B{i+1}(:)];
    end
    G(N*n+1:N*n+nx, N*n+1:N*n+nx) = Q{end};
    g(N*n+1:N*n+nx, 1) = q{end};
    C(1:nx, 1:nx) = eye(nx);
    c(1:nx) = x0;

    G_dense(N*nG+1:N*nG+nx^2, 1) = Q{end}(:);

    gamma = c + C*inv(G)*g;
%     disp('gamma original ')
%     disp(gamma)
    N = N + 1;
    gamma_t_ldl = zeros(N*nx, 1);
    for i=1:N
        gamma_t_ldl(1+(i-1)*nx:i*nx,1) = T_ldl{i}*gamma(1+(i-1)*nx:i*nx,1);
    end
   gamma_t_chol = zeros(N*nx, 1);
    for i=1:N
        gamma_t_chol(1+(i-1)*nx:i*nx,1) = T_chol{i}*gamma(1+(i-1)*nx:i*nx,1);
    end
%     disp(gamma_t_ldl)

    lam_gt = S \ gamma;
    max_iter_lin = 1e4;
    [lam_pcg, ~, ~, iter_pcg] = pcg(S, gamma, 1e-8, max_iter_lin, inv(P));
    [lam_org, I]= PCG(P, S, gamma, zeros(N*nx,1), 1e-8, max_iter_lin);
    [lam_t_ldl, I_t_ldl]= PCG(P_t_ldl, S_t_ldl, gamma_t_ldl, zeros(N*nx,1), 1e-8, max_iter_lin);
    [lam_t_chol, I_t_chol]= PCG(P_t_chol, S_t_chol, gamma_t_chol, zeros(N*nx,1), 1e-8, max_iter_lin);

    l_trans_ldl = zeros(N*nx, 1);
    for i=1:N
        l_trans_ldl(1+(i-1)*nx:i*nx,1) = T_ldl{i}'*lam_t_ldl(1+(i-1)*nx:i*nx,1);
    end
    l_trans_chol = zeros(N*nx, 1);
    for i=1:N
        l_trans_chol(1+(i-1)*nx:i*nx,1) = T_chol{i}'*lam_t_chol(1+(i-1)*nx:i*nx,1);
    end
    
    disp(['relres of lambda ground truth = ', num2str(norm(S*lam_gt - gamma)/norm(gamma))])
    disp(['relres of lambda matlab pcg = ', num2str(norm(S*lam_pcg - gamma)/norm(gamma))])
    disp(['relres of lambda original pcg = ', num2str(norm(S*lam_org - gamma)/norm(gamma))])
    disp(['relres of lambda transformed ldl pcg = ', num2str(norm(S*l_trans_ldl - gamma)/norm(gamma))])
    disp(['relres of lambda transformed chol pcg = ', num2str(norm(S*l_trans_chol - gamma)/norm(gamma))])

    
    disp(['Matlab PCG condition number= ' num2str(cond(P*S)) ', iteration= ' num2str(iter_pcg)])
    disp(['Original PCG condition number= ' num2str(cond(P*S)) ', iteration= ' num2str(I)])
    disp(['Transformed ldl PCG condition number= ' num2str(cond(P_t_ldl*S_t_ldl)) ', iteration= ' num2str(I_t_ldl)])
    disp(['Transformed chol PCG condition number= ' num2str(cond(P_t_chol*S_t_chol)) ', iteration= ' num2str(I_t_chol)])
    
    disp(['Unique eigenvalues of P*S count= ' num2str(sortUniqueEigen(P*S))])

    xu_gt = inv(G)*(C'*lam_gt - g);
    disp(['norm of xu_gt: ' num2str(norm(xu_gt))])
    xu_org = inv(G)*(C'*lam_org - g);
    disp(['norm of xu_org: ' num2str(norm(xu_org))])
    xu_t_ldl = inv(G)*(C'*l_trans_ldl - g);
    disp(['norm of xu_t_ldl: ' num2str(norm(xu_t_ldl))]) 
    xu_t_chol = inv(G)*(C'*l_trans_chol - g);
    disp(['norm of xu_t_chol: ' num2str(norm(xu_t_chol))]) 
    
    N = N-1;

    for i=0:N-1
        x(:,i+1) = xu_gt(1+i*n:i*n+nx);
        u(:,i+1) = xu_gt(i*n+nx+1:(i+1)*n);
    end
    x(:,N+1) = xu_gt(N*n+1:end);
    
end

function l = sortUniqueEigen(A)
    eigens = sort(eig(A));
    eigens = real(eigens);
    l = length(uniquetol(eigens, 1e-6));
end

function plotSpectrum(P, S)
    nS = size(S, 1);
    result = P * S;
    eigens = sort(eig(result));
    eigens = real(eigens);
    figure
    subplot(2,1,1)
    scatter(eigens, zeros(nS, 1), 'r', 'filled')
    subplot(2,1,2)
    plot(eigens, 'r')
end

function [D, O, S] = formKKTSchur(A, B, Q, R, N)
    D = cell(1, N);
    O = cell(1, N-1);

    nx = size(A{1}, 1);
    D{1} = inv(Q{1});
    for i=1:N-1
        D{i+1} = A{i}*inv(Q{i})*A{i}' + B{i}*inv(R{i})*B{i}' + inv(Q{i+1}); % theta{i}
        O{i} = -A{i}*inv(Q{i}); % phi{i}
        O{i} = O{i}';
    end

    S = composeBlockDiagonalMatrix(D, O, N, nx);
end

function P = formPreconditionerSS(D, O, N, nx)
    D_p = cell(1, N);
    O_p = cell(1, N-1); 

    for i=1:N-1
        D_p{i} = inv(D{i});
        O_p{i} = -inv(D{i})*O{i}*inv(D{i+1});
    end
    D_p{N} = inv(D{N});

    P = composeBlockDiagonalMatrix(D_p, O_p, N, nx);
end

function [D_pre, O_pre, S_pre, T] = preprocessSChol(D, O, N, nx)
    D_pre = cell(1,N);
    O_pre = cell(1,N-1);
    T = cell(1, N);

    L1 = chol(D{1}, 'lower');
    T{1} = inv(L1);
    D_pre{1} = eye(nx);

    for i=1:N-1
        Li1 = chol(D{i+1}, 'lower');
        T{i+1} = inv(Li1);
        D_pre{i+1} = eye(nx);
        O_pre{i} = T{i}*O{i}*T{i+1}';
    end

    S_pre = composeBlockDiagonalMatrix(D_pre, O_pre, N, nx);
end

function [D_pre, O_pre, S_pre, T] = preprocessSLDL(D, O, N, nx)
    D_pre = cell(1,N);
    O_pre = cell(1,N-1);
    T = cell(1, N);

    [L1, D1] = ldl(D{1});
    T{1} = inv(L1);
    D_pre{1} = D1;

    for i=1:N-1
        [Li1, Di1] = ldl(D{i+1});
        T{i+1} = inv(Li1);
        D_pre{i+1} = Di1;
        O_pre{i} = T{i}*O{i}*T{i+1}';
    end

    S_pre = composeBlockDiagonalMatrix(D_pre, O_pre, N, nx);
end

function out = composeBlockDiagonalMatrix(D, O, N, nx)
    out = zeros(N*nx);
    out(1:nx, 1:2*nx) = [D{1}, O{1}];
    for i=2:N-1
        out((i-1)*nx+1:i*nx, (i-2)*nx+1:(i+1)*nx) = [O{i-1}', D{i}, O{i}];
    end
    out(end-nx+1:end, end-2*nx+1:end) = [O{N-1}', D{N}];
end

function [lambda, i] = PCG(Pinv, S, gamma, lambda_0, tol, max_iter)
    lambda = lambda_0;
    r = gamma - S*lambda;
    p = Pinv*r;
    r_tilde = p;
    nu = r'*r_tilde;
    for i=1:max_iter
       tmp = S*p;
       alpha = nu / (p'*tmp);
       r = r - alpha*tmp;
       lambda = lambda + alpha*p;
       r_tilde = Pinv*r;
       nu_prime = r'*r_tilde;
       % different exiting conditions
%        if abs(nu_prime) < tol
       if norm(r)/norm(gamma) < tol
          break
       end
       beta = nu_prime / nu;
       p = r_tilde + beta*p;
       nu = nu_prime;
    end
end