% create randn matrix for LASSO problem
clc;clear;
par.seed = 2012;
% addpath('../SparseSolvers/l1testpack');

% if isfield(par,'seed')
%     fprintf('Seed = %d\n',par.seed);
%     RandStream.setDefaultStream(RandStream('mt19937ar','seed',par.seed));
% end
% m = 16384; n =32768;
% k = 400;
% A = randn(m, n);
% xs = zeros(n,1);
% xs(randsample(n,k)) = randn(k,1);
% b = A*xs ;

%% generate random matrix data

lambda1 = 1e-1;
m = 256; n = 512;
options.dim = [m n];
options.sparsity = 20;
options.seed = 1;
disp('A: 1024 x 2048  Gauss; x: Gauss; s: 32; lambda: 1e-1')
[A,b,xdagger,tau,sigma,~] = construct_bpdn_instance('gauss',...
    'gauss',lambda1,options);

xs = xdagger;
mkdir Gaussian;
idx = [1 2 4 8 16];
for i = 1:5;
    nBlock = idx(i);
    mkdir('Gaussian',num2str(nBlock));
    for i=1:nBlock
        str1 = strcat(['Gaussian/',num2str(nBlock),'/A', num2str(i), '.dat']);
        str2 = strcat(['Gaussian/',num2str(nBlock),'/xs', num2str(i), '.dat']);
        mmwrite(str1,A( :,(i-1)*n/nBlock+1: i*n/nBlock));
        mmwrite(str2,xs((i-1)*n/nBlock+1: i*n/nBlock,:));
    end
    str3 = strcat(['Gaussian/',num2str(nBlock),'/b.dat']);
    mmwrite(str3, b);
end

% %% generate partial DCT data
% m = 1024;
% n = 2048;
% options.dim = [m, n];
% options.sparsity = 32;
% options.seed = 1;
% disp('A: 1024 x 2048  Gauss; x: Gauss; s: 32;');
% [A, b, xs] = construct_bpdn_instance('partdct','gauss',options);
% mkdir partdct;
% idx = [1 2 4 8 16 32 64];
% for i = 1:7;
%     nBlock = idx(i);
%     mkdir('partdct',num2str(nBlock));
%     for i=1:nBlock
%         str1 = strcat(['partdct/',num2str(nBlock),'/A', num2str(i), '.dat']);
%         str2 = strcat(['partdct/',num2str(nBlock),'/xs', num2str(i), '.dat']);
%         mmwrite(str1,A( :,(i-1)*n/nBlock+1: i*n/nBlock));
%         mmwrite(str2,xs((i-1)*n/nBlock+1: i*n/nBlock,:));
%     end
%     str3 = strcat(['partdct/',num2str(nBlock),'/b.dat']);
%     mmwrite(str3, b);
% end
%
% %% generate bernoulli data
% m = 1024;
% n = 2048;
% options.dim = [m, n];
% options.sparsity = 32;
% options.seed = 1;
% disp('A: 1024 x 2048  Gauss; x: Gauss; s: 32;');
% [A, b, xs] = construct_bpdn_instance('bernoulli','gauss',options);
% mkdir bernoulli;
% idx = [1 2 4 8 16 32 64];
% for i = 1:7;
%     nBlock = idx(i);
%     mkdir('bernoulli',num2str(nBlock));
%     for i=1:nBlock
%         str1 = strcat(['bernoulli/',num2str(nBlock),'/A', num2str(i), '.dat']);
%         str2 = strcat(['bernoulli/',num2str(nBlock),'/xs', num2str(i), '.dat']);
%         mmwrite(str1,A( :,(i-1)*n/nBlock+1: i*n/nBlock));
%         mmwrite(str2,xs((i-1)*n/nBlock+1: i*n/nBlock,:));
%     end
%     str3 = strcat(['bernoulli/',num2str(nBlock),'/b.dat']);
%     mmwrite(str3, b);
% end
