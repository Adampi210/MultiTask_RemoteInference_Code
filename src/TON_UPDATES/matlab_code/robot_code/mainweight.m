close all;
clear all;
clc;

B=40; % AoI Bound
delta=1:B;

weight=1:1:10;

for w=weight

%% penalty function
p(1, 1, 1:B)=readtable("detection_results_robot_9_error_function.csv").Error(2:end);
p(1, 2, 1:B)=w*readtable("detection_results_robot_1_error_function.csv").Error(2:end);
%p(1, 3, 1:B)=readtable("detection_results_robot_9_error_function.csv").Error(2:end);
%p(1, 4, 1:B)=readtable("detection_results_robot_9_error_function.csv").Error(2:end);
p(2, 1, 1:B)=readtable("detection_results_robot_8_error_function.csv").Error(2:end);
%p(2, 2, 1:B)=readtable("detection_results_robot_9_error_function.csv").Error(2:end);
p(3, 1, 1:B)=readtable("detection_results_robot_2_error_function.csv").Error(2:end);
% p(3, 2, 1:B)=readtable("detection_results_robot_8_error_function.csv").Error(2:end);
p(4, 1, 1:B)=readtable("detection_results_robot_4_error_function.csv").Error(2:end);
%p(4, 2, 1:B)=readtable("detection_results_robot_9_error_function.csv").Error(2:end);

% p(1, 1, 1:B)=1:B;
% p(1, 2, 1:B)=0.3*(1:B);
% p(2, 1, 1:B)=exp(0.5*(1:B));
% p(2, 2, 1:B)=0.1*log10(1:B);
%% number of sources, channels, bound, and time, one task per time
km=[2, 1, 1, 1];
M=4;
N=2;
T=100;
gamma=0.1;
%w=ones(M,km);
%w(1,:)=0.3;

%pmgf=MGF(M,N,km,T,B,w,gamma,p);
pmgfreoptimized(w)=MGFReoptimized(M,N,km,T,B,gamma,p);
pmaf(w)=MAF(M,N,km,T,B,gamma,p);
prand(w)=randpolicy(M,N,km,T,B,gamma,p);
end
plot(weight, prand, 'ro-', weight, pmaf, 'bx-', weight, pmgfreoptimized, 'kd-');
legend('Random Policy', 'MAF Policy', 'Reoptimized MGF Policy')
xlabel('weight $w_{1,2}$'); ylabel('Dis. Sum of Inference Errors')


