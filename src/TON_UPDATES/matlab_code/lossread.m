close all;
clear all;
clc;

filename = 'smooth_segmentation_averaged_multi_k_loss_pk_data.csv';
M = readmatrix(filename);
p1=100*M(2:end,2);
AoI=M(2:end, 1);
variance1=100*M(2:end,3);
figure(1)
errorbar(AoI, p1, variance1, 'horizontal')
ylabel('Inference Error 100(1-IoU)'), xlabel('AoI')
filename='smooth_averaged_detection_test_loss_pk_data.csv';
M = readmatrix(filename);
p2=M(2:end,2);
AoI=M(2:end, 1);
variance2=M(2:end,3);
figure(2)
errorbar(AoI, p2, variance2)
ylabel('Inference Error (MSE)'), xlabel('AoI')
file='loss.mat';
save(file,"p1","p2");
figure(3)
yyaxis left
errorbar(AoI, p1, variance1)
xlabel('AoI'), ylabel('Inference Error 100(1-IoU)')
yyaxis right
errorbar(AoI, p2, variance2)
xlabel('AoI'), ylabel('Inference Error (MSE)')
legend('Segmentation', 'Traffic Prediction')

