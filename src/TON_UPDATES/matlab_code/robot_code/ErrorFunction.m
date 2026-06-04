close all;
clear all;
clc;
clf;

p1=readtable("detection_results_robot_9_error_function.csv").Error(1:end);
p2=readtable("detection_results_robot_1_error_function.csv").Error(1:end);
p3=readtable("detection_results_robot_8_error_function.csv").Error(1:end);
p4=readtable("detection_results_robot_2_error_function.csv").Error(1:end);
p5=readtable("detection_results_robot_4_error_function.csv").Error(1:end);

plot(0:40,p1, 'bo--', 0:40,p2,'rx--', 0:40,p3, 'gd--', 0:40,p4, 'k-', 0:40,p5, 'kx-');
ylabel('1-IoU'), xlabel('AoI')
legend('robotCar1', 'robotCar2', 'robotCar3', 'robotCar4', 'robotCar5')
