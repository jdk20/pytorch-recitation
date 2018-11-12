clc; clear variables; close all;

net = importONNXNetwork('alexnet.onnx', 'OutputLayerType','classification');
plot(net);