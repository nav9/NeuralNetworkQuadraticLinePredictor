clc;clear;close all;

%NNLayers has to have a minimum of two layers. Even if should be a single
%layer, the second layer here is an indicator of the number of outputs. The
%input to the first layer has to be a minimum of two values. The second
%value being a bias.
NNLayers = [2 3 3 1];%number of nodes in each layer
NNLen = size(NNLayers, 2);
numTrainings = 1000;
exemplars = -1:0.1:1;
bias = 1;
numPredictions = length(exemplars);
activationFn = 'sigmoid';

disp('===== C R E A T E =====');
NN = [];%framework of nodes. The Neural Net
for i = 1:NNLen
    if i == NNLen, L = Layer(NNLayers(i), NNLayers(i), i, activationFn); 
    else L = Layer(NNLayers(i), NNLayers(i+1), i, activationFn); end
    NN = [NN L];
end
if length(NNLayers) > 2, 
    NN(length(NNLayers)-1).setAsSecondLastLayer();
    disp('Only linear combiner in final layer');
end
for i = 1:NNLen%create a doubly linked list
    if i == 1, NN(i).setNeighbours(NN(i), NN(i+1));
    else if i == NNLen, NN(i).setNeighbours(NN(i-1), NN(i));
        else NN(i).setNeighbours(NN(i-1), NN(i+1));end
    end
end

desired = zeros(size(numTrainings));
trained = desired;
trainingOutput = desired;
trainingMSE = [];
disp('===== T R A I N =====');
for epo = 1:numTrainings
    if mod(epo,100)==0, fprintf('Epoch %d\n', epo);end
    for i = 1:length(exemplars),
        x = [exemplars(i); bias];%exemplars(randperm(length(exemplars)))
        desired(i) = 2 * x(1)^2+1;
        trained(i) = x(1);
        NN(NNLen).setDesiredOutput(desired(i));%set in last layer
        NN(1).Train(x);%invoke from first layer 
        trainingOutput(i) = NN(NNLen).getOutput();        
    end
    trainingMSE = [trainingMSE mean(desired-trainingOutput).^2];
    NN(1).UpdateWeights();
    %NN(1).showWeights();
end    

disp('===== P R E D I C T =====');
outputStore = zeros(size(exemplars));
for i = 1:length(exemplars)
    x = [exemplars(i); bias];
    NN(1).Predict(x);
    outputStore(i) = NN(NNLen).getOutput();%last layer stores output
end    

figure(1);
plot(trained, desired, '.b', exemplars, outputStore, '.r');
legend('ideal output', 'trained output');
xlabel('y');ylabel('x');title('outputs');

figure(2);
plot(linspace(1,numTrainings,numTrainings), trainingMSE, 'b');
legend('training MSE');
xlabel('MSE');ylabel('epochs');title('Mean Squared Error (MSE)');

