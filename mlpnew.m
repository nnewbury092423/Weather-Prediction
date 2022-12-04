%% load in data
filename = 'weatherHistory.csv'

opts = detectImportOptions(filename);
opts.Delimiter = {','};
opts.VariableTypes={'char','char','char','double','double','double','double','double','double','double','double','char'}
T = readtable(filename,opts);


% split into weeks
timearr = cell2mat(T.Var1);
time = datetime(timearr(:,1:19));
T.Var1 = time;
T = sortrows(T,1);
wname = 'db6';
lev = 3;
%% train net

traininginterval = 144 ;
len = 96480 - traininginterval;
jump = 1;
coef = cell(1,2*lev )
swac = []
swdc = []
for i= 1:traininginterval:len
    [swa, swd] = swt(T.Var7(i:i+traininginterval-1),lev,wname);
    swac = [swac swa];
    swdc = [swdc swd];
end

swdc
%x = iswt(swa,swd,wname);


for level = 1:lev
    coef{level} = {reshape(swdc(level,:),traininginterval,len/traininginterval)'} ;
    coef{lev + level} = {reshape(swac(level,:),traininginterval,len/traininginterval)'};
end

%coef{1} = {reshape(T.Var7(1:len),traininginterval,len/traininginterval)'} 
 
trainednets = cell(1,lev*2)
for i =1:size(coef,2)
    net = feedforwardnet(30,'trainlm');
    net.trainParam.max_fail  = 20;
    net.trainParam.epochs  = 400;
    feature = cell2mat(coef{i})';
    %featuretransposed = feature'
    trainednets{i} = train(net,feature(1:end-1,1:500),feature(end,1:500));
end


%% test net
% getcoef
coeff = zeros(2*lev,len);
for i =1:size(coef,2)
    feature = cell2mat(coef{i})';
    net2 = trainednets{i};
    feature(end,:) = net2(feature(1:end-1,:));
    coeff(i,:) = reshape(feature,1,numel(feature));
    figure
    hold on
    %plot(coeff(i,traininginterval:))
   
end

signal = iswt(coeff(lev+1:2*lev,:),coeff(1:lev,:),wname);
figure
hold on
prediction = signal(traininginterval:traininginterval:end)
test = T.Var7(traininginterval:traininginterval:end)
real = test(1:end)
plot(prediction(1:end))
plot(real)
legend('prediction', 'real')