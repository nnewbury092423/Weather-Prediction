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
lev = 5;
%% train net
len = 96448;
[swa, swd] = swt(T.Var7(1:len),lev,wname);
x = iswt(swa,swd,wname);
traininginterval = 137 ;
jump = 1;
coef = cell(1,2*lev )


for level = 1:lev
    coef{level} = {reshape(swd(level,:),traininginterval,len/traininginterval)'} ;
    coef{lev + level} = {reshape(swa(level,:),traininginterval,len/traininginterval)'};
end


trainednets = cell(1,level)
for i =1:size(coef,2)
    net = feedforwardnet(5,'trainlm');
    net.trainParam.max_fail  = 20
    feature = cell2mat(coef{i})';
    %featuretransposed = feature'
    trainednets{i} = train(net,feature(1:end-1,1:352),feature(end,1:352));
end


%% test net
% getcoef
coeff = zeros(2*lev,len);
for i =1:size(coef,2)
    feature = cell2mat(coef{i})';
    net2 = trainednets{i};
    feature(end,:) = net2(feature(1:end-1,:));
    coeff(i,:) = reshape(feature,1,numel(feature));
end

signal = iswt(coeff(lev:2*lev,:),coeff(1:lev,:),wname);
figure
hold on
prediction = signal(1:137:end)
real = T.Var7(1:137:end)
plot(prediction)
plot(real)
legend('prediction', 'real')