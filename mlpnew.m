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
wname = 'db2';
lev = 5;
%% train net
len = 96448;
[swa, swd] = swt(T.Var7(1:len),lev,wname);
x = iswt(swa,swd,wname);
traininginterval = 704 ;
jump = 1;
coef = cell(1,2*lev )
for level = 1:lev
    coef{level} = {reshape(swd(level,:),traininginterval,len/traininginterval)'} ;
    coef{lev + level} = {reshape(swa(level,:),traininginterval,len/traininginterval)'};
end

net = feedforwardnet(30,'trainlm');
net.trainParam.max_fail  = 20
trainednets = cell(1,level)
for i =1:size(coef,2)
    feature = cell2mat(coef{i});
    trainednets{i} = train(net,feature(:,end-1)',feature(:,end)');
end


%% test net
swdf
% getcoef
for i =1:size(coef,2)
    feature = cell2mat(coef{i});
    net2 = trainednets{i};
    feature(:,end) = net2(feature(:,end-1)');
    reshape(feature,1,96448);
end