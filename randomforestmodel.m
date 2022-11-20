%% test
filename = 'weatherHistory.csv'

opts = detectImportOptions(filename);
opts.Delimiter = {','};
opts.VariableTypes={'char','char','char','double','double','double','double','double','double','double','double','char'}
T = readtable(filename,opts);
X = [T.Var4]% T.Humidity];
% split into weeks
timearr = cell2mat(T.Var1);
time = datetime(timearr(:,1:19));
%index = find((day(time)==1)&(hour(time)==0));
timeinc = 720  ;
numtimeinc = 100
bigarr = zeros(numtimeinc,timeinc);
Y = zeros(numtimeinc,1);
for  i = 1:numtimeinc
    bigarr(i,:)= X((i -1)*(timeinc)+1:i*timeinc);
    Y(i) = mean(X((i-1)*(timeinc)+1:i*timeinc));
end

%Y = zeros(size(X,1),1)+ mean(T.WindSpeed_km_h_);
%T = readtable('myfile.csv','NumHeaderLines',3);





Mdl = TreeBagger(100,bigarr,Y, 'Method', 'regression',OOBPrediction="on",NumPredictorsToSample=50);

plot(oobError(Mdl))
xlabel('Number of Trees')
ylabel('MSE Error')