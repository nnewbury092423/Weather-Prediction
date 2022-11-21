%% test
clear all
filename = 'weatherHistory.csv'

opts = detectImportOptions(filename);
opts.Delimiter = {','};
opts.VariableTypes={'char','char','char','double','double','double','double','double','double','double','double','char'}
T = readtable(filename,opts);
X = [T.Var7]% T.Humidity];
% split into weeks
timearr = cell2mat(T.Var1);
time = datetime(timearr(:,1:19));
%index = find((day(time)==1)&(hour(time)==0));
timeinc = 2000 ;
numtimeinc = floor(numel(X)/timeinc)- 1;
bigarr = zeros(numtimeinc,2000);
Y = zeros(numtimeinc,1);
lev = 3;
wname = 'db1';
nbcol = 64


%[C1,L1] = wavedec(X,1,'db1');
%[C2,L2] = wavedec(X,2,'db1');


for  i = 1:numtimeinc
    %C3 = X((i -1)*(timeinc)+1:i*timeinc);
    [C3,L3] = wavedec(X((i -1)*(timeinc)+1:i*timeinc),3,'db1');
     [cd1,cd2,cd3] = detcoef(C3,L3,[1 2 3]);
    bigarr(i,:)= C3;
    Y(i) = mean(X(i*timeinc:i*timeinc+24));
end

%Y = zeros(size(X,1),1)+ mean(T.WindSpeed_km_h_);
%T = readtable('myfile.csv','NumHeaderLines',3);





    Mdl = TreeBagger(500,bigarr,Y, 'Method', 'regression','OOBPrediction','on','NumPredictorsToSample',50);
figure
plot(oobError(Mdl))
xlabel('Number of Trees')
ylabel('MSE Error')