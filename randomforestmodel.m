

%% load data into table and sort
clear all
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
%% find correlation of coefficents

 %%
close all
wnames = ["haar" "db2" "db3" "db4" "db5" "db6" "sym1" "sym2" "sym3" "sym4" "coif1" "coif2" "coif3" "coif4"]
for i = 1:numel(wnames)
    [psi,xval] = wavefun(wnames(i));
    psi = resample(psi,48,numel(psi));
    arr = 1:48;
    %disp(trapz(arr,psi));
    psi = psi/trapz(arr,psi);
    %psi = [0 1 0]
    %figure
    %plot(psi);
    %figure
    %plot(T.Var4(2000:2720));
    %figure
    r = xcorr(T.Var4(2000:2720),psi);
    figure
    plot(r);
    title(wnames(i))
    disp(mean(r.^2));
    disp('max')
    disp(max(r));
end
%%
close all
figure
wname = 'db2'
wnames = ["haar" "db2" "db3" "db4" "db5" "db6" "sym1" "sym2" "sym3" "sym4" "coif1" "coif2" "coif3" "coif4"]

wt = modwt(T.Var4(1:720),wname);
ww = modwt(T.Var7(1:720),wname);
xcseq = modwtxcorr(wt,ww);
[psi,xval] = wavefun(wname,10);
figure
plot(psi)
title('element')
lev = 6;
%for j = 1:4
    for i = 1:numel(wnames)
        [C3,L3] = wavedec(T.Var4,lev,wnames(i));
        figure
        plot(C3);
        
        %plot(xcseq{i})
        %plot(beep)
        title(wnames(i));
        %ylabel(j)
    end
%end





%%
%index = find((day(time)==1)&(hour(time)==0));



%training

close all
Coef = [];
Xvar = [T.Var4 T.Var7] %T.Var7] %T.Var6]; %T.Var6] %T.Var6 T.Var11] %T.Var6 T.Var9] %T.Var9 T.Var6];% T.Humidity];
Yvar = [T.Var6];
trainingcut = 50000 %floor(size(Yvar,1)*3/4);
trainingX = gpuArray(Xvar(1:trainingcut,:));

testingX = gpuArray(Xvar(trainingcut:end,:));
trainingY = gpuArray(Yvar(1:trainingcut));
testingY = gpuArray(Yvar(trainingcut:end,:));
jump = 168


interval = 48*364;
pint = 168;
%numtraininc = floor(numel(trainingX)/timeinc);

%testing
num = 1460;
coefarr = gpuArray([]);%gpuArray(zeros(trainingcut-interval+1,num));
meanY =gpuArray([]);%gpuArray(zeros(trainingcut-interval+1,1));
lev = 5;
wname = 'db6';


%[C1,L1] = wavedec(X,1,'db1');
%[C2,L2] = wavedec(X,2,'db1');

saven = 0;
%create coef etc
normalxarr = [];

for  i = interval:jump:trainingcut
    %C3 = X((i -1)*(timeinc)+1:i*timeinc);
    %split into wavelet group
    Coef = [];
    % temp coef
    [C3,L3] = wavedec(trainingX(i-interval+1:i,1),lev,wname);
    [cd1,cd2,cd3,cd4,cd5] = detcoef(C3,L3,[1 2 3 4 5]);
    C3 = gather(C3);
    a5 = appcoef(C3,L3,wname,5);
    test = randn(size(cd3));
    Coef = [a5];
    
    [C3,L3] = wavedec(trainingX(i-interval+1:i,2),lev,wname);
    [cd4,cd5] = detcoef(C3,L3,[4,5]);
    %Coef = [Coef;cd5(300:end);cd4(700:end)];
    normalx = trainingX(i-interval+1:i,2);
    
    coefarr= [coefarr;Coef'];
    normalxarr = [normalxarr; normalx'];
    meanY = [mean(Yvar(i:i+pint));meanY];
    if(i>saven+10000)
        disp(i)
        saven = i;
    end
    
end


coefarr = gather(coefarr);
meanY = gather(meanY);
normalxarr= gather(normalxarr);
Mdl = TreeBagger(100,coefarr,meanY, 'Method', 'regression','OOBPrediction','on','NumPredictorsToSample',50,'MinLeafSize',2);
Mdlc = TreeBagger(10,normalxarr,meanY, 'Method', 'regression','OOBPrediction','on','NumPredictorsToSample',50,'MinLeafSize',3);
figure
plot(oobError(Mdl))
xlabel('Number of Trees')
ylabel('MSE Error')
%Y = zeros(size(X,1),1)+ mean(T.WindSpeed_km_h_);
%T = readtable('myfile.csv','NumHeaderLines',3);

% calculate MSE From train to test

% split testing set
%numtestinc = floor(numel(testingX)/timeinc);
coefarrtest =gpuArray([]);% gpuArray(zeros(10000,num));
meanTest = gpuArray([]);%gpuArray(zeros(10000,1));
normalxtarr = [];
for i = interval:jump:numel(testingY)- pint
    Coef = [];

        [C3,L3] = wavedec(testingX(i-interval+1:i,1),lev,wname);
        [cd1,cd2,cd3,cd4,cd5] = detcoef(C3,L3,[1 2 3 4 5]);
        C3 = gather(C3);
        a5 = appcoef(C3,L3,wname,5);
        Coef = [a5];
        
        normalxt = testingX(i-interval+1:i,2);
        
        [C3,L3] = wavedec(testingX(i-interval+1:i,2),lev,wname);
    [cd4,cd5] = detcoef(C3,L3,[4, 5]);
    %Coef = [Coef;cd5(300:end);cd4(700:end)];
        
        
    %[cd1,cd2,cd3,cd4] = detcoef(C3,L3,[1 2 3 4]);
       normalxtarr = [normalxtarr; normalxt'];
        coefarrtest= [coefarrtest;Coef'];
        meanTest= [mean(testingY(i:i+pint));meanTest];
end
coefarrtest = gather(coefarrtest);
meanTest= gather(meanTest);
predarr = predict(Mdl,coefarrtest);
err = immse(predarr,meanTest)
normalxtarr = gather(normalxtarr);
predarrc = predict(Mdlc, normalxtarr);

err2 = immse(predarrc,meanTest)
figure
hold on
plot(predarr);
plot(meanTest);
legend('Prediction','Truth')