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
wname = 'db2';
lev = 6;
%% find coefficients
      
    [C3,L3] = wavedec(T.Var7,lev,wname);
    [cd1,cd2,cd3,cd4,cd5,cd6] = detcoef(C3,L3,[1 2 3 4 5 6]);
    a1 = appcoef(C3,L3,wname,1);
    
   
