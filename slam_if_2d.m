%
% SLAM_IF_2D performs an interactive simulation of SLAM with information filter.
%   SLAM_IF_2D() is a two-dimensional linear simulation of Simultaneous 
%   Localization and Mapping (SLAM) with the information filter. It is meant
%   to serve as an introduction to the basic steps of IF SLAM and to explore
%   some important traits of feature-based SLAM information filters.
%
%   This code is rewritten on the basis of Matthew R. Walter's code, which as 
%   part of a 2006 SLAM Summer School(Massachusetts Institute of Technology)
%   lab on the information filter.
%
%   For the most part, I have tried to be pretty verbose when it comes
%   to my comments and hope that they help you to understand what is going on.
%   Also, while there were not any obvious bugs upon my final writing, I can
%   not guarantee that the code is free of errors.
%
% ------------
% References:
%
%   R. Eustice, H. Singh, and J. Leonard. Exactly Sparse Delayed State
%   Filters. ICRA 2005.
%
%   M. Walter, R. Eustice, and J. Leonard. A Provably Consistent Method
%   for Imposing Exact Sparsity in Feature-based SLAM Information Filters.
%   ISRR 2005.
%
%-----------------------------------------------------------------
%    History:
%    Date            Who         What
%    -----------     -------     -----------------------------
%    08-10-2006      mrw         Created and written.
%    31-7-2015       Li Ying     Rewritten 

function slam_if_2d()


% Clear variables
clear global;
clear all;

global TheJournal;
global Truth;
global Graphics;
global Params;
global Data;


Params.verbose = input('Enter 1 for verbose output, else 0: ');





%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Simulation Parameters

Params.PlotPeriod = 20;
Params.Verbose = 0;
Params.PlotSwitch = 1;

% System Parameters
Params.Q = [0.15 0.1; 0.1 0.15].^2;
Params.R = [0.2 0.1; 0.1 0.2].^2;


Params.MaxObsRange = 10;
Params.nObsPerIteration = 8;
Params.density = 0.05;      % density of features


% The information form utilizes the inverse of Q and R in sparse form
Params.Q = sparse(Params.Q);
Params.R = sparse(Params.R);
Params.Qinv = inv(Params.Q);
Params.Rinv = inv(Params.R);



Params.MapSize = 20;        % width and height of the map
Params.nRepeat = 2;         % number of times that the vehicle goes around the loop
Params.maxVelocity = 0.3;   

Params.numFeatures = floor(Params.density*Params.MapSize^2);
Params.FeatureSpacing = ((Params.MapSize^2)/Params.numFeatures)/5;


Params.ObsFrequency = 1;        % [0,1] likelihood of observations




%
% Filter initialization

Sigma = diag([0.001 0.001]);     % initial 2x2 covariance matrix
x = [0; 0];                      % initial state

% Initial information matrix, Lambda = inv(Sigma), and information vector, eta = Lambda*x
% We store the information matrix in sparse form and allocate all the memory beforehand.
% As a result, size(Lambda,1) is proportional to the total number of features in the environment
% and not the size of the current state, which is 2. This speeds things up by allowing us to avoid
% some overhead in Matlab but requires that we carry around a list of indices of where stuff is in this
% large matrix. Right now we only have 2 states, so our index is [1; 2]
TheJournal.II = [1:2]';
TheJournal.Lambda = spalloc(2*(1+Params.numFeatures),2*(1+Params.numFeatures),(2*(1+Params.numFeatures))^2);
Temp = sparse(inv(Sigma));
Temp = (Temp + Temp')/2;
TheJournal.Lambda(1:2,1) = Temp(:,1);
TheJournal.Lambda(1:2,2) = Temp(:,2);
TheJournal.eta = TheJournal.Lambda(TheJournal.II,TheJournal.II)*x;


% A lookup table to keep track of feature id's and their corresponding location within the state vector
TheJournal.LookUP = [NaN 1; 1 1];


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%







if(Params.PlotSwitch)
    Graphics.MainFigure = figure(1);
    clf;
    subplot(1,2,1)
    hold on;
    axis square;
    set(gcf,'DoubleBuffer','on','Color',[1 1 1]);
    set(gca,'Box','on');
    
    subplot(1,2,1)
    cla;
    set(gcf,'ColorMap',colormap(flipud(colormap('gray'))));
    
    if(Params.verbose)
        Graphics.SecondFigure = figure(2);
        clf;
        Temp = get(gcf,'Position');
        Temp(3:4) = [300 300];
        set(gcf,'ColorMap',colormap(flipud(colormap('gray'))),'Position',Temp);
    end;
        
    figure(Graphics.MainFigure);
    
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Graphic settings
    Graphics.TrueColor = 'g';
    Graphics.Color = 'r';

    Graphics.curLaser = [];
    Graphics.TrueVehiclePose = [];  % handle for plot of true vehicle pose
    Graphics.VehiclePose = [];      % handle for plot of filter estimate for vehicle pose
    Graphics.VehicleEllipse = [];   %   "     "    "  "     "   pose estimate uncertainty ellipse
    Graphics.Features = [];         % handle for plot of filter estimates for feature locations
    Graphics.FeatureEllipse = [];   % handle for plot of feature uncertainty ellipse
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
end;




%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Generate the predicted and true motion (i.e. noise-corrupted) of the vehicle
MakeSurvey();
% Place the features in the world
MakeFeatures();
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%





%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Various logs for post-analysis
TheJournal.Feature = [];
TheJournal.ProjectionTime = [];
TheJournal.UpdateTime = [];


% The vehicle history log keeps track of the estimated vehicle pose error together with
% the diagonal of the Sigma_vehicle sub-block and the chi-square error.
% Each column corresponds to a different time
verror = x - Truth.VehicleTrajectory(2:3,1);
TheJournal.VehicleHistory = [1; verror; diag(Sigma); verror'*inv(Sigma)*verror];
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%




%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%   NOW WE LOOP OVER THE SIMULATION AND IMPLEMENT THE FILTER


tstep = floor(0.1*Truth.VehicleTrajectory(1,end));  % time step simply for printing progress update


for k=2:size(Truth.VehicleTrajectory,2)

    if(~mod(k,tstep))
        fprintf(1,'Simulation is at k = %i, %.2f%% done \n',k,100*k/Truth.VehicleTrajectory(1,end));
    end;


    if(Params.PlotSwitch & ~isempty(Graphics.curLaser))
        set(Graphics.curLaser,'Visible','off');
    end;


    % Perform the motion update in the information form as the addition of the new pose
    % to the state followed by the marginalization of the past pose
    Information_DoProjection(k);
    %%
    % To calculate the mean vector of robot pose based on odemetry in every step
    Sigma = TheJournal.Lambda(TheJournal.II,TheJournal.II)\eye(length(TheJournal.II));
    x2 = Sigma*TheJournal.eta;
    TheJournal.muPre(1,k) = x2(1);
    TheJournal.muPre(2,k) = x2(2);

    % update the plot of the vehicle and map;
    if(Params.PlotSwitch)
        PlotState(k);
        title(['k = ' int2str(k) ' ( ' num2str(100*k/size(Truth.VehicleTrajectory,2),2) ' percent )']);
        drawnow;
    end;
    
    
    % Simulate feature observations
    Observations = [];
    if(rand <= Params.ObsFrequency)
        Observations = GetFeatureObservations(k);
    end;

    if(~isempty(Observations))

        % data association is given by the first row
        DA = Observations(1,:);

        % Differentiate between features that have already been mapped and those that are new
        if(size(TheJournal.LookUP,1) > 2)
            [C,IA,IB] = intersect(DA,TheJournal.LookUP(3:end,1));
            OldObs = Observations(:,IA);
            
            [C,IA] = setdiff(Observations(1,:),OldObs(1,:));
            NewObs = Observations(:,IA);
        else
            OldObs = [];
            NewObs = Observations;
        end;
        
        % Visualize observations
        if(Params.PlotSwitch)
            xv = GetMean(1:2);  % Estimate of the current vehicle pose
            [C,IA,IB] = intersect(Truth.Features(1,:),Observations(1,:));

            X = [repmat(xv(1),1,size(Observations,2)); Truth.Features(2,IA)];
            Y = [repmat(xv(2),1,size(Observations,2)); Truth.Features(3,IA)];
            
            figure(Graphics.MainFigure);
            subplot(1,2,1);
            Graphics.curLaser = plot(X,Y,'y');
            set(Graphics.curLaser,'Color',[0.268 0.686 1.00]);
        end;
            
        
        % If the vehicle has observed new features, add them
        if(~isempty(NewObs))
            Information_AddFeature(NewObs);
        end;
        
        % If we have reobserved known features, run an update step
        if(~isempty(OldObs))
            Information_DoUpdate(OldObs);
        end;
    end;


    % update the plot of the vehicle and map;
    if(Params.PlotSwitch)
        PlotState(k);
        title(['k = ' int2str(k) ' ( ' num2str(100*k/size(Truth.VehicleTrajectory,2),2) ' percent )']);
        drawnow;
        ShowLinkStrength();
    end;
   
    if(~mod(k,5))
        DoLogging(k)
    end;

    if(~issparse(TheJournal.Lambda))
        fprintf(1,'k = %i : Lambda is no longer sparse \n',k);
    end;
    %%
    % To calculate the mean vector of robot pose based on canonical gaussian parameterization in every step
    Sigma = TheJournal.Lambda(TheJournal.II,TheJournal.II)\eye(length(TheJournal.II));
    x1 = Sigma*TheJournal.eta;
    TheJournal.mu(1,k) = x1(1);
    TheJournal.mu(2,k) = x1(2);
    ss = inv(TheJournal.Lambda);
    [m,n] = size(ss);
    l=0;
    ll=0;
    for i = 1:m
        for j = 1:n
            if (TheJournal.Lambda(i,j)==0)
                l =l+1;
            end
            if (ss(i,j)==0)
                ll = ll+1;
            end
        end
    end
    L(k) = l;
    LL(k) = ll
    

end;



% Plot the history for the shared information between the first 5 features and the robot.
figure(3);
clf;
hold on;
cmap = [colormap('hot'); colormap('jet'); colormap('gray')];
Temp = randperm(size(cmap,1));

Handles = [];
IDs = [];
for i=1:min(size(TheJournal.Feature,2),5)
    IDs = [IDs; TheJournal.Feature{i}.id];
    thandle = plot(TheJournal.Feature{i}.Link(1,:),(abs(TheJournal.Feature{i}.Link(2,:))),'k-');
    set(thandle,'Color',cmap(Temp(i),:));
    Handles(end+1) = thandle;
end;
    
legend(Handles,int2str(IDs));
xlabel('Time','FontSize',14);
ylabel('Link Strength','FontSize',14);
title('Shared Feature-Vehicle Information vs Time','FontSize',14);
axis tight;
set(gca,'box','on','yscale','log','ylim',[0 Truth.VehicleTrajectory(1,end)],'FontSize',14);

%%
figure(4);
clf;
%     subplot(1,2,1)
hold on;box on;
timeSeq = 1 : size(Truth.VehicleTrajectory,2);
    hh1 = plot (Truth.VehicleTrajectory(2,:),Truth.VehicleTrajectory(3,:),'.-b');
    
    hh2 = plot(Truth.Features(2,:),Truth.Features(3,:),['k.']);
    text(Truth.Features(2,:),Truth.Features(3,:)-1.0,int2str(Truth.Features(1,:)'));
% end
%  
TheJournal.mu(:,1) = x;
hh3 = plot(TheJournal.mu(1,:),TheJournal.mu(2,:),'-r');
TheJournal.muPre(:,1) = x;
hh4 = plot(TheJournal.muPre(1,:),TheJournal.muPre(2,:),'g--');
label = [hh1, hh2, hh3, hh4];
hx    = legend(label,'真实路径','真实路标','正则参数化估计','里程估计');
% set(hx,'box','off','location','NorthWest','Orientation','horizontal');
set(hx);
%%
%  figure(5);
figure('name','errorInX','color','w');
clf;
width = 1;
errPre = abs(TheJournal.muPre(1,:) - Truth.VehicleTrajectory(2,:));
errUp = abs(TheJournal.mu(1,:) - Truth.VehicleTrajectory(2,:));
hold on;box on;
timeSeq = 1 : size(Truth.VehicleTrajectory,2);
h1    = plot(timeSeq, errPre(1, timeSeq), 'g--','linewidth',width);
h2    = plot(timeSeq, errUp(1, timeSeq), '-r','linewidth',width);
label = [h1, h2];
hx    = legend(label,'里程估计','正则参数化估计');
set(hx,'box','off','location','NorthWest','Orientation','horizontal');
xlabel('时间(s)');
ylabel('x方向绝对误差(m)');
%%
figure('name','稀疏性','color','w');
clf;
width = 1.5;
% errPre = abs(xxx(1,:) - Truth.VehicleTrajectory(2,:));
% errUp = abs(xx(1,:) - Truth.VehicleTrajectory(2,:));
% plot(err1,'r');
% plot(err2,'g');
% figure('name','errorInX','color','w');
hold on;box on;
L(1)=2;
LL(1)=2;
timeSeq = 1 : size(Truth.VehicleTrajectory,2);
h1    = plot(timeSeq, L(1, timeSeq), '--','linewidth',width);
h2    = plot(timeSeq, LL(1, timeSeq), '-r','linewidth',width);
% h3    = plot(timeSeq, errFastSLAM(1, timeSeq), 'b:','linewidth',width);
label = [h1, h2];
hx    = legend(label,'信息矩阵','协方差矩阵');
set(hx);
% set(hx,'box','off','location','NorthWest','Orientation','horizontal');
% % axis([min(timeSeq) max(timeSeq) 0 12]); 
xlabel('时间(s)');
ylabel('矩阵中元素为零的个数');


return;






%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This function executes the motion update for the information filter by forming it as the addition
% of the new pose to the state vector followed by the marginalization of the previous pose
function Information_DoProjection(k);

global Params;
global TheJournal;
global Data;
global Graphics;


% Motion model:
%   x_new = x_old + velocity_x * delta_time
%   y_new = y_old + velocity_y * delta_time

% Jacobian
F = speye(2);

% Velocity vector
u = Data.Velocity(2:3,find(Data.Velocity(1,:)==k));


LambdaOld = TheJournal.Lambda(TheJournal.II,TheJournal.II);


%%%%%%%%%%%%%%%
% The time projection operation for the information matrix and information vector
% as described in Eustice et al. (eustice05a) Section III.C.
tic;
Psi = inv(Params.Q + F*inv(TheJournal.Lambda(1:2,1:2))*F');
Omegainv = inv(TheJournal.Lambda(1:2,1:2) + F*Params.Qinv*F');

eta1 = Params.Qinv*F*Omegainv*TheJournal.eta(1:2) + Psi*u;
if(length(TheJournal.eta) > 2)
    eta2 = TheJournal.eta(3:end) - TheJournal.Lambda(3:TheJournal.II(end),1:2)*Omegainv*(TheJournal.eta(1:2)-F'*Params.Qinv*u);

    Lambda12 = Params.Qinv*F*Omegainv*TheJournal.Lambda(1:2,3:TheJournal.II(end));
    Lambda22 = TheJournal.Lambda(3:TheJournal.II(end),3:TheJournal.II(end)) - TheJournal.Lambda(3:TheJournal.II(end),1:2)*Omegainv*TheJournal.Lambda(1:2,3:TheJournal.II(end));

    TheJournal.eta = [eta1; eta2];
    TheJournal.Lambda(TheJournal.II,TheJournal.II) = [Psi Lambda12; Lambda12' Lambda22];
else
    TheJournal.eta = eta1;
    TheJournal.Lambda(TheJournal.II,TheJournal.II) = Psi;
end
comptime = toc;
TheJournal.ProjectionTime = [TheJournal.ProjectionTime [k; length(TheJournal.II); comptime]];
%%%%%%%%%%%%%%%


% Update the timestamp in the LookUP table
TheJournal.LookUP(1,2) = k;


if(Params.verbose)
    fprintf(1,'\n\n');
    fprintf(1,'----------------------- \n\n');
    fprintf(1,'TIME PROJECTION STEP: \n\n')
    fprintf(1,'    The IF has just performed the time prediction step. \n');
    fprintf(1,'    Compare the new information matrix in Figure 1 to the\n');
    fprintf(1,'    structure beforehand as shown in Figure 2.\n');
    ShowLinkStrength(TheJournal.Lambda(TheJournal.II,TheJournal.II),0);
    ShowLinkStrength(LambdaOld,0,Graphics.SecondFigure);
    pause;
    fprintf(1,'----------------------- \n\n');
end;



return;







%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Function to perform update step for information filter
function Information_DoUpdate(Observations)

global TheJournal;
global Params;
global Graphics;


k = TheJournal.LookUP(1,end);
xi = GetStateIndex(1,k);


% Measurement model:
%   z_x = x_f - x_v
%   z_y = y_f - y_v
% Note: with a linear observation model, we don't need the mean

LambdaOld = TheJournal.Lambda(TheJournal.II,TheJournal.II);

% Loop over the measurement data, performing an individual update for each observation
H_I = []; H_J = []; H_S = [];
Rinv_I = []; Rinv_J = []; Rinv_S = [];
Z = [];
II = [1:2]';
for i=1:size(Observations,2)
    id = Observations(1,i);
    xfi = GetStateIndex(id,k);
        
    %%%%%%%%%%%%%%%
    % The update step for the information filter as described
    % in Eustice et al. (eustice05a) Section III.B.
    
    
    temp = 2*i - 1;
    H_I = [H_I; [temp; temp; temp+1; temp+1]];
    H_J = [H_J; [1; temp+2; 2; (temp+2)+1]];
    H_S = [H_S; [-1; 1; -1; 1]];
    
    II = [II; [xfi; xfi+1]];
    
    Rinv_I = [Rinv_I; [temp; temp; temp+1; temp+1]];
    Rinv_J = [Rinv_J; [temp; temp+1; temp; temp+1]];
    Rinv_S = [Rinv_S; Params.Rinv(:)];
    
    Z = [Z; Observations(2:3,i)];
    
    %%%%%%%%%%%%%%%
end;

H = sparse(H_I,H_J,H_S,max(H_I),max(H_J));
RINV = sparse(Rinv_I,Rinv_J,Rinv_S);

tic;
TheJournal.Lambda(II,II) = TheJournal.Lambda(II,II) + H'*RINV*H;
TheJournal.eta(II) = TheJournal.eta(II) + H'*RINV*Z;
comptime = toc;

TheJournal.UpdateTime = [TheJournal.UpdateTime [k; length(TheJournal.II); comptime]];


if(Params.verbose)
    fprintf(1,'\n\n');
    fprintf(1,'----------------------- \n\n');
    fprintf(1,'MEASUREMENT UPDATE STEP: \n\n')
    fprintf(1,'    The IF has just performed the update step based upon observations \n');
    fprintf(1,'    of features: ');
    disp(Observations(1,:)); fprintf(1,'\n');
    fprintf(1,'    Compare the new information matrix in Figure 1 to the\n');
    fprintf(1,'    structure beforehand as shown in Figure 2.\n');
    ShowLinkStrength(TheJournal.Lambda(TheJournal.II,TheJournal.II),0);
    ShowLinkStrength(LambdaOld,0,Graphics.SecondFigure);
    pause;
    fprintf(1,'----------------------- \n\n');
end;





return;


    
            


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Function to add features
function Information_AddFeature(Observations)

global TheJournal;
global Params;
global Graphics;


% Measurement model:
%   z_x = x_f - x_v
%   z_y = y_f - y_v

% Jacobian
G = speye(2);

xi = GetStateIndex(1,TheJournal.LookUP(1,end));



if(Params.verbose)
    ShowLinkStrength(TheJournal.Lambda(TheJournal.II,TheJournal.II),0,Graphics.SecondFigure);
end;


% Note that, in the linear case, we don't need the mean to add the feature
% xf = g(xv,z) + w = (mu_v + z) + w
for i=1:size(Observations,2)
    id = Observations(1,i);
    
    II_new = TheJournal.II(end-1:end) + 2;
    
    %%%%%%%%%%%%%%%
    % The operation of adding features with the information form as described
    % in Eustice et al. (eustice05a) Section III.A.
    
    TheJournal.Lambda(1:2,II_new) = -G'*Params.Rinv;
    TheJournal.Lambda(II_new,1:2) = -Params.Rinv*G;
    TheJournal.Lambda(II_new,II_new) = Params.Rinv;
    TheJournal.Lambda(1:2,1:2) = TheJournal.Lambda(1:2,1:2) + G'*Params.Rinv*G;

    etaf_new = Params.Rinv*Observations(2:3,i);

    TheJournal.eta = [TheJournal.eta; etaf_new];
    TheJournal.eta(1:2) = TheJournal.eta(1:2) - G'*Params.Rinv*Observations(2:3,i);
    
    TheJournal.II = [TheJournal.II; II_new];
    %%%%%%%%%%%%%%%
    
    % Plotting stuff
    if(Params.PlotSwitch)
        xyf = GetMean(1:2) + Observations(2:3,i);
        figure(Graphics.MainFigure);
        subplot(1,2,1);
        thandle = plot(xyf(1),xyf(2),'rx');

        Graphics.Features = [Graphics.Features [id; thandle]];
    end;
        
    % Keep track of the shared information between the robot and feature
    TheJournal.Feature{end+1}.id = id;
    LambdaNorm = rhomatrix(TheJournal.Lambda(TheJournal.II,TheJournal.II));
    TheJournal.Feature{end}.Link = [TheJournal.LookUP(1,2); det(LambdaNorm(1:2,end-1:end))];
    
    % Now, update the LookUP table
    TheJournal.LookUP = [TheJournal.LookUP; id NaN*ones(1,size(TheJournal.LookUP,2)-1)];
    TheJournal.LookUP(end,end) = length(TheJournal.eta)-1;
end;
    

if(Params.verbose)
    fprintf(1,'\n\n');
    fprintf(1,'+++++++++++++++++++++++ \n\n');
    fprintf(1,'ADDING FEATURE(S): \n\n')
    fprintf(1,'    We have added features: ');
    disp(Observations(1,:));
    fprintf(1,'\n\n');
    fprintf(1,'    Notice that they share information only with the robot state\n\n');
    fprintf(1,'+++++++++++++++++++++++ \n\n');
    ShowLinkStrength(TheJournal.Lambda(TheJournal.II,TheJournal.II),0);
    pause;
end;


return;




%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Function which pictoralizes links in the information matrix
function ShowLinkStrength(Lambda,twoposeflag,fhandle)

global TheJournal;
global Graphics

if(nargin <2)
    twoposeflag = 0;
end;

% We are intersted in the normalized information matrix
if(nargin==0)
    LambdaNorm = rhomatrix(TheJournal.Lambda(TheJournal.II,TheJournal.II));
else
    LambdaNorm = rhomatrix(Lambda);
end;

% The strength of each link is proportional to the determinant of the
% corresponding 2x2 submatrix of Lambda. In general, the determinant
% of the matrices along the diagonal of Lambda (which represent self-links)
% will be much larger and we "tone them down" for the sake of rendering.
ILinks = sparse(size(LambdaNorm,1)/2,size(LambdaNorm,2)/2);
for i=1:2:size(LambdaNorm,1)
    for j=(i):2:size(LambdaNorm,2)
        temp = det(LambdaNorm(i:(i+1),j:(j+1)));
        ILinks((i+1)/2,(j+1)/2) = temp;
        ILinks((j+1)/2,(i+1)/2) = temp;
    end;
end;

if(size(ILinks,1) > 2)
    %ILinks = spdiags(max(ILinks(:))*ones(size(ILinks,1),1),0,ILinks);
end;

if(nargin < 3)
    figure(Graphics.MainFigure);
    subplot(1,2,2);
else
    figure(fhandle);
end;

cla;
imagesc(ILinks);
axis image;
hold on;
set(gca,'xtick',1.5:1:(size(ILinks,1)-0.5),'ytick',1.5:1:(size(ILinks,1)-0.5));

set(gca,'xtick',[],'ytick',[],'box','on');
xlim = get(gca,'xlim');
ylim = get(gca,'ylim');

[X,Y] = meshgrid(xlim(1):1:xlim(2),xlim(1):1:xlim(2));
plot(X,Y,'r-');
plot(Y,X,'r-');

temp = int2str(TheJournal.LookUP(3:end,1));
if(twoposeflag)
    temp = strvcat('v+','v-',temp);
else
    temp = strvcat('v+',temp);
end;

X = 1:size(ILinks,2);
Y = repmat(0.2,1,length(X));
text(X,Y,temp);
text(Y-0.05*X(end),X,temp)


[I,J] = find(ILinks);
plot(I,J,'k.');
xlabel('Dots denote nonzero entries','Fontsize',10);


return;





%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This function records data
function DoLogging(k)

global TheJournal;
global Params;
global Truth;


if(size(TheJournal.LookUP,1) == 2)
    return;
end;


Lambda = rhomatrix(TheJournal.Lambda(TheJournal.II,TheJournal.II));

for i=3:size(TheJournal.LookUP,1)
    xfi = TheJournal.LookUP(i,2);
    
    TheJournal.Feature{i-2}.Link(:,end+1) = [k; det(Lambda(1:2,xfi:(xfi+1)))];
end;



% Enable to log errors in the state estimates
if(0)
    % Calculate the covariance matrix and mean via brute-force inversion of the information matrix
    Sigma = inv(TheJournal.Lambda(TheJournal.II,TheJournal.II));
    Sigma = (Sigma + Sigma')/2;     % Ensures symmetry
    x = Sigma*TheJournal.eta;


    % Log vehicle data
    xv_truth = Truth.VehicleTrajectory(2:3,find(Truth.VehicleTrajectory(1,:)==k));
    verror = x(1:2) - xv_truth;
    chisqr_error = verror'*inv(Sigma(1:2,1:2))*verror;

    TheJournal.VehicleHistory = [TheJournal.VehicleHistory [k; verror; diag(Sigma(1:2,1:2)); chisqr_error]];



    if(0 & size(TheJournal.LookUP,1) > 2)

        TheJournal.MapError = [TheJournal.MapError [k; 0]];

        nFeatures = size(TheJournal.LookUP,1)-2;

        for i=3:size(TheJournal.LookUP,1)
            id = TheJournal.LookUP(i,1);
            xi = GetStateIndex(id,k);

            % Error in estimated feature position
            xf_truth = Truth.Features(2:3,find(Truth.Features(1,:)==id));
            feat_error = x(xi:xi+1)-xf_truth;

            %TheJournal.Feature{id}.History = [TheJournal.Feature{id}.History [k; feat_error; diag(Sigma(xi:xi+1,xi:xi+1)); feat_error'*inv(Sigma(xi:xi+1,xi:xi+1))*feat_error]];

            % Keep track of map convergence
            TheJournal.MapError(2,end) = TheJournal.MapError(2,end) + (feat_error'*feat_error)/nFeatures;
        end;

    end;
end;


return;




    


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Function to extract mean state estimates
%
function x = GetMean(xi)

global TheJournal;


% Right now, fully invert the information matrix to get the mean
mu = TheJournal.Lambda(TheJournal.II,TheJournal.II)\TheJournal.eta;
x = mu(xi);
    
return;





%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Function to generate feature observations
function Observations = GetFeatureObservations(k)

global TheJournal;
global Truth;
global Params;
global Data;

Observations = [];

xv = Truth.VehicleTrajectory(2:3,find(Truth.VehicleTrajectory(1,:)==k));
vvel = Data.Velocity(2:3,find(Data.Velocity(1,:)==k));

Temp = sqrt(sum((Truth.Features(2:3,:)-repmat(xv,1,size(Truth.Features,2))).^2));
Temp = [Truth.Features(1,:); Temp];
Temp = Temp(1,find(Temp(2,:) <= Params.MaxObsRange));

    
% Generate correlated measurement noise using the Cholesky Decomposition: R = X'*X
% for which: w = U*randn(2,1) ---> cov(w) = U*I*U' = U*U' = R;  (U*U'=R=X'X --> U=X')
U = chol(Params.R)';


if(length(Temp) >= Params.nObsPerIteration)
    while(size(Observations,2) < Params.nObsPerIteration)
        i = ceil(rand*length(Temp));
        fid = Temp(i);
        Temp(i) = [];
        
        xf = Truth.Features(2:3,find(Truth.Features(1,:)==fid));
        ObsNoise = (xf-xv) + U*randn(2,1);
        Observations = [Observations [fid; ObsNoise]];
    end;
else
    for i=1:length(Temp)
        fid = Temp(i);
        
        xf = Truth.Features(2:3,find(Truth.Features(1,:)==fid));
        ObsNoise = (xf-xv) + U*randn(2,1);
        Observations = [Observations [fid; ObsNoise]];
    end;
end;

return;
        



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function PlotState(k);

global TheJournal;
global Graphics;
global Truth;

figure(Graphics.MainFigure);
subplot(1,2,1)

% Ground truth vehicle plot
x = Truth.VehicleTrajectory(2:3,k);
if(isempty(Graphics.TrueVehiclePose))
    Graphics.TrueVehiclePose = plot(x(1),x(2),'ko','MarkerFaceColor','r','MarkerSize',6);
    set(Graphics.TrueVehiclePose,'Color',Graphics.TrueColor);
else
    set(Graphics.TrueVehiclePose,'xdata',x(1),'ydata',x(2));
end;


% Brute-force estimate of the covariance matrix and mean vector
Sigma = TheJournal.Lambda(TheJournal.II,TheJournal.II)\eye(length(TheJournal.II));
x = Sigma*TheJournal.eta;

if(isempty(Graphics.VehiclePose))
    Graphics.VehiclePose = plot(x(1),x(2),'k^','MarkerFaceColor','r','MarkerEdgeColor','k','MarkerSize',8);
    set(Graphics.VehiclePose,'Color',Graphics.Color);
else
    lastvx = get(Graphics.VehiclePose,'xdata');
    lastvy = get(Graphics.VehiclePose,'ydata');

    set(Graphics.VehiclePose,'xdata',x(1),'ydata',x(2));
end;


if(isempty(Graphics.VehicleEllipse))
    Graphics.VehicleEllipse = DrawEllipse([x(1); x(2)],Sigma(1:2,1:2),3,Graphics.Color);
else
    [Sigmax,Sigmay] = GetEllipse([x(1); x(2)],Sigma(1:2,1:2),3);
    set(Graphics.VehicleEllipse,'xdata',Sigmax,'ydata',Sigmay);
end;

for i=1:size(Graphics.Features,2)
    id = Graphics.Features(1,i);
    thandle = Graphics.Features(2,i);

    xfi = GetStateIndex(id,k);
    try
        set(thandle,'xdata',x(xfi),'ydata',x(xfi+1));
    catch
        keyboard;
    end;
    
    if(isempty(Graphics.FeatureEllipse) | isempty(find(Graphics.FeatureEllipse(1,:)==id)))
        thandle = DrawEllipse(x(xfi:xfi+1),Sigma(xfi:xfi+1,xfi:xfi+1),3,Graphics.Color);
        Graphics.FeatureEllipse = [Graphics.FeatureEllipse [id; thandle]];
    else
        [Sigmax,Sigmay] = GetEllipse(x(xfi:xfi+1),Sigma(xfi:xfi+1,xfi:xfi+1),3);
        thandle = Graphics.FeatureEllipse(2,find(Graphics.FeatureEllipse(1,:)==id));
        set(thandle,'xdata',Sigmax,'ydata',Sigmay);
    end;
end;



return;



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function h = DrawEllipse(x,P,nSigma,RGBCol)

[V,D] = eig(P);
y = nSigma*[cos(0:0.1:2*pi);sin(0:0.1:2*pi)];
el = V*sqrtm(D)*y;
el = [el el(:,1)]+repmat(x,1,size(el,2)+1);

h = line(el(1,:),el(2,:));
set(h, 'markersize',0.5,'color',RGBCol);

return;


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [XData,YData] = GetEllipse(x,P,nSigma)

[V,D] = eig(P);
y = nSigma*[cos(0:0.1:2*pi);sin(0:0.1:2*pi)];
el = V*sqrtm(D)*y;
el = [el el(:,1)]+repmat(x,1,size(el,2)+1);

XData = el(1,:);
YData = el(2,:);

return;







%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Function which returns the index for a particular feature/pose
function xi = GetStateIndex(id,k)

global TheJournal;


iRow = find(TheJournal.LookUP(:,1) == id);
iColumn = find(TheJournal.LookUP(1,:) == k);

if(isempty(iRow) || isempty(iColumn))
    fprintf(1,'!!!!!!   Error determining index for vehicle/feature %i at time %i \n',id,k);
else
    xi = TheJournal.LookUP(iRow,iColumn); 
end;

return;




%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Function to generate point features
function MakeFeatures()

global Params;
global Truth;
global Graphics;

Truth.Features = [];

% dist = Params.MaxObsRange/2;
dist = Params.MaxObsRange/3;
minx = min(Truth.VehicleTrajectory(2,:)) - dist;
maxx = max(Truth.VehicleTrajectory(2,:)) + dist;
miny = min(Truth.VehicleTrajectory(3,:)) - dist;
maxy = max(Truth.VehicleTrajectory(3,:)) + dist;


Truth.Features = repmat([minx; miny],1,Params.numFeatures) + [(maxx-minx) 0; 0 (maxy-miny)]*rand(2,Params.numFeatures);
Truth.Features = [2:(Params.numFeatures+1); Truth.Features];

if(Params.PlotSwitch)
    figure(Graphics.MainFigure);
    subplot(1,2,1)
    plot(Truth.Features(2,:),Truth.Features(3,:),['k.']);
    text(Truth.Features(2,:),Truth.Features(3,:)-1.0,int2str(Truth.Features(1,:)'));
end;

return;

 






%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Make me a survey pattern for a vehicle with 2 uncoupled dof
function pattern = MakeSurvey()

global Params;
global Truth;
global Data;


% Generate the 4 waypoints that comprise the corners of the route (a box)
% a = 0.8*Params.MapSize;
% b = 0.8*Params.MapSize;
a = 0.6*Params.MapSize;
b = 0.6*Params.MapSize;
X0 = Params.MapSize*[0.1; 0.1];

Params.Path = repmat([0 a a  0; 0 0 b b],1,Params.nRepeat);
Params.Path = Params.Path + repmat(X0,1,size(Params.Path,2));


Data.Velocity = [1 0 0]';
xold = Params.Path(:,1);
for i=2:size(Params.Path,2)
    tlast = Data.Velocity(1,end);
    xold = Params.Path(:,i-1);
    xnew = Params.Path(:,i);
    xdiff = xnew(1)-xold(1);
    ydiff = xnew(2)-xold(2);
    if(abs(xdiff) > abs(ydiff))
        tend = tlast + ceil(abs(xdiff)/Params.maxVelocity);
        xvel = sign(xdiff)*Params.maxVelocity;
        yvel = ydiff/(tend-tlast);
    elseif(abs(ydiff) > abs(xdiff))
        tend = tlast + ceil(abs(ydiff)/Params.maxVelocity);
        yvel = sign(ydiff)*Params.maxVelocity;
        xvel = xdiff/(tend-tlast);
    end;

    XVel = repmat(xvel,1,tend-tlast);
    YVel = repmat(yvel,1,tend-tlast);
    Data.Velocity = [Data.Velocity [(tlast+1):tend; XVel; YVel]];
end;

% Generate the true, NOISE CORRUPTED, vehicle trajectory
%
% Generate correlated noise using the Cholesky Decomposition: Q = X'*X
% for which: v = U*randn(2,1) --> cov(v) = U*I*U' = U*U' = Q    (U*U' = X'*X --> U = X')
U = chol(Params.Q)';
Truth.VehicleTrajectory = [1; [0;0]];
for i=2:size(Data.Velocity,2)
    VelocityNoise = Data.Velocity(2:3,i) + U*randn(2,1);
    Truth.VehicleTrajectory = [Truth.VehicleTrajectory [i; Truth.VehicleTrajectory(2:3,end)+VelocityNoise]];
end;


return;

