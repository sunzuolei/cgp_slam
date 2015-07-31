%
% SLAM_IF_1D performs an interactive simulation of SLAM with information filter.
%   SLAM_IF_1D() is a one-dimensional (i.e. monobot) linear simulation of 
%   Simultaneous Localization and Mapping (SLAM) with the information filter. 
%   It is meant to serve as an introduction to the basic steps of IF SLAM and 
%   to explore some important traits of feature-based SLAM information filters.
%
%   This code is rewritten on the basis of Matthew R. Walter's code, which as 
%   part of a 2006 SLAM Summer School(Massachusetts Institute of Technology)
%   lab on the information filter
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

function slam_if_1d()


% Clear variables
clear global;
clear all;

global TheJournal;
global Truth;
global Graphics;
global Params;
global Data;



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Simulation Parameters

Params.verbose = 1;     % Verbose output
Params.PlotSwitch = 1;  % Flag that turns plotting on/off. Should be set to 1 (plotting on)

Params.nSteps = 20;

Params.density = 0.35;      % density of features
numfeatures = 5;
Params.nObsPerIteration = 1;

% System Parameters
Params.Q = 0.15^2;
Params.R = 0.05^2;


% Filter initialization
Sigma = 0.001;     % initial 2x2 covariance matrix
x = [0];                      % initial state


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%





%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Set simulation parameters and generate vehicle motion


Params.maxVelocity = 0.5;   

Params.ObsFrequency = 1;        % [0,1] likelihood of observations


Data.Velocity = [1:Params.nSteps; repmat(Params.maxVelocity,1,Params.nSteps)];
U = chol(Params.Q)';
TrueVelocity = Data.Velocity(2,:) + U*randn(1,Params.nSteps);
Truth.VehicleTrajectory = [1:Params.nSteps; cumsum(TrueVelocity)];

mapsize = max(Truth.VehicleTrajectory(2,:));        % width and height of the map
Params.MaxObsRange = 0.3*mapsize;


% Make the features
Truth.Features = 1:((mapsize-1)/numfeatures):mapsize;
Truth.Features = [2:(size(Truth.Features,2)+1); Truth.Features];

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Initial information matrix, Lambda = inv(Sigma), and information vector, eta = Lambda*x
TheJournal.Lambda = spalloc(1,1,(1+size(Truth.Features,2))^2);
TheJournal.Lambda(1,1) = inv(Sigma);
TheJournal.eta = TheJournal.Lambda*x;

% A lookup table to keep track of feature id's and their corresponding location within the state vector
TheJournal.LookUP = [NaN 1; 1 1];




% The information form utilizes the inverse of Q and R in sparse form
Params.Q = sparse(Params.Q);
Params.R = sparse(Params.R);
Params.Qinv = inv(Params.Q);
Params.Rinv = inv(Params.R);

TheJournal.Feature = [];
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



% Set the figure margins
if(Params.PlotSwitch)
    Graphics.MainFigure = figure(1);
    clf;
    Graphics.axes1 = subplot(2,1,1);
    hold on;
    set(gcf,'DoubleBuffer','on','Color',[1 1 1]);
    set(gca,'Box','on');
    set(gca,'xlim',[-1 mapsize]);
    set(gca,'ylim',[-0.5 0.5]);
    temp = get(gca,'position');
    temp(end) = 0.1;
    temp(2) = 0.8;
    set(gca,'position',temp,'ytick',[],'yticklabel',[]);
    plot([-1 mapsize]',[0 0]','k:');
        
    Graphics.axes2 = subplot(2,1,2);
    set(gcf,'ColorMap',colormap(flipud(colormap('gray'))));
    temp = get(gca,'position');
    temp(4) = 0.5;
    set(gca,'position',temp);
    
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
    Graphics.Features = [];         % handle for plot of filter estimates for feature locations
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    % Plot true feature locations
    subplot(Graphics.axes1);
    plot(Truth.Features(2,:), zeros(1,size(Truth.Features,2)),'k.');
    
end;

for k=1:size(Truth.VehicleTrajectory,2)

    if(Params.PlotSwitch && ~isempty(Graphics.curLaser))
        set(Graphics.curLaser,'Visible','off');
    end;


    % Perform the motion update in the information form as the addition of the new pose
    % to the state followed by the marginalization of the past pose
    Information_DoProjection(k);


    % update the plot of the vehicle and map;
    if(Params.PlotSwitch)
        PlotState(k);
        title(['k = ' int2str(k) ' ( ' num2str(100*k/size(Truth.VehicleTrajectory,2),2) ' percent )']);
        drawnow;
    end;
    
    
    % Simulate feature observations
    % Observations is a 2xm array with one column per observation, [feature_id; x_feature - x_vehicle]
    Observations = GetFeatureObservations(k);

    if(~isempty(Observations))

        % Differentiate between features that have already been mapped and those that are new
        if(size(TheJournal.LookUP,1) > 2)
            [C,IA,IB] = intersect(Observations(1,:),TheJournal.LookUP(3:end,1));
            OldObs = Observations(:,IA);
            
            [C,IA] = setdiff(Observations(1,:),OldObs(1,:));
            NewObs = Observations(:,IA);
        else
            OldObs = [];
            NewObs = Observations;
        end;
        
        % Visualize observations
        if(Params.PlotSwitch)
            xv = GetMean(1);  % Estimate of the current vehicle pose
            [C,IA,IB] = intersect(Truth.Features(1,:),Observations(1,:));

            X = Truth.Features(2,IA);
            Y = repmat(0,1,size(Observations,2));
            
            figure(Graphics.MainFigure);
            subplot(Graphics.axes1);
            Graphics.curLaser = plot(X,Y,'yo','MarkerSize',12);
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

    
    DoLogging(k);

    if(~issparse(TheJournal.Lambda))
        fprintf(1,'k = %i : Lambda is no longer sparse \n',k);
    end;

end;




% Plot the strength of the links between the robot and map over time
figure(3);
hold on;
cmap = [colormap('hot'); colormap('jet'); colormap('gray')];
Temp = randperm(size(cmap,1));

Handles = [];
IDs = [];
for i=1:size(TheJournal.Feature,2)
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
F = speye(1);

% Velocity vector
u = Data.Velocity(2,find(Data.Velocity(1,:)==k));


%%%%%%%%%%%%%%%
% The time projection operation for the information matrix and information vector
% as viewed as the process of (1) augmenting the state to include the new
% pose and, subsequently, (2) marginalizing over the old pose. A
% description can be found in Eustice et al. (eustice05a) Section III.A.

% Step 1:   Add the new pose, xv(t+1), into the state vector
%
% Notice that the modifications to the information matrix require a small,
% bounded number of computations, proportional to the size of the robot
% pose. None of the elements corresponding to the map are affected.
LambdaOld = TheJournal.Lambda;
Lambda_11 = Params.Qinv;
Lambda_12 = -Params.Qinv*F;
Lambda_13 = zeros(1,size(TheJournal.Lambda,2)-1);
Lambda_22 = TheJournal.Lambda(1,1) + F'*Params.Qinv*F;
Lambda_23 = TheJournal.Lambda(1,2:end);
Lambda_33 = TheJournal.Lambda(2:end,2:end);


Lambda = [Lambda_11 Lambda_12 Lambda_13;...
          Lambda_12' Lambda_22 Lambda_23;...
          Lambda_13' Lambda_23' Lambda_33];



% As with Lambda, the computational cost of altering the information vector
% is minimal and does not scale with the size of the map. Of course, in the
% nonlinear case, the computational benefits are contingent on knowledge of
% the mean for xv(t).
eta_1 = Params.Qinv*u;
eta_2 = TheJournal.eta(1) - F'*Params.Qinv*u;
eta_3 = TheJournal.eta(2:end);

eta = [eta_1; eta_2; eta_3];


% Let's show how the structure of the information matrix has changed now
% that the new robot pose has been added
if(Params.verbose)
    ShowLinkStrength(Lambda,1);
    ShowLinkStrength(LambdaOld,0,Graphics.SecondFigure);
    fprintf(1,'\n\n');
    fprintf(1,'----------------------- \n\n');
    fprintf(1,'TIME PROJECTION STEP: \n\n')
    fprintf(1,'i)  The first component of the time projection step is to add \n');
    fprintf(1,'    the new robot pose x_v(t+1) into the state vector. \n\n');
    fprintf(1,'    Notice in Figure 1 that the new pose is only linked to the \n')
    fprintf(1,'    old pose while the rest of the structure remains unchanged. \n');
    fprintf(1,'    Compare this with the structure prior to state augmentation (Figure 2). \n');
    pause;
end;



% Now, we marginalize over the old pose. Referring to the marginalization
% table:
%           alpha = [xv(t+1); map];     what we want to keep
%           beta  = xv(t);              what we are marginalizing over
alpha_indices = [1 3:size(Lambda,1)];
beta_indices = 2;

Lambda_aa = Lambda(alpha_indices,alpha_indices);
Lambda_ab = Lambda(alpha_indices,beta_indices);
Lambda_bb = Lambda(beta_indices,beta_indices);

eta_a = eta(alpha_indices);
eta_b = eta(beta_indices);

TheJournal.Lambda = Lambda_aa - Lambda_ab*inv(Lambda_bb)*Lambda_ab';
TheJournal.eta = eta_a - Lambda_ab*inv(Lambda_bb)*eta_b;

% Update the schematic of the information matrix
if(Params.verbose)
    ShowLinkStrength();
    ShowLinkStrength(Lambda,1,Graphics.SecondFigure);
    fprintf(1,'\n\n');
    fprintf(1,'ii)  The final step is to marginalize over the old pose, x_v(t) \n\n');
    fprintf(1,'     By doing so, we have created links among all features that were \n');
    fprintf(1,'     linked to x_v(t). Over time, this will fill in the matrix\n\n');
    fprintf(1,'----------------------- \n\n');

    pause;
end;


% Update the timestamp in the LookUP table
TheJournal.LookUP(1,2) = k;


return;







%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Function to perform update step for information filter
function Information_DoUpdate(Observations)

global TheJournal;
global Params;
global Graphics;


k = TheJournal.LookUP(1,end);
xi = GetStateIndex(1,k);

innov = [];
G = [];
Gr = [];
Zinv = [];

% Measurement model:
%   z_x = x_f - x_v
% Note: with a linear observation model, we don't need the mean



if(Params.verbose)
    fprintf(1,'\n\n');
    fprintf(1,'+++++++++++++++++++++++ \n\n');
    fprintf(1,'MEASUREMENT UPDATE: \n\n')
    ShowLinkStrength(TheJournal.Lambda,0,Graphics.SecondFigure);
end;

% Loop over the measurement data, performing an individual update for each observation
for i=1:size(Observations,2)
    id = Observations(1,i);
    xfi = GetStateIndex(id,k);
    
         
    %%%%%%%%%%%%%%%
    % The update step for the information filter as described
    % in Eustice et al. (eustice05a) Section III.B.
    z = Observations(2,i);
    
    CT = [-1 sparse(1,size(TheJournal.Lambda,1)-1)];
    CT(:,xfi) = 1;
    
    TheJournal.Lambda = TheJournal.Lambda + CT'*Params.Rinv*CT;
    TheJournal.eta = TheJournal.eta + CT'*Params.Rinv*z;
    %%%%%%%%%%%%%%%

    
    if(Params.verbose)
        fprintf(1,'    We updated the filter based on an observation of feature %i. \n\n',id);
        fprintf(1,'    Notice how the shared information between this feature and the \n');
        fprintf(1,'    robot increases by comparing with the old matrix in Figure 2. \n\n')
        ShowLinkStrength(TheJournal.Lambda);
        pause;
    end;
end;

if(Params.verbose)
    fprintf(1,'\n');
    fprintf(1,'+++++++++++++++++++++++ \n\n');
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

% Jacobian
F = 1;


if(Params.verbose)
    fprintf(1,'\n\n');
    fprintf(1,'\n\n');
    fprintf(1,'+++++++++++++++++++++++ \n\n');
    fprintf(1,'ADDING A FEATURE: \n\n')
    ShowLinkStrength(TheJournal.Lambda,0,Graphics.SecondFigure);
end;
    
% Note that, in the linear case, we don't need the mean to add the feature
% xf = g(xv,z) + w = (mu_v + z) + w
for i=1:size(Observations,2)
    id = Observations(1,i);
    
    
    %%%%%%%%%%%%%%%
    % The operation of adding features with the information form as described
    % in Eustice et al. (eustice05a) Section III.A.
    H12 = [-F'*Params.Rinv; sparse(size(TheJournal.Lambda,1)-1,1)];
    
    TheJournal.Lambda = [TheJournal.Lambda H12; H12' Params.Rinv];
    TheJournal.Lambda(1,1) = TheJournal.Lambda(1,1) + F'*Params.Rinv*F;

    etaf_new = Params.Rinv*Observations(2,i);

    TheJournal.eta = [TheJournal.eta; etaf_new];
    TheJournal.eta(1) = TheJournal.eta(1) - F'*Params.Rinv*Observations(2,i);
    %%%%%%%%%%%%%%%
    
    % Plotting stuff
    if(Params.PlotSwitch)
        xyf = GetMean(1) + Observations(2,i);
        figure(Graphics.MainFigure);
        subplot(Graphics.axes1);
        thandle = plot(xyf(1),0,'rx','MarkerSize',12);

        Graphics.Features = [Graphics.Features [id; thandle]];
    end;
        
    % Now, update the LookUP table
    TheJournal.LookUP = [TheJournal.LookUP; id NaN*ones(1,size(TheJournal.LookUP,2)-1)];
    TheJournal.LookUP(end,end) = length(TheJournal.eta);
    
    % Keep track of the shared information between the robot and feature
    TheJournal.Feature{end+1}.id = id;
    LambdaNorm = rhomatrix(TheJournal.Lambda);
    TheJournal.Feature{end}.Link = [TheJournal.LookUP(1,2); LambdaNorm(1,end)];
    
    
    if(Params.verbose)
        ShowLinkStrength(TheJournal.Lambda);

        fprintf(1,'\n\n');
        fprintf(1,'    We have added feature %i to the map. \n\n',id);
        fprintf(1,'    Figure 1 shows the new information matrix and Figure 2 the previous version. \n');
        fprintf(1,'    A new row and column have been added to the information matrix \n');
        fprintf(1,'    and the new feature is only linked to the robot pose. \n');
        pause;
    end;
        
end;

if(Params.verbose)
    fprintf(1,'\n');
    fprintf(1,'+++++++++++++++++++++++ \n\n');
end;


return;




%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function DoLogging(k)

global TheJournal;

if(size(TheJournal.LookUP,1) == 2)
    return;
end;


Lambda = rhomatrix(TheJournal.Lambda);

for i=3:size(TheJournal.LookUP,1)
    xfi = TheJournal.LookUP(i,2);
    
    TheJournal.Feature{i-2}.Link(:,end+1) = [k; Lambda(1,xfi)];
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
if(nargin<1)
    ILinks = rhomatrix(TheJournal.Lambda);
else
    ILinks = rhomatrix(Lambda);
end;

if(nargin < 3)
    figure(Graphics.MainFigure);
    subplot(Graphics.axes2);
else
    figure(fhandle);
end;
cla;
imagesc(abs(ILinks),[0 1]);
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
text(Y,X,temp)

[I,J] = find(ILinks);
plot(I,J,'k.');
xlabel('Dots denote nonzero entries','Fontsize',10);

return;






%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Function to extract mean state estimates
%
function x = GetMean(xi)

global TheJournal;


% Right now, fully invert the information matrix to get the mean
mu = TheJournal.Lambda\TheJournal.eta;
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

xv = Truth.VehicleTrajectory(2,find(Truth.VehicleTrajectory(1,:)==k));
Temp = Truth.Features(2,:) - repmat(xv,1,size(Truth.Features,2));
I = find(abs(Temp) <= Params.MaxObsRange);
Observations = [Truth.Features(1,I); Temp(I)];
[Y,I] = sort(abs(Observations(2,:)));
Observations = Observations(:,I(1:min(length(I),Params.nObsPerIteration)));


% Generate correlated measurement noise using the Cholesky Decomposition: R = X'*X
% for which: w = U*randn(2,1) ---> cov(w) = U*I*U' = U*U' = R;  (U*U'=R=X'X --> U=X')
U = chol(Params.R)';


Observations(2,:) = Observations(2,:) + U*randn(1,size(Observations,2));


return;
        



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function PlotState(k);

global TheJournal;
global Graphics;
global Truth;

figure(Graphics.MainFigure);
subplot(Graphics.axes1);


% Brute-force estimate of the covariance matrix and mean vector
Sigma = TheJournal.Lambda\eye(size(TheJournal.Lambda));
x = Sigma*TheJournal.eta;

if(isempty(Graphics.VehiclePose))
    Graphics.VehiclePose = plot(x(1),0,'k^','MarkerFaceColor','r','MarkerEdgeColor','k','MarkerSize',8);
    set(Graphics.VehiclePose,'Color',Graphics.Color);
else
    lastvx = get(Graphics.VehiclePose,'xdata');
    set(Graphics.VehiclePose,'xdata',x(1),'ydata',0);
end;


for i=1:size(Graphics.Features,2)
    id = Graphics.Features(1,i);
    thandle = Graphics.Features(2,i);

    xfi = GetStateIndex(id,k);
    set(thandle,'xdata',x(xfi),'ydata',0);
end;



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