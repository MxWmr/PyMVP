% NO3_from_Orens.m
% orens 16/01/2014

function [NO3_adj]=NO3_from_Orens(T_cal,lambda,ENO3,ESW,I_ref,CTD_TEMP,CTD_SAL,CTD_PRES,P_SUNA,I_d,I_lambda,S,alpha,beta)

%---------------------------- Constantes ---------------------------------%

% coefficients de l'Algorithme TCSS
Coeff1=1.15002760;
Coeff2=0.02840000;
Coeff3=-0.3101349;
Coeff4=0.00122200;

% T_cal : temperature de calibration (fichier .cal)
% lambda longeur d'ondes (fichier .cal)
% ENO3 coeeficient d'extinction des nitrates (fichier .cal)
% ESW : coeeficient d'extinction de l'eau sal�e (fichier .cal)
% I_ref : Intensit� mesur�e pour un �chantillon d'eau milli-Q (fichier .cal)

% T_SUNA : temp�rature mesur�e par le SUNA (fichier NETcdf)
% S_SUNA : Salinit� mesur�e par le SUNA (fichier NETcdf)
% I_d : intensit� du DARK (fichier NETcdf)
% I_lambda : intensit� � lambda donn�e (fichier NETcdf)

 wl=209; % wavelenght offset. Par defaut 210 mais peut varier entre 206 & 2012.
% S : -0.0021 difference de pente entre profil insitu et profil SUNA
% alpha : 0.8895 gain
% beta :  0.9237 offset

NO3_algo3=[];

%------------------ Alignement CTD et SUNA -------------------------------%

 [a b]=unique(CTD_PRES);
 T_SUNA=interp1(CTD_PRES(b),CTD_TEMP(b),P_SUNA,'linear','extrap');
 S_SUNA=interp1(CTD_PRES(b),CTD_SAL(b),P_SUNA,'linear','extrap'); 
 
%----------------    Calcul de l'Absorbance   ----------------------------%

Diff_I=[I_lambda-ones(size(I_lambda,1),1)*I_d]';
%I_ref=I_ref';
A_lambda=-log10(Diff_I./[ones(size(Diff_I,1),1)*I_ref]);

%-----------------    Calcul de ESW_TIS    -------------------------------%

% calcul de ASW_Tcal
ASW_Tcal=(Coeff1+Coeff2*T_cal)*exp((Coeff3+Coeff4*T_cal).*(lambda-wl));
% calcul de ASW_Tis
for i=1:length(T_SUNA)
    for j=1:length(lambda)
        ASW_Tis(i,j)=(Coeff1+Coeff2*T_SUNA(i))*exp((Coeff3+Coeff4*T_SUNA(i))*(lambda(j)-wl));
    end
end

% Calcul de ESW sous forme matricielle
ESW_mat=[ESW*ones(1,size(ASW_Tis,1))]';
% calcul de ESW_Tis
ESW_Tis=[ESW_mat.*ASW_Tis]'./(ASW_Tcal*ones(1,size(ASW_Tis,1)));

%------------------    Calcul de A_lambda2    ----------------------------%

% calcul de la salinit� sous forme matricielle
S_SUNA_mat=[ones(size(I_lambda,1),1)*S_SUNA];
% calcul de ASW_Tis
ASW=ESW_Tis.*(S_SUNA_mat);
%calcul de A_lambda2
A_lambda2=A_lambda-ASW';

%-------------------------- optimisation ---------------------------------%
 
% variables ind�pendantes
X=[ones(size(lambda)) ENO3 lambda];
% variable d�pendanp^�te
Y=[A_lambda2'];

for i=1:length(P_SUNA)
    % MLR
    [B,BINT,R,RINT,STATS]=regress(Y(:,i),X);
    % NO3 pr�dit par le mod�le
    Y2=B(1)+B(2).*ENO3+B(3).*lambda;
    NO3_algo3=[NO3_algo3 B(2)];
end

%----------------------- Fin algorithme 3 --------------------------------%

%---------- supression Spikes  et des valeurs negatives ------------------%

for i=1:length(NO3_algo3)-2
    if ((NO3_algo3(i)-NO3_algo3(i+1))>0.4 & NO3_algo3(i+2)>NO3_algo3(i))
         NO3_algo3(i+1)=NO3_algo3(i);
    end
    if ((NO3_algo3(i)-NO3_algo3(i+1))>0.4 & NO3_algo3(i+2)<NO3_algo3(i))
         NO3_algo3(i)=NO3_algo3(i+2);
    end

    if ((NO3_algo3(i)-NO3_algo3(i+1) )<-0.4  & NO3_algo3(i+2)<NO3_algo3(i+1))
        NO3_algo3(i+1)=NO3_algo3(i);
    end
end

%---------------------- correction de profil -----------------------------%

%NO3_algo3=smooth(NO3_algo3,3)'; 

for j=1:length(P_SUNA)
    corr(j)=(S./(2.*P_SUNA(length(P_SUNA)))).*( P_SUNA(j).*P_SUNA(j));
end

NO3_algo3=NO3_algo3-corr; % correction additionnelle  en pression
NO3_algo3=(NO3_algo3-beta)./alpha; % Correction  de Johnson 

NO3_adj=NO3_algo3';
