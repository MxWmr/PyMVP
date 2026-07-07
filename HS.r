########################################################################################################################################################
## fonction qui calcule la concentration en nitrate ?|  partir des spectres d'absorptions

## données en entrées
##---------------------
# cal_file <- fichier de calibration du SUNA
# dat_file <- fichier de données
# N <- nombre de longueur d'onde pour l'algorithme (on pars toujousr de 217 (ou la première longueur d'ondd
e juste apres. Si N est NULL on prend toutes les longueurs d'onde entre 217 et 240 nm. 
#
#
# History
# This routine was developped by orens de Fommervault and adapt by Antoine poteau
# then adapt in june 2015 to RT_QC Nitrate by Catherine Schmechtig
#
#
# Attention on a adapte le Deep-delta
#
#################################################################################################################################################################################################



NO3_HSorens_RTQC <- function(cal_file, profNumber,TEMP, PSAL, PRES, SUNA_SPECTRUM, SunaDarkSpectrumMean,NO3, NO3_climato, N=NULL, Tcal=NULL, plot=FALSE, wl, deep_delta, lag, f,drift){

# COEFF SAKAMOTO ET AL.2009
AAA <- 1.1500276
BBB <- 0.02840
CCC <- -0.3101349
DDD <- 0.001222

## lecture du  fichier calibration SUNA
cal <- read.table(cal_file, sep=",", comment.char="H")
cal <- cal[,-1]
names(cal) <- c( "lambda", "ESW", "ENO3", "EHS",  "Iref")

# read TCAL
if(is.null(Tcal)){
ccc <- scan(file=cal_file,what="character", sep="\n")
it <- grep(ccc, pattern="T_CAL")
Tcal <- as.numeric(substr(ccc[it], 13, 22))

}

if(is.null(N)){
N <- length(which((217 <= cal$lambda) &(cal$lambda <= 280)))
}

istart <- which(cal$lambda >= 217)[1]
lambda <- cal$lambda[istart:(istart+N-1)]
Iref <- cal$Iref[istart:(istart+N-1)]
ENO3 <- cal$ENO3[istart:(istart+N-1)]
ESW <- cal$ESW[istart:(istart+N-1)]
EHS <- cal$EHS[istart:(istart+N-1)]

sstart <- 1
send <- N
Ndepth <- dim(SUNA_SPECTRUM)[1]
no3_TCSS <- rep(NA, Ndepth)
hs_TCSS <- rep(NA,Ndepth)

## algoritme TCSS (temperature corrected salinity variable) modified by Orens

# CORRECTION DU DECALAGE VERTICAL CTD & SUNA
TEMPsuna <- approx(PRES, TEMP, PRES+lag, rule=2)$y
SALsuna <- approx(PRES, PSAL, PRES+lag, rule=2)$y

for(p in 1:Ndepth){
if(length(which(!is.na(SUNA_SPECTRUM[p, 1:N]))) == N){
temp <- TEMPsuna[p]
sal <- SALsuna[p]
pres <- PRES[p]
I <- SUNA_SPECTRUM[p, 1:N]
Idark <- SunaDarkSpectrumMean[p]


if(I[1]!=99999){


# calcul de l'absorbance

A <- as.vector(as.matrix(-log10((I - Idark)/Iref)))
ASWTcal <- (AAA + BBB*Tcal)*exp((CCC+DDD*Tcal)*(lambda-wl))
ASWTis <- (AAA + BBB*temp)*exp((CCC+DDD*temp)*(lambda-wl))
ESWTis <- (ESW*ASWTis)/ASWTcal


ASW <- ESWTis*sal


# CORRECTION PRESSION (1% par 1000m ORENS + KEN)
ASW <- ASW * (1-(0.015*pres) / 1000 )


Aprim <- A - ASW

# NO3/ HS
lm3 <- NA
lm3$coefficients <- NA
lm3$coefficients[2] <- NA
lm3$coefficients[4] <- NA
try(lm3 <- lm(Aprim~ENO3+lambda+EHS, na.action=na.omit))
#try(lm3 <- lm(Aprim~ENO3+lambda, na.action=na.omit))
no3_TCSS[p] <- lm3$coefficients[2]
hs_TCSS[p] <- lm3$coefficients[4]
#hs_TCSS[p] <- lm3$coefficients[2]
}
}
}
# fin algo TCSS modifie 
no3_orens <- no3_TCSS
hs_orens <- hs_TCSS

## correction au fond 
ind_no3=which(NO3!=99999.)
PRES_NO3=PRES[ind_no3]
NO3_NO3=no3_orens[ind_no3]
NO3_deep=NO3_NO3[which.min(abs(PRES_NO3-1000))]

if(!is.na(NO3_climato)){
deep_delta=NO3_deep-NO3_climato
}
deep_delta=0
#print("deep_delta") 
#print(deep_delta)

no3_orens <- no3_orens - deep_delta

### drift

no3_orens <-   no3_orens - (drift * profNumber)


no3_orens[!ind_no3]=999.99
hs_orens[!ind_no3]=999.99

return(list(no3_orens,hs_orens))
}

