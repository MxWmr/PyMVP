"""
NO3_HSorens_RTQC - Calcul de la concentration en nitrate à partir des spectres d'absorption SUNA
 
History:
    Développé par Orens de Fommervault, adapté par Antoine Poteau,
    puis adapté en juin 2015 pour RT_QC Nitrate par Catherine Schmechtig.
    Traduit en Python depuis R.
"""
 
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from sklearn.linear_model import LinearRegression
 
 
def NO3_HSorens_RTQC(
    cal_file,
    profNumber,
    TEMP,
    PSAL,
    PRES,
    SUNA_SPECTRUM,
    SunaDarkSpectrumMean,
    NO3,
    NO3_climato=0,
    N=None,
    Tcal=None,
    wl=208.5,
    deep_delta=None,
    lag=0,
    drift=0,
):
    """
    Calcule la concentration en nitrate (et HS) à partir des spectres d'absorption SUNA.
 
    Paramètres
    ----------
    cal_file          : str   - Chemin vers le fichier de calibration SUNA
    profNumber        : int   - Numéro de profil (pour correction de dérive)
    TEMP              : array - Températures (dbar)
    PSAL              : array - Salinités
    PRES              : array - Pressions (dbar)
    SUNA_SPECTRUM     : array 2D (Ndepth x Nlambda) - Spectres SUNA bruts
    SunaDarkSpectrumMean : array - Spectre sombre moyen par profondeur
    NO3               : array - Nitrate brut (99999 = manquant)
    NO3_climato       : float - Valeur climatologique du nitrate en profondeur
    N                 : int   - Nombre de longueurs d'onde (None = 217–280 nm)
    Tcal              : float - Température de calibration (lue dans cal_file si None)
    wl                : float - Longueur d'onde de référence
    deep_delta        : float - Correction en profondeur
    lag               : float - Décalage vertical CTD / SUNA (dbar)
    drift             : float - Coefficient de dérive par profil
 
    Retourne
    --------
    (no3_orens, hs_orens) : tuple de deux arrays numpy
    """
 
    # --- Coefficients Sakamoto et al. 2009 ---
    AAA = 1.1500276
    BBB = 0.02840
    CCC = -0.3101349
    DDD = 0.001222
 
    # --- Lecture du fichier de calibration SUNA ---
    # On saute les lignes commençant par 'H'
    cal_rows = []
    tcal_line = None
    with open(cal_file, "r") as f_cal:
        for line in f_cal:
            stripped = line.strip()
            if stripped.startswith("H"):
                if "T_CAL" in stripped:
                    tcal_line = stripped
                continue
            if stripped == "":
                continue
            cal_rows.append(stripped.split(","))
 
    cal = pd.DataFrame(np.array(cal_rows)[:,1:], dtype=float)
    # cal = cal.iloc[:, 1:]
    cal.columns = ["lambda", "ENO3", "ESW", "EHS", "Iref"]
 
    # --- Lecture de Tcal si non fourni ---
    if Tcal is None:
        if tcal_line is not None:
            Tcal = float(tcal_line[12:22].strip())
        else:
            raise ValueError("Tcal introuvable dans le fichier de calibration.")
 
    # --- Sélection des longueurs d'onde ---
    mask_217 = cal["lambda"] >= 217
    if N is None:
        mask_range = mask_217 & (cal["lambda"] <= 280)
        N = int(mask_range.sum())
 
    istart = int(np.where(mask_217.values)[0][0])
 
    lambda_ = cal["lambda"].values[istart : istart + N]
    Iref    = cal["Iref"].values[istart : istart + N]
    ENO3    = cal["ENO3"].values[istart : istart + N]
    ESW     = cal["ESW"].values[istart : istart + N]
    EHS     = cal["EHS"].values[istart : istart + N]
 
    Ndepth = SUNA_SPECTRUM.shape[0]
    no3_TCSS = np.full(Ndepth, np.nan)
    hs_TCSS  = np.full(Ndepth, np.nan)
 
    # --- Correction du décalage vertical CTD / SUNA ---
    def safe_interp(x, y, xnew):
        """Interpolation avec extrapolation constante (rule=2 en R)."""
        f_interp = interp1d(x, y, bounds_error=False, fill_value=(y[0], y[-1]))
        return f_interp(xnew)
 
    TEMPsuna = safe_interp(PRES, TEMP, PRES + lag)
    SALsuna  = safe_interp(PRES, PSAL, PRES + lag)
 
    # --- Boucle sur les profondeurs ---
    for p in range(Ndepth):
        spectrum_slice = SUNA_SPECTRUM[p, :N]

        if np.sum(~np.isnan(spectrum_slice)) == N:
            temp = TEMPsuna[p]
            sal  = SALsuna[p]
            pres = PRES[p]
            I    = spectrum_slice
            Idark = SunaDarkSpectrumMean[p]

 
            if I[0] != 99999:
                # Calcul de l'absorbance
                A = -np.log10((I - Idark) / Iref)
 
                # Correction en température
                ASWTcal = (AAA + BBB * Tcal)  * np.exp((CCC + DDD * Tcal)  * (lambda_ - wl))
                ASWTis  = (AAA + BBB * temp)   * np.exp((CCC + DDD * temp)  * (lambda_ - wl))
                ESWTis  = (ESW * ASWTis) / ASWTcal
                ASW     = ESWTis * sal
 
                # Correction pression (1% par 1000 m)
                ASW = ASW * (1 - (0.015 * pres) / 1000)
 
                Aprim = A - ASW
 
                # Régression linéaire : Aprim ~ ENO3 + lambda + EHS
                try:
                    X = np.column_stack([ENO3, lambda_, EHS])
                    # Avec constante (intercept)
                    reg = LinearRegression(fit_intercept=True)
                    reg.fit(X, Aprim)
                    # coef[0]=ENO3, coef[1]=lambda, coef[2]=EHS
                    no3_TCSS[p] = reg.coef_[0]
                    hs_TCSS[p]  = reg.coef_[2]
                except Exception:
                    pass  # Laisse NaN si la régression échoue
 
    no3_orens = no3_TCSS.copy()
    hs_orens  = hs_TCSS.copy()
 
    # --- Correction au fond ---
    ind_no3   = np.where(NO3 != 99999.0)[0]
    PRES_NO3  = PRES[ind_no3]
    NO3_NO3   = no3_orens[ind_no3]
    idx_1000  = np.argmin(np.abs(PRES_NO3 - 1000))
    NO3_deep  = NO3_NO3[idx_1000]
 
    if NO3_climato!=0:
        deep_delta = NO3_deep - NO3_climato
 
    deep_delta = 0  # Forcé à 0 
 
    no3_orens = no3_orens - deep_delta
 
    # --- Correction de dérive ---
    no3_orens = no3_orens - (drift * profNumber)
 
    # --- Masquage des valeurs manquantes ---
    mask_missing = np.ones(Ndepth, dtype=bool)
    mask_missing[ind_no3] = False
    no3_orens[mask_missing] = 999.99
    hs_orens[mask_missing]  = 999.99
 
    return no3_orens, hs_orens
