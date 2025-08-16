import math
from typing import Tuple, Optional
import numpy as np
import pandas as pd
from scipy.stats import norm

class Implied_asset_value:
    def __init__(self,
                 sigma: float = 4.6940,
                 weight: float = 0.3466,
                 trading_days_per_year: int = 250,
                 tol: float = 1e-8,
                 max_iter: int = 200):
        self.sigma = float(sigma)
        self.weight = float(weight)
        self.trading_days = int(trading_days_per_year)
        self.tol = float(tol)
        self.max_iter = int(max_iter)
    
    def comp_L(self,
               short_term_debt: float,
               long_term_debt: float,
               other_liability: float) -> float:
            return float(short_term_debt) + 0.5 * float(long_term_debt) + self.weight * float(other_liability)
    
    def comp_E(self,
               V: float,
               L: float,
               rf: float) -> float:
        sigma = self.sigma
        T = 1
        d_plus = (math.log(V / L) + (rf + 0.5 * self.sigma ** 2)) / (sigma * T)
        d_minus = (math.log(V / L) + (rf - 0.5 * self.sigma ** 2)) / (sigma * T)
        E = V * norm.cdf(d_plus) - math.exp(-rf * T) * L * norm.cdf(d_minus)
        return E
    
    def comp_AV(self,
                market_cap: float,
                short_term_debt: float,
                long_term_debt: float,
                other_liability: float,
                rf: float,
                V_guss: float = 2.0,
                V_upperlimit: float = 100.0)-> Tuple[Optional[float], bool, dict]:
        
        L = self.comp_L(short_term_debt, long_term_debt, other_liability)
        if L <= 0:
            return None, False, {"error": "Non-positive L computed", "L": L}

        rf_annual = float(rf) * self.trading_days

        def f(V):
            return self.comp_E(V, L, rf_annual) - float(market_cap)
        
        V_low = 1e-8
        V_high = max(market_cap * V_guss, L * 1.1, 1.0)
        f_low = f(V_low)
        f_high = f(V_high)

        iter_expansion = 0
        while f_low * f_high > 0 and iter_expansion < 60 and V_high < L * V_upperlimit:
            V_high *= 2.0
            f_high = f(V_high)
            iter_expansion += 1

        info = {
            "L": L,
            "r_annual": rf_annual,
            "V_low_initial": V_low,
            "V_high_initial": V_high,
            "f_low": f_low,
            "f_high": f_high,
            "expansions": iter_expansion
        }

        if f_low * f_high > 0:
            info["error"] = "Failed to bracket root"
            return None, False, info
        
        a, b = V_low, V_high
        fa, fb = f_low, f_high
        mid = None
        converged = False
        for i in range(self.max_iter):
            mid = 0.5 * (a + b)
            fm = f(mid)
            if abs(fm) < self.tol:
                converged = True
                break
            if fa * fm <= 0:
                b, fb = mid, fm
            else:
                a, fa = mid, fm
            if (b - a) / max(1.0, b) < self.tol:
                converged = True
                break

        if mid is None:
            info["error"] = "No midpoint computed"
            return None, False, info

        V_final = mid
        final_residual = f(V_final)
        info.update({
            "V_final": V_final,
            "final_residual": final_residual,
            "iterations": i + 1,
            "a": a,
            "b": b
        })

        ok = converged and (V_final > 0) and (abs(final_residual) <= max(1e-8, self.tol * max(1.0, abs(market_cap))))
        return (V_final if ok else V_final), ok, info

    def comp_DTD(self, V: float, L: float, T = 1) -> float:
        if V <= 0 or L <= 0:
            return float("nan")
        return math.log(V / L) / (self.sigma * math.sqrt(T))
    
    def comp_row(self, row:dict) -> dict:
        m_cap = float(row['Market Capitalization'])
        s_d = float(row['Short Term Debt'])
        l_d = float(row['Long Term Debt'])
        o_l = float(row['Other Liability'])
        t_a = float(row['Total Asset'])
        rf_daily = float(row['Daily Risk-Free Rate'])

        V, ok, info = self.comp_AV(market_cap=m_cap, short_term_debt=s_d, long_term_debt=l_d, other_liability=o_l, rf=rf_daily)
        L = self.comp_L(short_term_debt=s_d, long_term_debt=l_d, other_liability=o_l)
        DTD = self.comp_DTD(V, L) if V is not None else float("nan")

        output = {
            "Liability": L,
            "Implied Asset Value": V,
            "DTD": DTD,
            "Validation": "Valid" if bool(ok) else "Not Valid",
        }
        
        return output
    
    def comp_df(self, df: pd.DataFrame, colnames: dict = None) -> pd.DataFrame:
        if colnames is None:
            colnames = {}
        def get(col):
            return colnames.get(col, col)
        
        results = []
        for _, r in df.iterrows():
            rowdict = {
                "Market Capitalization": r[get("Market Capitalization")],
                "Short Term Debt": r[get("Short Term Debt")],
                "Long Term Debt": r[get("Long Term Debt")],
                "Other Liability": r[get("Other Liability")],
                "Total Asset": r[get("Total Asset")],
                "Daily Risk-Free Rate": r[get("Daily Risk-Free Rate")]
            }
            res = self.comp_row(rowdict)
            results.append(res)

        res_df = pd.DataFrame(results)
        output = pd.concat([df.reset_index(drop=True), res_df], axis=1)
        return output
