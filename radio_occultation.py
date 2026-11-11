# GPS radio occultation retrieval with geometric optics.
# Phase and orbits -> bending angle -> refractivity -> dry pressure and temperature.
# From the "Standard RO Inversions in the Neutral Atmosphere" (Sokolovskiy, UCAR/CDAAC).

# Vectors are arrays of shape (3, n), so tx[0] is x and tx[:, i] is sample i.

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import cumulative_trapezoid
from scipy.io import netcdf_file
from scipy.optimize import curve_fit
from scipy.signal import savgol_filter

F1 = 1575.42e6                      # GPS L1 frequency, Hz
F2 = 1227.60e6                      # GPS L2 frequency, Hz
KION = F2**2 / (F1**2 - F2**2)      # ionosphere free combination factor
WIN1 = 17                           # filter window in samples, about one Fresnel scale at 50 Hz
WIN2 = 3 * WIN1                     # wider window for the noisy L2
A_E = 6378137.0                     # WGS84 equatorial radius, m
E2 = 6.69437999014e-3               # WGS84 eccentricity squared
R_DRY = 287.06                      # gas constant of dry air, J/kg/K
K1 = 77.6                           # dry refractivity constant, K/hPa
GPS_UTC = 15                        # GPS minus UTC leap seconds, correct for 2009 to mid 2012
J2000 = 630763200                   # 2000-01-01 12:00 in seconds since the GPS start, 1980-01-06


def dot(a, b):
    return np.sum(a * b, axis=0)


def norm(a):
    return np.sqrt(dot(a, a))


def unit(a):
    return a / norm(a)


def cross(a, b):
    return np.cross(a, b, axis=0)


def angle(a, b):
    return np.arctan2(norm(cross(a, b)), dot(a, b))


def read_nc(path):
    f = netcdf_file(path, "r", mmap=False)
    variables = {name: f.variables[name].data.astype(float) for name in f.variables}
    attributes = dict(f._attributes)
    f.close()
    return variables, attributes


def geodetic(pos):
    # latitude (rad), longitude (rad) and height (m) above the WGS84 ellipsoid
    x, y, z = pos
    p = np.hypot(x, y)
    lat = np.arctan2(z, p * (1 - E2))
    for _ in range(6):
        n = A_E / np.sqrt(1 - E2 * np.sin(lat)**2)
        lat = np.arctan2(z + E2 * n * np.sin(lat), p)
    n = A_E / np.sqrt(1 - E2 * np.sin(lat)**2)
    h = p * np.cos(lat) + z * np.sin(lat) - n * (1 - E2 * np.sin(lat)**2)
    return lat, np.arctan2(y, x), h


def bisect(f, lo, hi):
    # bisection for many roots at once, nan where f does not change sign
    f_lo = f(lo)
    bad = np.sign(f_lo) == np.sign(f(hi))
    for _ in range(50):
        mid = 0.5 * (lo + hi)
        same = np.sign(f(mid)) == np.sign(f_lo)
        lo = np.where(same, mid, lo)
        hi = np.where(same, hi, mid)
    return np.where(bad, np.nan, 0.5 * (lo + hi))


def smooth(x, window):
    # step 9: 3rd order Savitzky-Golay filter applied 3 times
    for _ in range(3):
        x = savgol_filter(x, window, 3)
    return x


def blend(low, high, h, h1, h2):
    # smooth switch from "low" below h1 to "high" above h2
    w = np.clip((h - h1) / (h2 - h1), 0, 1)
    w = 0.5 - 0.5 * np.cos(np.pi * w)
    return np.where(w == 0, low, np.where(w == 1, high, (1 - w) * low + w * high))


def climatology():
    # US Standard Atmosphere 1976, dry, turned into bending angle by a forward Abel integral
    # returns impact height (m) and log of bending angle
    R = 6371e3
    z = np.arange(0, 150e3 + 1, 100.0)
    T = np.interp(z, [0, 11e3, 20e3, 32e3, 47e3, 51e3, 71e3, 86e3],
                  [288.15, 216.65, 216.65, 228.65, 270.65, 270.65, 214.65, 186.95])
    g = 9.80665 * (R / (R + z))**2
    P = 1013.25 * np.exp(-cumulative_trapezoid(g / (R_DRY * T), z, initial=0))
    n = 1 + 1e-6 * K1 * P / T
    x = n * (R + z)
    dlnn = np.gradient(np.log(n), x)
    alpha = np.zeros(len(x) - 1)
    for k in range(len(x) - 1):
        s = np.sqrt((x[k:] - x[k]) * (x[k:] + x[k]))
        alpha[k] = -2 * x[k] * np.trapezoid(dlnn[k:] / x[k:], s)
    return x[:-1] - R, np.log(alpha)


CLIM_H, CLIM_LOG_ALPHA = climatology()


def clim_bending(h):
    # climatology bending angle (rad) at impact height h (m)
    return np.exp(np.interp(h, CLIM_H, CLIM_LOG_ALPHA))


def limb_point(tx, rx):
    # point of the straight line TX-RX closest to the earth center
    u = unit(rx - tx)
    return tx - dot(tx, u) * u


def occultation_point(tx, rx):
    # steps 11-12: tangent point latitude, longitude (inertial) and azimuth for one TX, RX sample
    limb = limb_point(tx, rx)
    r0 = norm(limb) - geodetic(limb)[2]
    rt, rr = norm(tx), norm(rx)

    # step 4: ray through the climatology, solve alpha(p) = gt + gr + theta - pi for p
    def misfit(p):
        return np.arcsin(p / rt) + np.arcsin(p / rr) + angle(tx, rx) - np.pi - clim_bending(p - r0)

    p = bisect(misfit, r0, rr - 1)

    # step 11: rotate the RX direction toward TX by acos(p/rr) + alpha/2
    up = unit(rx)
    toward_tx = unit(tx - dot(tx, up) * up)
    phi = np.arccos(p / rr) + clim_bending(p - r0) / 2
    tangent = p * (np.cos(phi) * up + np.sin(phi) * toward_tx)
    lat, lon, _ = geodetic(tangent)

    # step 12: azimuth from north of the horizontal direction toward TX, same as CDAAC
    horizontal = unit(cross(tangent, unit(cross(tx, rx))))
    east = np.array([-np.sin(lon), np.cos(lon), 0])
    north = np.array([-np.sin(lat) * np.cos(lon), -np.sin(lat) * np.sin(lon), np.cos(lat)])
    azimuth = np.arctan2(dot(horizontal, east), dot(horizontal, north))
    return lat, lon, azimuth


def curvature(lat, lon, azimuth):
    # step 14: radius and center of local curvature of the ellipsoid along the azimuth
    s = np.sin(lat)
    n = A_E / np.sqrt(1 - E2 * s**2)                # prime vertical radius
    m = A_E * (1 - E2) / (1 - E2 * s**2)**1.5       # meridian radius
    rc = 1 / (np.cos(azimuth)**2 / m + np.sin(azimuth)**2 / n)
    up = np.array([np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), s])
    surface = np.array([n * np.cos(lat) * np.cos(lon), n * np.cos(lat) * np.sin(lon), n * (1 - E2) * s])
    return rc, surface - rc * up


def bending_angle(rate, tx, rx, vtx, vrx):
    # step 17: find the ray whose Doppler matches the measured excess phase rate (m/s)
    # Bouguer rule p = r sin(angle to vertical), with n = 1 at the satellites
    rt, rr = norm(tx), norm(rx)
    ut, ur = tx / rt, rx / rr
    ht = unit(rx - dot(rx, ut) * ut)                # horizontal at TX, toward RX
    hr = unit(tx - dot(tx, ur) * ur)                # horizontal at RX, toward TX
    u = unit(rx - tx)
    measured = dot(vrx - vtx, u) + rate

    def misfit(p):
        gt, gr = np.arcsin(p / rt), np.arcsin(p / rr)
        kt = -np.cos(gt) * ut + np.sin(gt) * ht     # ray direction leaving TX
        kr = np.cos(gr) * ur - np.sin(gr) * hr      # ray direction arriving at RX
        return dot(vrx, kr) - dot(vtx, kt) - measured

    p_line = norm(cross(tx, u))                     # impact parameter of the straight line
    p = bisect(misfit, p_line - 2e3, np.minimum(p_line + 150e3, np.minimum(rt, rr) - 1))
    return p, np.arcsin(p / rt) + np.arcsin(p / rr) + angle(tx, rx) - np.pi


def to_grid(p, alpha, grid):
    # step 18: keep samples where p keeps going down, then interpolate to the grid
    ok = np.isfinite(p)
    p, alpha = p[ok], alpha[ok]
    keep = p < np.minimum.accumulate(np.r_[np.inf, p[:-1]])
    return np.interp(grid, p[keep][::-1], alpha[keep][::-1], left=np.nan, right=np.nan)


def abel(alpha, a):
    # step 27: ln n(a) = (1/pi) * integral of alpha(x) / sqrt(x^2 - a^2) dx from a to the top
    # with z = sqrt(x^2 - a^2) the integrand becomes alpha(x) / x dz and has no singularity
    ln_n = np.zeros(len(a))
    for k in range(len(a)):
        x = a[k:]
        z = np.sqrt((x - a[k]) * (x + a[k]))
        ln_n[k] = np.trapezoid(alpha[k:] / x, z) / np.pi
    return np.exp(ln_n)


def retrieve(path):
    v, attributes = read_nc(path)
    t = v["time"]
    tx = np.array([v["xGps"], v["yGps"], v["zGps"]]) * 1e3      # km to m
    rx = np.array([v["xLeo"], v["yLeo"], v["zLeo"]]) * 1e3
    vtx = np.array([v["xdGps"], v["ydGps"], v["zdGps"]]) * 1e3
    vrx = np.array([v["xdLeo"], v["ydLeo"], v["zdLeo"]]) * 1e3
    L1, L2 = v["exL1"], v["exL2"]                               # excess phase, m

    # step 5: keep the closed loop part, open loop samples have a receiver model
    # step 7: stop at the first fill value too
    good = (v["xmdl"] == -999) & (L1 != -999) & (L2 != -999)
    n = np.argmin(good) if not good.all() else len(t)

    # skip what this simple version cannot do
    # the straight line TX-RX going up means a rising occultation
    line_radius = norm(limb_point(tx, rx))
    excess = L1 - L1[0]
    if line_radius[-1] > line_radius[0] or n < 10 * WIN2 or excess.max() < 500:
        print("  skipped: rising, or too little closed loop data")
        return None

    # step 13: occultation point is where L1 excess phase reaches 500 m
    # open loop samples are still used here, this step only needs the geometry
    i = np.argmax(excess >= 500)
    lat, lon, azimuth = occultation_point(tx[:, i], rx[:, i])
    rc, center = curvature(lat, lon, azimuth)

    # step 16: inertial to earth fixed longitude, using Greenwich sidereal time
    days = (attributes["startTime"] + t[i] - GPS_UTC - J2000) / 86400
    gmst = np.radians(280.46061837 + 360.98564736629 * days)
    lon = (np.degrees(lon - gmst) + 180) % 360 - 180

    # step 15: positions relative to the curvature center
    tx, rx = tx[:, :n] - center[:, None], rx[:, :n] - center[:, None]
    vtx, vrx, t, L1, L2 = vtx[:, :n], vrx[:, :n], t[:n], L1[:n], L2[:n]

    # steps 9 and 17: filtered phase rates and their bending angles
    p1, a1 = bending_angle(np.gradient(smooth(L1, WIN1), t), tx, rx, vtx, vrx)
    p1w, a1w = bending_angle(np.gradient(smooth(L1, WIN2), t), tx, rx, vtx, vrx)
    p2w, a2w = bending_angle(np.gradient(smooth(L2, WIN2), t), tx, rx, vtx, vrx)

    # step 21: impact height grid, 20 m step up to 150 km
    h = np.arange(np.ceil((np.nanmin(p1) - rc) / 20) * 20, 150e3 + 1, 20.0)
    a = h + rc
    bend1 = to_grid(p1, a1, a)
    bend1w = to_grid(p1w, a1w, a)
    correction = KION * (bend1w - to_grid(p2w, a2w, a))

    # step 24: ionospheric correction, held at its 20-25 km mean below 20 km because L2 is noisy there
    hold = np.nanmean(correction[(h >= 20e3) & (h <= 25e3)])
    bend3 = bend1 + blend(hold, correction, h, 20e3, 21e3)

    # step 26: narrow window below 30 km, wide window above 40 km to beat down noise
    bend3 = blend(bend3, bend1w + correction, h, 30e3, 40e3)

    # step 26: fit c * guess^b to the data at 35-60 km, data below 35 km, fitted guess above 60 km
    guess = clim_bending(h)
    fit = (h >= 35e3) & (h <= 60e3) & np.isfinite(bend3)
    (c, b), _ = curve_fit(lambda x, c, b: c * x**b, guess[fit], bend3[fit], p0=[1.0, 1.0])
    bend = blend(bend3, c * guess**b, h, 35e3, 60e3)

    # step 27: refractivity and height of each level
    n_air = abel(bend, a)
    N = (n_air - 1) * 1e6
    z = a / n_air - rc

    # step 29: hydrostatic dry pressure, integrated from the top, then dry temperature
    rho = 100 * N / (K1 * R_DRY)
    g0 = 9.780327 * (1 + 0.0053024 * np.sin(lat)**2 - 0.0000058 * np.sin(2 * lat)**2)
    g = g0 * (6371e3 / (6371e3 + z))**2
    P = -cumulative_trapezoid((rho * g)[::-1], z[::-1], initial=0)[::-1] / 100      # hPa
    keep = z < 100e3                                        # N rounds to 0 near the 150 km top
    z, bend, N, P = z[keep], bend[keep], N[keep], P[keep]
    T = K1 * P / N

    return {"lat": np.degrees(lat), "lon": lon, "azimuth": np.degrees(azimuth), "rc": rc,
            "z": z, "bend": bend, "N": N, "P": P, "T": T}


def main():
    folder = Path(__file__).parent / "data"
    fig, ax = plt.subplots(1, 3, figsize=(14, 6), sharey=True)

    for path in sorted(folder.glob("atmPhs_*")):
        name = path.name[7:30]
        print("\n" + name)
        r = retrieve(path)
        cdaac_path = folder / path.name.replace("atmPhs", "atmPrf")
        if r is None or not cdaac_path.exists():
            continue

        c, at = read_nc(cdaac_path)
        print(f"  ours  lat {r['lat']:8.3f} lon {r['lon']:8.3f} azim {r['azimuth']:8.2f} rc {r['rc'] / 1e3:9.3f} km")
        print(f"  CDAAC lat {at['lat']:8.3f} lon {at['lon']:8.3f} azim {at['azim']:8.2f} rc {at['rfict']:9.3f} km")

        # CDAAC profile on our heights
        ok = (c["Ref"] > 0) & (c["MSL_alt"] > 0)
        z_c = c["MSL_alt"][ok][::-1] * 1e3
        N_c = np.interp(r["z"], z_c, c["Ref"][ok][::-1], left=np.nan, right=np.nan)
        T_c = np.interp(r["z"], z_c, c["Temp"][ok][::-1] + 273.15, left=np.nan, right=np.nan)
        dN = 100 * (r["N"] - N_c) / N_c
        dT = r["T"] - T_c

        for z1, z2 in [(5, 10), (10, 20), (20, 30), (30, 40)]:
            band = (r["z"] >= z1 * 1e3) & (r["z"] < z2 * 1e3)
            print(f"  {z1:2d}-{z2:2d} km  N diff {np.nanmedian(dN[band]):+6.2f} %   T diff {np.nanmedian(dT[band]):+6.2f} K")

        low = r["z"] < 40e3
        z_km = r["z"][low] / 1e3
        ax[0].semilogx(r["bend"][low], z_km, label=name[13:])
        ax[1].plot(dN[low], z_km)
        ax[2].plot(dT[low], z_km)

    ax[0].set_xlabel("bending angle (rad)")
    ax[0].set_ylabel("height (km)")
    ax[0].legend()
    ax[1].set_xlabel("N ours minus CDAAC (%)")
    ax[1].set_xlim(-3, 3)
    ax[2].set_xlabel("T ours minus CDAAC (K)")
    ax[2].set_xlim(-5, 5)
    for x in ax:
        x.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(Path(__file__).parent / "results.png", dpi=120)
    print("\nsaved results.png")


if __name__ == "__main__":
    main()
