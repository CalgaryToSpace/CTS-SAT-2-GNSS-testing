# GNSS-testing

## Radio occultation

**What it does:** `radio_occultation.py` turns GPS signals received by a satellite in low Earth orbit into vertical profiles of atmospheric refractivity, pressure and temperature, and into ionosphere electron density. It uses COSMIC-1 and COSMIC-2 data (folders `data`, `data_cosmic2`, `data_iono` and `data_geoid`) and checks its results against the official CDAAC profiles.

**Why:** As a GPS satellite sets behind the Earth, its signal passes through deeper and deeper layers of the atmosphere and bends. The amount of bending depends on air density, so measuring it tells us about the atmosphere. This gives accurate, high-resolution profiles anywhere on Earth, without balloons or ground stations.

**How:**

1. Read the signal phase and satellite orbits.
2. Filter the phase and take its rate of change to get the Doppler shift caused by the atmosphere.
3. Find the ray path that matches that Doppler shift, which gives the bending angle at each height. Low in the troposphere, wave optics is used instead.
4. Combine the L1 and L2 frequencies to remove the effect of the ionosphere.
5. Above 35 km, gradually replace the noisy data with a fitted US Standard Atmosphere model.
6. Apply an Abel inversion to convert bending angle into refractivity.
7. Get pressure from hydrostatic balance, then dry temperature from refractivity and pressure.

The L1 minus L2 phase also gives the ionosphere electron density (slant TEC, then an Abel inversion).

**Accuracy:** For the 4 COSMIC-1 and 8 COSMIC-2 test profiles, the median difference from CDAAC is within about 0.1% for refractivity (5-30 km) and within a few tenths of a K for temperature (5-20 km). Below 5 km the worst profile is about 5% off, and above 30 km temperature can be 5-7 K off. The electron density peak height is within 5 km for all 8 ionosphere profiles. This is a small sample from a single day per mission.

Method: "Standard RO Inversions in the Neutral Atmosphere" (Sokolovskiy, UCAR/CDAAC).
