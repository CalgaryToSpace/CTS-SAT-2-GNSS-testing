# GNSS-testing

## Radio occultation

**What it does:** `radio_occultation.py` turns GPS signals received by a satellite in low Earth orbit into vertical profiles of atmospheric refractivity, pressure and temperature. It uses COSMIC-1 data and checks its results against the official CDAAC profiles.

**Why:** As a GPS satellite sets behind the Earth, its signal passes through deeper and deeper layers of the atmosphere and bends. The amount of bending depends on air density, so measuring it tells us about the atmosphere. This gives accurate, high-resolution profiles anywhere on Earth, without balloons or ground stations.

**How:**

1. Read the signal phase and satellite orbits.
2. Filter the phase and take its rate of change to get the Doppler shift caused by the atmosphere.
3. Find the ray path that matches that Doppler shift, which gives the bending angle at each height.
4. Combine the L1 and L2 frequencies to remove the effect of the ionosphere.
5. Above 35 km, gradually replace the noisy data with a fitted US Standard Atmosphere model.
6. Apply an Abel inversion to convert bending angle into refractivity.
7. Get pressure from hydrostatic balance, then dry temperature from refractivity and pressure.

**Accuracy:** For the 4 test profiles, the median difference from CDAAC is within about 0.35% for refractivity (5-30 km) and within 1 K for temperature (5-20 km). This is a small sample from a single day.

**Limitations:** This is a simplified version of the CDAAC method. It uses closed-loop data from setting occultations only, so profiles stop near 8 km. Temperature is less reliable above about 25 km, and heights are above the ellipsoid rather than sea level.

Method: "Standard RO Inversions in the Neutral Atmosphere" (Sokolovskiy, UCAR/CDAAC).