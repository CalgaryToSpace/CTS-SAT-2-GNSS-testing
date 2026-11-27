# GNSS-testing

GNSS test code for CTS-SAT-1 and CTS-SAT-2.

## Radio occultation

`radio_occultation.py` and `radio_occultation.ipynb` hold the same code. The notebook also shows the plots. Both turn GPS carrier phase and orbit data from a satellite in low Earth orbit into profiles of the atmosphere, and compare the result with the official CDAAC (UCAR) profiles.

**Why:** As a GPS satellite sets behind the Earth, its signal bends in the atmosphere. The bending depends on air density, so it gives accurate vertical profiles anywhere on Earth. The same phase data also gives the ionosphere electron density from the L1 and L2 signals.

**Steps** (from "Standard RO Inversions in the Neutral Atmosphere 2013-2020", Sokolovskiy, CDAAC):

1. Read the phase and orbits, remove open loop navigation bits (COSMIC-1), replace untracked L2.
2. Filter the phase and get the Doppler shift caused by the atmosphere.
3. Bending angle from the ray whose Doppler matches (geometric optics), and from wave optics below the troposphere top.
4. Remove the ionosphere with L1 and L2, extrapolate it above 80 km (Zeng 2016).
5. Abel inversion to refractivity, EGM96 geoid for heights.
6. Hydrostatic pressure and dry temperature, quality control, tropopause height.
7. Electron density from L1 minus L2 (slant TEC, Abel inversion).

**Data** (CDAAC, one day per mission, in this repo):

| Folder | Content |
| --- | --- |
| `data` | COSMIC-1 `atmPhs` and `atmPrf`, 2010/001 |
| `data_cosmic2` | COSMIC-2 `conPhs` and `atmPrf`, 2024/001 |
| `data_iono` | COSMIC-1 `ionPhs` and `ionPrf`, 2010/001 |
| `data_geoid` | EGM96 geoid grid |

**Results against CDAAC** (plots in `results/`):

- Refractivity within about 0.1 % from 5 to 30 km, dry temperature within a few tenths of a K up to 20 km.
- Noisier below 5 km (worst profile about -5 %) and above 30 km (temperature up to 5-7 K off).
- Electron density peak height within 5 km for all 8 profiles.

**Not done:** error estimate (local spectral width), boundary layer and sporadic E scalars, 2020 updates 3, 4, 8, 9, 11 and 12.

**Limits:** only COSMIC data was tested, there is no CTS-SAT data yet and a reader for the CTS-SAT receiver logs still has to be written. The ionosphere part needs L1 and L2: CTS-SAT-2 (OEM7720) has both, CTS-SAT-1 (OEM719) has L1 only. One global climatology is used as first guess, so QC test 3 can fail by mistake at high latitude.

## Setup and checks

Needs [uv](https://docs.astral.sh/uv/) and optionally [just](https://github.com/casey/just).

```powershell
uv sync                         # install everything from uv.lock
uv run python radio_occultation.py   # run the retrieval, takes about 4 minutes, saves plots in results/
```

Checks that CI runs and that must pass before pushing:

```powershell
uv run ruff format --check .
uv run ruff check .
uv run pyright
uv run pytest
```

`just check` runs all four, `just fmt` fixes formatting and lint. A git pre-push hook that runs the checks can be installed with `just hook`.
