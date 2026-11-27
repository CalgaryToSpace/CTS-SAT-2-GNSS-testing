import numpy as np

import radio_occultation as ro


def test_vector_helpers():
    a = np.array([[3.0], [4.0], [0.0]])
    assert ro.norm(a)[0] == 5.0
    assert np.allclose(ro.norm(ro.unit(a)), 1.0)
    assert np.isclose(ro.angle(a, np.array([[0.0], [1.0], [0.0]]))[0], np.arctan2(3, 4))


def test_geoid_height_is_reasonable():
    # EGM96 lat/lon in degrees: Indian Ocean low is about -100 m, near Iceland about +65 m
    assert -110 < ro.geoid_height(5.0, 78.0) < -85
    assert 50 < ro.geoid_height(64.0, 340.0) < 80


def test_geodetic_height_on_equator():
    pos = np.array([[ro.A_E + 500e3], [0.0], [0.0]])
    lat, lon, h = ro.geodetic(pos)
    assert np.isclose(h[0], 500e3, atol=1.0)
    assert np.isclose(lat[0], 0.0)
    assert np.isclose(lon[0], 0.0)


def test_cosmic2_profile_matches_cdaac():
    # one COSMIC-2 occultation, refractivity and temperature must be close to the CDAAC profile
    path = next((ro.FOLDER / "data_cosmic2").glob("conPhs_*G11*"))
    r = ro.retrieve(path)
    assert r is not None
    c, _ = ro.read_nc(
        ro.FOLDER / "data_cosmic2" / path.name.replace("conPhs", "atmPrf")
    )
    ok = (c["Ref"] > 0) & (c["MSL_alt"] > 0)
    z_c = c["MSL_alt"][ok][::-1] * 1e3
    n_c = np.interp(r["z"], z_c, c["Ref"][ok][::-1])
    t_c = np.interp(r["z"], z_c, c["Temp"][ok][::-1] + 273.15)
    band = (r["z"] >= 10e3) & (r["z"] < 25e3)
    assert np.nanmedian(np.abs(100 * (r["N"] - n_c) / n_c)[band]) < 0.5
    assert np.nanmedian(np.abs(r["T"] - t_c)[band]) < 2.0
