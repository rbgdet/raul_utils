import numpy as np
import healpy as hp
from astropy.io import fits

def plot_radec_mollweide(ra_deg, dec_deg, nside=512, weights=None,
                         overlay_points=False, s=None, xsize=1600, title='RA/Dec density (HEALPix)'):
    """
    Parameters
    ----------
    ra_deg, dec_deg : array-like
        Right ascension [deg, 0..360] and declination [deg, -90..90].
    nside : int
        HEALPix NSIDE (power of 2). e.g. 256, 512, 1024.
    weights : array-like or None
        If provided, accumulates weighted sum per pixel instead of counts.
        For a mean map, pass weights and set 'normalize=True' below.
    overlay_points : bool
        If True, draw the input points on top.
    title : str
        Title for the Mollweide map.
    """

    # Convert to HEALPix angles
    ra_deg  = np.asarray(ra_deg, dtype=float)
    dec_deg = np.asarray(dec_deg, dtype=float)
    theta = np.radians(90.0 - dec_deg)       # colatitude
    phi   = np.radians(ra_deg % 360.0)       # longitude

    npix = hp.nside2npix(nside)
    pix  = hp.ang2pix(nside, theta, phi, nest=False)

    # Bin: counts or weighted sum
    if weights is None:
        counts = np.bincount(pix, minlength=npix)
        m = counts.astype(float)
        unit = "counts per pixel"
    else:
        weights = np.asarray(weights, dtype=float)
        m = np.bincount(pix, weights=weights, minlength=npix).astype(float)
        unit = "weighted sum per pixel"

    # Mask empty pixels for nicer plotting
    m_masked = np.full(npix, hp.UNSEEN, dtype=float)
    nonzero = m > 0
    m_masked[nonzero] = m[nonzero]

    # Plot HEALPix map in Mollweide
    hp.mollview(m_masked, title=title, unit=unit, coord='C', norm='hist',
                xsize=xsize, cbar=True, notext=False)
    hp.graticule()

    # Optionally overlay the input points
    if overlay_points:
        if s is None:
            s = 0.5 * (xsize / np.sqrt(len(ra_deg)))**2  # heuristic point size based on number of points and image size
        # healpy takes lon/lat in degrees if lonlat=True
        hp.projscatter(ra_deg, dec_deg, lonlat=True, s=s, alpha=0.6)