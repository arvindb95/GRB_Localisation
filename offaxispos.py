#!/usr/bin/env python2.7

"""
    offaxispos.py

    Aim: Find the position of an off-axis transient

    **** NOTE: not yet a generalized code ****
    Needs the following files to be present in the same folder:
        nonoise_grb_AS1P01_003T01_9000000002cztM0_level2.events
        czti_Aepix.out (compiled from czti_Aepix.f and czti_pixarea.f)

    Algorithm steps: 
      Make a grid of responses around Ra0, Dec0
      Fit data = const + mult * resp, find chi^2
      See which position gives least chi^2

    Version  : $Rev: 1186 $
    Last Update: $Date: 2016-07-18 11:13:39 +0530 (Mon, 18 Jul 2016) $

"""

# v1.0  : First code added to SVN, works only for GRB151006A
# v1.1  : Updated to work with new .out files containing detx,dety. Actually works!
# v2.0  : Major upgrade: use healpy to get well spaced grid, read configuration from files, make pdf plots, etc
# v2.1  : Added fancy plots
# v2.2  : Added the "noloc" keyword for making plots without doing localisation calculations
# v2.3  : Changes in plotting order, handling zero count cases correctly
# v2.4  : Made it easier for external codes to write a config file
version = "2.4"

import subprocess, os, shutil
import ConfigParser, argparse
from astropy.table import Table
from astropy.io import fits
from astropy.stats import sigma_clip, sigma_clipped_stats
from scipy.optimize import curve_fit
import astropy.coordinates as coo
import astropy.units as u
import glob
import numpy as np
from scipy.integrate import simps
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import random
from scipy.optimize import curve_fit

parser = argparse.ArgumentParser(epilog="""

    Version  : $Rev: 1186 $
    Last Update: $Date: 2016-07-18 11:13:39 +0530 (Mon, 18 Jul 2016) $

""")
parser.add_argument("configfile", nargs="?", help="Name of configuration file", type=str)
parser.add_argument('--noloc', dest='noloc', action='store_true')
parser.set_defaults(noloc=False)


#------------------------------------------------------------------------
# Define functions required for processing

def plot_map(data, pixlist, nside, runconf, title, unit):
    """
    Use healpy to make a plot and add coordinate grid
    """
    hpmap = np.repeat(np.nan, hp.pixelfunc.nside2npix(nside))
    hpmap[pixlist] = data
    plot_lon_cen, plot_lat_cen = runconf['ra'], runconf['dec']
    if plot_lon_cen > 180:
        plot_lon_cen -= 360
    plot_bounds = (1.0 + 1.1 * runconf['radius']) * np.array([-1.0, 1.0])
    
    hp.cartview(hpmap, rot=(plot_lon_cen, plot_lat_cen, 0), 
            lonra=plot_bounds, latra=plot_bounds,
            notext=True, unit=unit, title=title, 
            min=min(data), max=max(data), coord='C', flip='astro')

    dec0 = np.round(runconf['dec'])
    dec_spacing = runconf['radius']/2.0
    decs = dec0 + np.arange(-2*runconf['radius'], 2.1*runconf['radius'], dec_spacing)
    decs_min, decs_max = min(decs), max(decs)

    ra0 = np.round(runconf['ra'])
    cosdelt =np.cos(np.deg2rad(dec0)) 
    ra_spacing = runconf['radius']/cosdelt / 2.0
    ras = ra0 + np.arange(-2.0*runconf['radius']/cosdelt, 2.1*runconf['radius']/cosdelt, ra_spacing)
    ras_min, ras_max = min(ras), max(ras)
    #num_ras = np.ceil(1.0 * runconf['radius'] / grid_spacing / np.cos(np.deg2rad(min(decs))) )

    line_dec = np.linspace(decs_min, decs_max, 100)
    line_ra = np.linspace(ras_min, ras_max, 100)
    for ra in ras:
        hp.projplot(np.repeat(ra, 100), line_dec, lonlat=True, ls='dashed', color='black')
        hp.projtext(ra, dec0, r"{ra:0.1f}$^\circ$".format(ra=ra), lonlat=True, clip_on=True)
    for dec in decs:
        hp.projplot(line_ra, np.repeat(dec, 100), lonlat=True, ls='dashed', color='black')
        hp.projtext(ra0, dec, r"{dec:0.1f}$^\circ$".format(dec=dec), lonlat=True, clip_on=True, rotation=90)
    return

def rd2txy(ra0, dec0, twist, ra_tran, dec_tran):
    """
    Call Dipankar's radec2txty fortran code to get the theta_x, theta_y of any
    point on the sky.
    In tests, the run time was 7 ms/call
    """
    tcalc = subprocess.Popen([runconf['txycode']],  stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    out, err = tcalc.communicate("{ra0} {dec0}\n{twist}\n{ra_tran} {dec_tran}".format(ra0=ra0, dec0=dec0, twist=twist, ra_tran=ra_tran, dec_tran=dec_tran))
    junk, tx, ty = out.rsplit(None, 2)
    return float(tx), float(ty)

def get_twist(mkffile, trigtime, time_range = 100.0):
    """
    Open the MKF file, return roll angle to trigtime and return
    To safegaurd against data jumps, I use a median value from '+-time_range'
    around the trigger time
    """
    mkfdata = fits.getdata(mkffile, 1)
    sel = abs(mkfdata['time'] - trigtime) < 100
    return 0.0 - np.median(mkfdata['roll_rot'][sel])

def txy(mkffile, trigtime, ra_tran, dec_tran):
    """
    Calculate thetax, thetay using astropy
    Use pitch, roll and yaw information from the MKF file
    """
    # x = -yaw
    # y = +pitch
    # z = +roll

    # Read in the MKF file
    mkfdata = fits.getdata(mkffile, 1)
    sel = abs(mkfdata['time'] - trigtime) < 100

    # Get pitch, roll, yaw
    # yaw is minus x
    pitch = coo.SkyCoord( np.median(mkfdata['pitch_ra'][sel]) * u.deg, np.median(mkfdata['pitch_dec'][sel]) * u.deg )
    roll = coo.SkyCoord( np.median(mkfdata['roll_ra'][sel]) * u.deg, np.median(mkfdata['roll_dec'][sel]) * u.deg )
    yaw_ra  = (180.0 + np.median(mkfdata['yaw_ra'][sel]) ) % 360
    yaw_dec = -np.median(mkfdata['yaw_dec'][sel])
    minus_yaw = coo.SkyCoord( yaw_ra * u.deg, yaw_dec * u.deg )

    # Transient:
    transient = coo.SkyCoord(ra_tran * u.deg, dec_tran * u.deg)

    # Angles from x, y, z axes are:
    ax = minus_yaw.separation(transient)
    ay = pitch.separation(transient)
    az = roll.separation(transient)

    # the components are:
    cx = np.cos(ax.radian) # The .radian is not really needed, but anyway...
    cy = np.cos(ay.radian)
    cz = np.cos(az.radian)

    # Thetax = angle from z axis in ZX plane
    # lets use arctan2(ycoord, xcoord) for this
    thetax = u.rad * np.arctan2(cx, cz)
    thetay = u.rad * np.arctan2(cy, cz)

    return thetax.to(u.deg).value, thetay.to(u.deg).value, minus_yaw, pitch, roll, transient

def plot_xyzt(grbdir,ax, x, y, z, t):
    """
    Make a subplot that shows X, Y, Z axes and a transient vector
    The input coordinates are astropy.coordinate.SkyCoord objects
    """
    global runconf
	
    colors = ['blue', 'gray', 'red', 'black']
    names = ['X', 'Y', 'Z', grbdir]
    zdirs = ['x', 'y', 'z', None]

    mkffile = runconf['mkffile']
    trigtime = runconf['trigtime']
    ra_tran = runconf['ra']
    dec_tran = runconf['dec']
    mkfdata = fits.getdata(mkffile, 1)
    window = 10
    sel = abs(mkfdata['time'] - trigtime) < window	
    
    earthx = -np.median(mkfdata['posx'][sel])
    earthy = -np.median(mkfdata['posy'][sel]) 
    earthz = -np.median(mkfdata['posz'][sel]) 
    
    earth_vec_mag = np.sqrt(earthx**2 + earthy**2 + earthz**2)
    
    earth = coo.SkyCoord(earthx, earthy, earthz, frame='icrs', representation='cartesian')
    			
    ax.set_xlim(-1.2,1.2)
    ax.set_ylim(-1.2,1.2)
    ax.set_zlim(-1.2,1.2)

    for count, dirn in enumerate([x, y, z, t]):
        xx, yy, zz = dirn.cartesian.x.value, dirn.cartesian.y.value, dirn.cartesian.z.value
        ax.quiver(0, 0, 0, xx, yy, zz, color=colors[count])
        ax.text(xx, yy, zz, names[count], zdirs[count])
	
    ax.quiver(0,0,0,earthx/earth_vec_mag,earthy/earth_vec_mag,earthz/earth_vec_mag,color='green') 
    ax.text(earthx/earth_vec_mag,earthy/earth_vec_mag,earthz/earth_vec_mag,'Earth')
    
    #ax.set_xlabel("RA = 0")
    #ax.set_zlabel("Pole")
    return

def add_satellite(ax, coo_x, coo_y, coo_z):
    """
    Add a basic version of the satellite outline to the plots
    """
    from mpl_toolkits.mplot3d import Axes3D
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    tr = np.transpose(np.vstack((coo_x.cartesian.xyz.value, coo_y.cartesian.xyz.value, coo_z.cartesian.xyz.value)))

    alpha_czti = 0.5
    alpha_radiator = 0.5
    alpha_sat = 0.3

    color_czti = 'yellow'
    color_radiator = 'black'
    color_sat = 'green'

    c_w2 = 0.15 # czti half-width
    c_h  = 0.30 # czti height
    c_hr = 0.40 # czti radiator height
    sat_w = 0.6

    # For each surface, do the following:
    # verts = []
    # verts.append([tuple(tr.dot(np.array[cx, cy, cz]))])
    # surf = Poly3DCollection(verts)
    # surf.set_alpha()
    # surf.set_color()
    # ax.add_collection3d(surf)
    
    # +x rect
    verts = []
    verts.append(tuple(tr.dot(np.array([c_w2, c_w2, 0]))))
    verts.append(tuple(tr.dot(np.array([c_w2, c_w2, c_h]))))
    verts.append(tuple(tr.dot(np.array([c_w2, -c_w2, c_h]))))
    verts.append(tuple(tr.dot(np.array([c_w2, -c_w2, 0]))))
    surf = Poly3DCollection([verts])
    surf.set_alpha(alpha_czti)
    surf.set_color(color_czti)
    ax.add_collection3d(surf)
    
    # +y rect
    verts = []
    verts.append(tuple(tr.dot(np.array([c_w2, c_w2, 0]))))
    verts.append(tuple(tr.dot(np.array([c_w2, c_w2, c_h]))))
    verts.append(tuple(tr.dot(np.array([-c_w2, c_w2, c_h]))))
    verts.append(tuple(tr.dot(np.array([-c_w2, c_w2, 0]))))
    surf = Poly3DCollection([verts])
    surf.set_alpha(alpha_czti)
    surf.set_color(color_czti)
    ax.add_collection3d(surf)

    # -y rect
    verts = []
    verts.append(tuple(tr.dot(np.array([c_w2, -c_w2, 0]))))
    verts.append(tuple(tr.dot(np.array([c_w2, -c_w2, c_h]))))
    verts.append(tuple(tr.dot(np.array([-c_w2, -c_w2, c_h]))))
    verts.append(tuple(tr.dot(np.array([-c_w2, -c_w2, 0]))))
    surf = Poly3DCollection([verts])
    surf.set_alpha(alpha_czti)
    surf.set_color(color_czti)
    ax.add_collection3d(surf)
    
    # -x radiator plate
    verts = []
    verts.append(tuple(tr.dot(np.array([-c_w2, c_w2, 0]))))
    verts.append(tuple(tr.dot(np.array([-c_w2, c_w2, c_hr]))))
    verts.append(tuple(tr.dot(np.array([-c_w2, -c_w2, c_hr]))))
    verts.append(tuple(tr.dot(np.array([-c_w2, -c_w2, 0]))))
    surf = Poly3DCollection([verts])
    surf.set_alpha(alpha_radiator)
    surf.set_color(color_radiator)
    ax.add_collection3d(surf)

    # # Bottom CZTI only
    # verts = []
    # verts.append(tuple(tr.dot(np.array([c_w2, c_w2, 0]))))
    # verts.append(tuple(tr.dot(np.array([-c_w2, c_w2, 0]))))
    # verts.append(tuple(tr.dot(np.array([-c_w2, -c_w2, 0]))))
    # verts.append(tuple(tr.dot(np.array([c_w2, -c_w2, 0]))))
    # surf = Poly3DCollection([verts])
    # surf.set_alpha(alpha_czti)
    # surf.set_color(color_czti)
    # ax.add_collection3d(surf)

    # Satellite top
    verts = []
    verts.append(tuple(tr.dot(np.array([sat_w-c_w2, sat_w-c_w2, 0]))))
    verts.append(tuple(tr.dot(np.array([-c_w2, sat_w-c_w2, 0]))))
    verts.append(tuple(tr.dot(np.array([-c_w2, -c_w2, 0]))))
    verts.append(tuple(tr.dot(np.array([sat_w-c_w2, -c_w2, 0]))))
    surf = Poly3DCollection([verts])
    surf.set_alpha(alpha_sat)
    surf.set_color(color_sat)
    ax.add_collection3d(surf)

    # Satellite bottom
    verts = []
    verts.append(tuple(tr.dot(np.array([sat_w-c_w2, sat_w-c_w2, -sat_w]))))
    verts.append(tuple(tr.dot(np.array([-c_w2, sat_w-c_w2, -sat_w]))))
    verts.append(tuple(tr.dot(np.array([-c_w2, -c_w2, -sat_w]))))
    verts.append(tuple(tr.dot(np.array([sat_w-c_w2, -c_w2, -sat_w]))))
    surf = Poly3DCollection([verts])
    surf.set_alpha(alpha_sat)
    surf.set_color(color_sat)

    ax.add_collection3d(surf)

    # Satellite back (radiator side)
    verts = []
    verts.append(tuple(tr.dot(np.array([-c_w2, sat_w-c_w2, 0]))))
    verts.append(tuple(tr.dot(np.array([-c_w2, sat_w-c_w2, -sat_w]))))
    verts.append(tuple(tr.dot(np.array([-c_w2, -c_w2, -sat_w]))))
    verts.append(tuple(tr.dot(np.array([-c_w2, -c_w2, 0]))))
    surf = Poly3DCollection([verts])
    surf.set_alpha(alpha_sat)
    surf.set_color(color_sat)
    ax.add_collection3d(surf)

    # Satellite front (opposite radiator side)
    verts = []
    verts.append(tuple(tr.dot(np.array([sat_w-c_w2, sat_w-c_w2, 0]))))
    verts.append(tuple(tr.dot(np.array([sat_w-c_w2, sat_w-c_w2, -sat_w]))))
    verts.append(tuple(tr.dot(np.array([sat_w-c_w2, -c_w2, -sat_w]))))
    verts.append(tuple(tr.dot(np.array([sat_w-c_w2, -c_w2, 0]))))
    surf = Poly3DCollection([verts])
    surf.set_alpha(alpha_sat)
    surf.set_color(color_sat)
    ax.add_collection3d(surf)

    #dpix_mask Satellite right (-y, common to czti)
    verts = []
    verts.append(tuple(tr.dot(np.array([sat_w-c_w2, -c_w2, 0]))))
    verts.append(tuple(tr.dot(np.array([sat_w-c_w2, -c_w2, -sat_w]))))
    verts.append(tuple(tr.dot(np.array([-c_w2, -c_w2, -sat_w]))))
    verts.append(tuple(tr.dot(np.array([-c_w2, -c_w2, 0]))))
    surf = Poly3DCollection([verts])
    surf.set_alpha(alpha_sat)
    surf.set_color(color_sat)
    ax.add_collection3d(surf)

    # Satellite left (+y)
    verts = []
    verts.append(tuple(tr.dot(np.array([sat_w-c_w2, sat_w-c_w2, 0]))))
    verts.append(tuple(tr.dot(np.array([sat_w-c_w2, sat_w-c_w2, -sat_w]))))
    verts.append(tuple(tr.dot(np.array([-c_w2, sat_w-c_w2, -sat_w]))))
    verts.append(tuple(tr.dot(np.array([-c_w2, sat_w-c_w2, 0]))))
    surf = Poly3DCollection([verts])
    surf.set_alpha(alpha_sat)
    surf.set_color(color_sat)
    ax.add_collection3d(surf)

    return


def visualize_3d(grbdir,x, y, z, t, thetax, thetay, name):
    """
    Make a plot that allows us to visualize the transient location in 3d
    Use matplotlib Axes3D
    Uses the helper function plot_xyzt
    """
    # Set ax.azim and ax.elev to ra, dec
    global runconf

    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure()
    plt.suptitle(r"Visualisation of {name} in 3d:$\theta_x$={tx:0.1f},$\theta_y$={ty:0.1f}".format(name=name, tx=thetax, ty=thetay))
    # Z
    ax = plt.subplot(2, 2, 1, projection='3d')
    plot_xyzt(grbdir,ax, x, y, z, t)
    ax.azim = z.ra.deg
    ax.elev = z.dec.deg
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])
    add_satellite(ax, x, y, z)
    ax.set_title("View from CZTI pointing (z)")

    # Transient
    ax = plt.subplot(2, 2, 2, projection='3d')
    plot_xyzt(grbdir,ax, x, y, z, t)
    ax.azim = t.ra.deg
    ax.elev = t.dec.deg
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])
    add_satellite(ax, x, y, z)
    ax.set_title("View from nominal \n transient direction")

    # X
    ax = plt.subplot(2, 2, 3, projection='3d')
    plot_xyzt(grbdir,ax, x, y, z, t)
    ax.azim = x.ra.deg
    ax.elev = x.dec.deg
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])
    add_satellite(ax, x, y, z)
    ax.set_title("View from CZTI X axis")

    # Z
    ax = plt.subplot(2, 2, 4, projection='3d')
    plot_xyzt(grbdir,ax, x, y, z, t)
    ax.azim = y.ra.deg
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])
    add_satellite(ax, x, y, z)
    ax.set_title("View from CZTI Y axis")

    return

def get_default_configuration():
    """
    Return dicts with the default values for config files
    """
    # Pre-configured default values for various parameters:
    default_config = {
            "name":"Transient",
            "auto":True,
            "ra":0.0,
            "dec":0.0,
            "radius":10.0,
            "resolution":1.8,
            "energy":70.0,
            "pixsize": 16,
            "respcode":"czti_Aepix.out",
            "txycode":"radec2txty.out",
            "resppath":"pixarea",
            "plotfile":"plots/localize.pdf",
	    "lc_bin":5.0,
	    "typ":"band",
	    "comp_bin":20,	
            "verbose":True,
            "do_fit":True
            }
    required_config = {
            'l2file':"_level2.evt",
            'infile':"file.evt",
            'mkffile':"file.mkf",
            'trigtime':0.00,
            'transtart':0.00,
            'tranend':0.00,
            'bkg1start':0.00,
            'bkg1end':0.00,
            'bkg2start':0.00,
            'bkg2end':0.00,
	    'alpha':0.00,
	    'beta':0.00,
	    'E0':0.00,
	    'A':0.00
            }
    return default_config, required_config

def write_configuration(creator, creator_version, default_values, required_values, filename=None):
    """
    Print default config file on screen, or optionally save it to a file
    """
    default_config, required_config = get_default_configuration()
    default_config.update(default_values)
    required_config.update(required_values)
    printdict = {"prog":creator, "version": creator_version}
    printdict.update(default_config)
    printdict.update(required_config)
    configuration = """
# Sample configuration file for {prog} version {version}
# SVN revision $Rev: 1186 $
# Last updated $Date: 2016-07-18 11:13:39 +0530 (Mon, 18 Jul 2016) $
# Comment lines starting with # are ignored
# Do not add extra spaces!
#
#
[Config]
#------------------------------------------------------------------------
# Event-specific configuration
#
# Event name (for plots etc)
name:{name}
#
# Level2 file for reprocessing
l2file:{l2file}
# Input file for analysis
infile:{infile}
#
# MKF file for getting rotation
mkffile:{mkffile}
#
# Energy (keV)
energy:{energy}
# Trigger time in czti seconds
# Seconds since 1/1/2010, 00:00:00 UTC
trigtime:{trigtime}
#
# Start and end time for data to use for localisation
transtart:{transtart}
tranend:{tranend}
#
# Define two windows for estimating background to subtract for localisation
bkg1start:{bkg1start}
bkg1end:{bkg1end}
bkg2start:{bkg2start}
bkg2end:{bkg2end}
# 
# Transient location in decimal degrees
# Set auto:True if search is to be centered on spacecraft pointing
auto:{auto}
ra:{ra:0.2f}   ; Ignored if auto=True
dec:{dec:0.2f}  ; Ignored if auto=True
# 
# Transient search radius (degrees) and approximate resolution (degrees)
radius:{radius:0.2f}
resolution:{resolution:0.2f}
# actual resolution is the nearest healpix resolution available
# Some of the supported values are 7.33, 3.66, 1.83, 0.92, 0.46, 0.23, 0.11
# ANY value is allowed, the closest supported value will actually be used
#
#------------------------------------------------------------------------
# Generic configuration parameters
#
# Grouping pixels: group n x n pixels into a "superpixel"
# May be 1, 2, 4, 8, 16
pixsize:{pixsize}
#
# Use fitting for calculating resposnse, or just scale?
# If True, best-fit "flux" is calculated for image = background + flux * source
# If False, "flux" is simply (np.sum(source_image) - np.sum(bkg_image)) / np.sum(response)
do_fit:True
#
# Codes for calculating responses, and theta_x, theta_y from ra,dec
# Must be executable
respcode:{respcode}
txycode:{txycode}
#
# Location of response files
resppath:{resppath}
#
# Output plot pdf path
plotfile:{plotfile}
# 
# Give verbose output: True / False
verbose:{verbose}
""".format(**printdict)
    if filename is not None:
        with open(filename, 'w') as thefile:
            thefile.write(configuration)
    else:
        print configuration



def get_configuration(args):
    """
    Read the configuration file specified.
    Exit with error if any required parameter is missing.

    Dump default configuration file if no file specified on command line.
    """

    #------------------------------------------------------------------------
    # If configfile is not specified, dump a configfile on screen
    default_config, required_config = get_default_configuration()
    if args.configfile is None:
        write_configuration(parser.prog, version, default_config, required_config)
        raise SystemExit

    #------------------------------------------------------------------------
    # If you are here, then a configfile was given. Try to parse it
    userconf = ConfigParser.SafeConfigParser()
    userconf.read(args.configfile)
    runconf = {}

    if userconf.has_option('Config', 'verbose'): 
        runconf['verbose'] = userconf.getboolean('Config', 'verbose')
    else:
        runconf['verbose'] = default_config['verbose']

    if runconf['verbose']: print "Configuration for this run: "
    for key in required_config.keys():
        try:
            runconf[key] = userconf.getfloat('Config', key)
            if runconf['verbose']: print "  > {key} = {val}".format(key=key, val=runconf[key])
        except ConfigParser.NoOptionError:
            if runconf['verbose']: print "\nError: Required parameter {key} missing from config file!!".format(key=key)
            if runconf['verbose']: print "Update the file {configfile} and try again\n".format(configfile=args.configfile)
            raise SystemExit
        except ValueError:
            runconf[key] = userconf.get('Config', key)
            if runconf['verbose']: print "  > {key} = {val}".format(key=key, val=runconf[key])

    for key in default_config.keys():
        if key == 'verbose': continue
        try:
            runconf[key] = userconf.getfloat('Config', key)
        except ConfigParser.NoOptionError:
            runconf[key] = default_config[key]
            if runconf['verbose']: print "Using default value for {key}".format(key=key)
        except ValueError:
            runconf[key] = userconf.get('Config', key)
        if runconf['verbose']: print "  > {key} = {val}".format(key=key, val=runconf[key])

    # Now convert the bool values: verbose is already processed
    boolkeys = ['auto', 'do_fit']
    for b_key in boolkeys:
        #print b_key, runconf[b_key]
        test_string = "{b_key}".format(b_key = runconf[b_key])
        #print test_string
        if (test_string[0].upper() == 'T') or (test_string[0] == '1'):
            runconf[b_key] = True
        else:
            runconf[b_key] = False
        #print b_key, runconf[b_key]

    # Configuaration finally parsed!
    # At some future time, I should validate it...
    return runconf

def clean_mask(image, sig=5, iters=3):
    """
    Mask out input image by removing 5-sigma outliers and zero pixels.
    Return cleaned image and mask.
    """

    mean, median, stddev = sigma_clipped_stats(image[image>0], sigma=sig, iters=iters)

    mask_bad = (np.abs(image - median) > sig * stddev) | (image == 0)
    image_ret = np.copy(image)
    image_ret[mask_bad] = 0
    
    return image_ret, mask_bad

def resample(image, pixsize):
    """
    Take a 128 x 128 pixel image, and rebin it such that
    new pixels = pixsize x pixsize old pixels
    """
    assert pixsize in [1, 2, 4, 8, 16] # return error and exit otherwise
    imsize = 128/pixsize
    newimage = np.zeros((imsize, imsize))
    for xn, x in enumerate(np.arange(0, 128, pixsize)):
        for yn, y in enumerate(np.arange(0, 128, pixsize)):
            newimage[xn, yn] = np.nansum(image[x:x+pixsize, y:y+pixsize]) # Nansum is important as sum of masked array can be nan
    return newimage

def f2image(infile, maskit=True):
    """
    Convert the ".out" file from Dipankar's code into an image
    """
    global master_mask

    tab = Table.read(infile, format="ascii", names=("quadrant", "detx", "dety", "area"), comment="#")
    pixel_edges = np.arange(-0.5, 63.6)
    image = np.zeros((128,128))
    im_sub = np.zeros((64,64))

    data = tab[tab["quadrant"] == 0]
    im_sub[data['detx'], data['dety']] = data['area']
    im_sub = np.transpose(im_sub)
    image[64:128,0:64] = np.copy(im_sub)

    data = tab[tab["quadrant"] == 1]
    im_sub[data['detx'], data['dety']] = data['area']
    im_sub = np.transpose(im_sub)
    image[64:128,64:128] = np.copy(im_sub)

    data = tab[tab["quadrant"] == 2]
    im_sub[data['detx'], data['dety']] = data['area']
    im_sub = np.transpose(im_sub)
    image[0:64,64:128] = np.copy(im_sub)

    data = tab[tab["quadrant"] == 3]
    im_sub[data['detx'], data['dety']] = data['area']
    im_sub = np.transpose(im_sub)
    image[0:64,0:64] = np.copy(im_sub)

    if maskit:
        image[master_mask] = 0

    return image

def calc_resp(energy, theta_x, theta_y):
    """
    Call Dipankar's czti_Aepix fortran code to get the 
    detector response for a given angle. Note that the code 
    has been modified to give filenames in this format. If an
    output file of that name already exists, it is not created 
    again by this subroutine.
    """
    global runconf
    respfile_name = "pixarea_{energy:0d}_{theta_x:0d}_{theta_y:0d}.out".format(energy=int(energy), theta_x=int(theta_x), theta_y=int(theta_y))
    respfile_full = "{resppath}/{respname}".format(resppath=runconf['resppath'], respname=respfile_name)
    if not os.path.exists(respfile_full):
        respmake = subprocess.Popen([runconf['respcode']], stdin=subprocess.PIPE, stdout=subprocess.PIPE)
        respmake.communicate("{energy} {theta_x} {theta_y}".format(energy=int(energy), theta_x=int(theta_x), theta_y=int(theta_y)))
        shutil.move(respfile_name, respfile_full)
    else:
        # The fotran output file already exists
        pass
    return respfile_full

def plot_resp(energy, theta_x, theta_y):
    """
    Save a plot of the response calculated from Dipankar's codes
    """
    global runconf
    respfile_f = calc_resp(energy, theta_x, theta_y)
    response = f2image(respfile_f)
    plotfile = "{outdir}/pixarea_{energy:0d}_{theta_x:0d}_{theta_y:0d}.png".format(outdir=runconf['outdir'], 
            energy=int(energy), theta_x=int(theta_x), theta_y=int(theta_y))
    plt.clf()
    plt.imshow(response, origin="lower")
    plt.title(r"Energy: {energy:0d} keV, $\theta_x$ = {theta_x:0d}, $\theta_y$ = {theta_y:0d}".format(energy=int(energy), theta_x=int(theta_x), theta_y=int(theta_y)))
    plt.colorbar()
    plt.savefig(plotfile)
    return

def inject(energy, theta_x, theta_y, flux=1, noise=0):
    """
    Create a fake image, using response from a certain angle
    Add poission noise
    """
    respfile_f = calc_resp(energy, theta_x, theta_y)
    image = f2image(respfile_f) * flux + np.random.poisson(noise, (128, 128))
    return image

def evt2image(infile, tstart, tend):
        """
    Read an events file (with or without energy), and return a combined
    DPH for all 4 quadrants. 
    If tstart and tend are given, use data only in that time frame. The 
    default values are set to exceed expected bounds for Astrosat.
    """
	hdu = fits.open(infile)
	pixel_edges = np.arange(-0.5, 63.6)

	e_low = 100 # Energy cut lower bound 
	e_high = 150 # Energy cut upper bound

	data = hdu[1].data[np.where( (hdu[1].data['Time'] >= tstart) & (hdu[1].data['Time'] <= tend) )]
	sel = (data['energy'] > e_low) & (data['energy'] < e_high)
	data = data[sel]
	im1 = np.histogram2d(data['detx'], data['dety'], bins=(pixel_edges, pixel_edges))
	im1 = np.transpose(im1[0])
	data = hdu[2].data[np.where( (hdu[2].data['Time'] >= tstart) & (hdu[2].data['Time'] <= tend) )]
	sel = (data['energy'] > e_low) & (data['energy'] < e_high)
	data = data[sel]
	im2 = np.histogram2d(data['detx'], data['dety'], bins=(pixel_edges, pixel_edges))
	im2 = np.transpose(im2[0])
	data = hdu[3].data[np.where( (hdu[3].data['Time'] >= tstart) & (hdu[3].data['Time'] <= tend) )]
	sel = (data['energy'] > e_low) & (data['energy'] < e_high)
	data = data[sel]
	im3 = np.histogram2d(data['detx'], data['dety'], bins=(pixel_edges, pixel_edges))
	im3 = np.transpose(im3[0])
	data = hdu[4].data[np.where( (hdu[4].data['Time'] >= tstart) & (hdu[4].data['Time'] <= tend) )]
	sel = (data['energy'] > e_low) & (data['energy'] < e_high)
	data = data[sel]
	im4 = np.histogram2d(data['detx'], data['dety'], bins=(pixel_edges, pixel_edges))
	im4 = np.transpose(im4[0])

	image = np.zeros((128,128))
	image[0:64,0:64] = im4
	image[0:64,64:128] = im3
	image[64:128,0:64] = im1
	image[64:128,64:128] = im2

	image = np.flip(image,0)

	#plt.imshow(image, origin="lower")
	return image


def fitbkg(x, flux):
    global bkg_image
    return bkg_image.flatten() + abs(flux) * x

#---------------Plotter shifted from bottom ---------------------------


def old_plotter():
    #------------------------------------------------------------------------

    # Begin calculations

    # Source image:
    #runconf['infile'] = "test_dx30_dy60.evt"
    src_image = evt2image(runconf['infile'], runconf['transtart'], runconf['tranend'])   # t_trig+100 to +200 - note that peak is trig+150
    crude_source_image, mask_src = clean_mask(src_image)
    if runconf['verbose']: print "Source photons    : {sp:0d}".format(sp=int(np.sum(crude_source_image)))

    # Background image
    bkg_image1 = evt2image(runconf['infile'], runconf['bkg1start'], runconf['bkg1end'])   # Example: -100 to -400
    bkg_image2 = evt2image(runconf['infile'], runconf['bkg2start'], runconf['bkg2end'])   # Example: +500 to +800
    bkg_image0 = 1.0 * (bkg_image1 + bkg_image2) * (runconf['tranend'] - runconf['transtart']) / (runconf['bkg2end'] - runconf['bkg2start'] + runconf['bkg1end'] - runconf['bkg1start'])
    crude_bkg_image, mask_bkg = clean_mask(bkg_image0)
    if runconf['verbose']: print "Background photons (scaled by time): {bp:0d}".format(bp=int(np.sum(crude_bkg_image)))


    # now apply same mask to source and background image, then resample them
    # Mask is True for outliers
    master_mask = mask_src | mask_bkg
    crude_source_image[master_mask] = 0
    source_image = resample(crude_source_image, runconf['pixsize'])
    crude_bkg_image[master_mask] = 0
    bkg_image = resample(crude_bkg_image, runconf['pixsize'])

    # Plot source, background and S-B images
    if runconf['verbose']: print "Making source / background plots"
    plotfile = PdfPages(runconf['plotfile'])
    plt.figure()

    # source
    plt.clf()
    plt.imshow(source_image, interpolation='nearest', origin='lower')
    col = plt.colorbar()
    col.set_label("Counts")
    plt.title("Image for source: T={transtart:0.2f} to {tranend:0.2f}".format(**runconf))
    plt.suptitle("{name} - Data file: {infile}".format(**runconf))
    plotfile.savefig()

    # background
    plt.clf()
    plt.imshow(bkg_image, interpolation='nearest', origin='lower')
    col = plt.colorbar()
    col.set_label("Counts")
    plt.title("Image for scaled background: T={bkg1start:0.2f} to {bkg1end:0.2f},\nand T={bkg2start:0.2f} to {bkg2end:0.2f}".format(**runconf))
    #plt.suptitle("{name} - Data file: {infile}".format(**runconf))
    plotfile.savefig()
    plt.clf()

    # background-subtracted
    plt.imshow(source_image - bkg_image, interpolation='nearest', origin='lower')
    col = plt.colorbar()
    col.set_label("Counts")
    plt.title("Background-subtracted source image".format(**runconf))
    plt.suptitle("{name} - Data file: {infile}".format(**runconf))
    plotfile.savefig()
    plt.clf()

    if runconf['verbose']: print "Source / background plots complete"
    # First set of plots done. 
    
    # ------------------------------------------------------------------------
    # Now make RA Dec grid
    # First find out the pointing:
    data_header = fits.getheader(runconf['infile'], 0)
    ra_pnt, dec_pnt = data_header['ra_pnt'], data_header['dec_pnt']

    # Now see if ra/dec were "auto", in which case edit them
    if runconf['auto'] == True:
        runconf['ra'] = ra_pnt
        runconf['dec'] = dec_pnt

    # Option A:
    # # Now get the roll angle
    twist = get_twist(runconf['mkffile'], runconf['trigtime'])

    # # Calculate tx, ty for nominal transient position
    # transient_thetax, transient_thetay = rd2txy(ra_pnt, dec_pnt, twist, runconf['ra'], runconf['dec'])

    # Option B: calculate thetax, thetay directly
    transient_thetax, transient_thetay, coo_x, coo_y, coo_z, coo_transient = txy(runconf['mkffile'], runconf['trigtime'], runconf['ra'], runconf['dec'])

    if runconf['verbose']: 
        print "RA and Dec for satellite pointing are {ra:0.2f}, {dec:0.2f}".format(ra=ra_pnt, dec=dec_pnt)
        print "Twist angle is {twist:0.2f}".format(twist=twist)
        print "RA and Dec for nominal transient location are {ra:0.2f}, {dec:0.2f}".format(ra=runconf['ra'], dec=runconf['dec'])
        print "Theta_x and theta_y for nominal transient location are {tx:0.2f}, {ty:0.2f}".format(tx=transient_thetax, ty=transient_thetay)


    #------------------------------------------------------------------------
    # Plot the theoretical response at nominal direction
    try:
        respfile_f = calc_resp(runconf['energy'], transient_thetax, transient_thetay)
        response = resample(f2image(respfile_f), runconf['pixsize']) # this also applies master_mask
        plt.imshow(response, interpolation='nearest', origin='lower')
        col = plt.colorbar()
        col.set_label("Amplitude")
        plt.title(r"Calculated response at nominal $\theta_x$={tx:0.1f},$\theta_y$={ty:0.1f}".format(tx=transient_thetax, ty=transient_thetay))
        plt.suptitle("{name} - Data file: {infile}".format(**runconf))
        plotfile.savefig()
        plt.clf()
    except:
        # In case of any error, don't plot
        pass


    #------------------------------------------------------------------------
    # 3D visualisation of the satellite and transient
    visualize_3d(coo_x, coo_y, coo_z, coo_transient, transient_thetax, transient_thetay, runconf["name"])
    plotfile.savefig()
    plt.clf()

    #------------------------------------------------------------------------
    # Actual calculation on thetax thetay grid
    if not args.noloc:
        log2_nside = np.round(np.log2(1.02332670795 / np.deg2rad(runconf['resolution'])))
        nside = int(2**log2_nside)
        # hp.ang2vec takes theta, phi in radians
        # theta goes from 0 at NP to pi at SP: hence np.deg2rad(90-dec)
        # phi is simply RA
        trans_vec = hp.ang2vec(np.deg2rad(90.0 - runconf['dec']), np.deg2rad(runconf['ra']))
        pixlist = hp.query_disc(nside, trans_vec, np.deg2rad(runconf['radius']), inclusive=True)
        num_pix = len(pixlist)
        thetas, phis = hp.pix2ang(nside, pixlist)
        dec_calc = 90.0 - np.rad2deg(thetas)
        ra_calc = np.rad2deg(phis)
        if runconf['verbose']: print "RA-dec grid ready, with {num_pix} points".format(num_pix=num_pix)

        thetas_x, thetas_y = np.zeros(len(pixlist)), np.zeros(len(pixlist))
        for count in range(num_pix):
            thetas_x[count], thetas_y[count], junk1, junk2, junk3, junk4 = txy(runconf['mkffile'], runconf['trigtime'], ra_calc[count], dec_calc[count])

        redchi = np.zeros( num_pix )
        fluxes = np.zeros( num_pix )
        resp_strength = np.zeros( num_pix )

        if runconf['verbose']: print " # /Tot   T_x   T_y   |      chisq       redchi |    Flux   Bkgrnd"
        #tot = len(thetas_x) * len(thetas_y)
        for count in range(num_pix):
            theta_x, theta_y = thetas_x[count], thetas_y[count]
            # Calculate response:
            respfile_f = calc_resp(runconf['energy'], theta_x, theta_y)
            # Now read the response file and create an image
            response = resample(f2image(respfile_f), runconf['pixsize'])
            resp_strength[count] = np.sum(response)
            sf = source_image.flatten()
            rf = response.flatten()
            if runconf['do_fit']:
                dof = 128*128/runconf['pixsize']/runconf['pixsize'] - 2
                out = curve_fit(fitbkg, rf, sf) # curve_fit(function, x, y)
                fitvals = out[0]
                mf = fitbkg(rf, *fitvals)
                deltasq = (sf - mf)**2 / ((sf + 1e-8)**2)
            else:
                # out = flux * resp + bkg
                dof = 128*128/runconf['pixsize']/runconf['pixsize'] - 1
                calcflux = (np.sum(source_image) - np.sum(bkg_image)) / np.sum(response)
                fitvals = [calcflux]
                mf = fitbkg(rf, calcflux)
                deltasq = (sf - mf)**2 / ((sf + 1e-8)**2)
            chisq = np.sum(deltasq)
            redchi[count] = chisq
            fluxes[count] = fitvals[0]
            if runconf['verbose']: print "{count:3d}/{tot:3d} {tx:6.2f} {ty:6.2f} | {chisq:12.2f} {redchi:9.4f} | {flux:8.2f} {bkg:7.2f}".format(count=count+1, tot=num_pix, tx=theta_x, ty=theta_y, chisq=chisq, redchi=chisq/dof, flux=fitvals[0], bkg=0)

        plot_map(redchi, pixlist, nside, runconf, r"$\chi^2$ for fits", r"Reduced $\chi^2$")
        plotfile.savefig()

        plot_map(fluxes, pixlist, nside, runconf, "Approximate flux for fits", "Flux (amplitude)")
        plotfile.savefig()

        plot_map(resp_strength, pixlist, nside, runconf, "Total transmission as a function of direction", "Relative strength")
        plotfile.savefig()
    # endif not args.noloc - the actual localisation calculations end here

    plotfile.close()


#----------------GEANT enters here--------------------------------------
def powerlaw(E,alpha,A):
	"""
	Function that calculates the single power law scaling for the 
	simulated spectrum.
	"""

	return A*E**alpha


def band(E, alpha = -1.08, beta = -1.75, E0 = 189,  A = 5e-3):
	"""
	Function that calculates the 2 power law scaling for the flat
	simulated spectrum generated by mass model.
	Returns the scaling factor to be applied on the simulation.
	"""
        if (alpha - beta)*E0 >= E:
                return A*(E/100)**alpha*np.exp(-E/E0)
        elif (alpha - beta)*E0 < E:
                return A*((alpha - beta)*E0/100)**(alpha - beta)*np.exp(beta - alpha)*(E/100)**beta

def model(E,alpha,beta,E0,A,typ):
	"""
	Chooses the type of scaling: 2 or 1 power law scaling
	"""
	if typ == "powerlaw":
		return powerlaw(E,alpha,A)
	else :
		return band(E,alpha,beta,E0,A)
	
	
def simulated_dph(grbdir,typ,t_src,alpha,beta,E0,A):
	"""
	Function that creates simulated dph and badpixmap
	from given simulated data and badpix files respectively.
	"""
	filenames = glob.glob(grbdir + "/MM_out/*")
	badpixfile = glob.glob(grbdir + "/*badpix.fits")[0]
	filenames.sort()
	pix_cnts = np.zeros((16384,len(filenames)))
	err_pix_cnts = np.zeros((16384,len(filenames)))
	en = np.arange(5, 261., .5)
	sel  = (en>=100) & (en <= 150)
	en_range = np.zeros(len(filenames))
	for f in range(len(filenames)):
		en_range[f] = filenames[f][20:26]
	err_100_500 = (100.0 <= en_range.astype(np.float)) & (en_range.astype(np.float) <= 500.0)
	err_500_1000 = (500.0 < en_range.astype(np.float)) & (en_range.astype(np.float) <= 1000.0)
	err_1000_2000 = (1000.0 < en_range.astype(np.float)) & (en_range.astype(np.float) <= 2000.0)
	exist_1000_2000 = np.where(err_1000_2000 == True)
	E = np.array([])
	
	print "Indices where energy is in between 1000 and 2000 :",exist_1000_2000[0]
	
	for i,f in enumerate(filenames):
			data = fits.getdata(f + "/SingleEventFile.fits")
			E = np.append(E, float(f[20:26]))
			error = np.sqrt(data) 
			data[:,~sel] = 0.
			error[:,~sel] = 0.
			pix_cnts[:,i] = data.sum(1)*model(E[i], alpha, beta, E0, A,typ)/55.5
			err_pix_cnts[:,i] = np.sqrt(((error*model(E[i], alpha, beta, E0, A,typ)/55.5)**2).sum(1))		
			
	pix_cnts_total = np.zeros((16384,))
	err_100_500_total = np.sqrt((err_pix_cnts[:,err_100_500]**2).sum(1))*(E[err_100_500][1]-E[err_100_500][0])
	err_500_1000_total =  np.sqrt((err_pix_cnts[:,err_500_1000]**2).sum(1))*(E[err_500_1000][1]-E[err_500_1000][0])

	if (len(exist_1000_2000[0]) != 0):
		err_1000_2000_total = np.sqrt((err_pix_cnts[:,err_1000_2000]**2).sum(1))*(E[err_1000_2000][1]-E[err_1000_2000][0])
	else :
		err_1000_2000_total = 0
	
	err_pix_cnts_total = np.sqrt(err_100_500_total**2 + err_500_1000_total**2 + err_1000_2000_total**2) # dE is 5 from 100-500, 10 from 500-1000, 20 from 1000-2000

	for i in range(16384):
			pix_cnts_total[i] = simps(pix_cnts[i,:], E)			

	quad0pix = pix_cnts_total[:4096]
	quad1pix = pix_cnts_total[4096:2*4096]
	quad2pix = pix_cnts_total[2*4096:3*4096]
	quad3pix = pix_cnts_total[3*4096:]
		
	err_quad0pix = err_pix_cnts_total[:4096]
	err_quad1pix = err_pix_cnts_total[4096:2*4096]
	err_quad2pix = err_pix_cnts_total[2*4096:3*4096]
	err_quad3pix = err_pix_cnts_total[3*4096:]
	
	quad0 =  np.reshape(quad0pix, (64,64), 'F')
	quad1 =  np.reshape(quad1pix, (64,64), 'F')
	quad2 =  np.reshape(quad2pix, (64,64), 'F')
	quad3 =  np.reshape(quad3pix, (64,64), 'F')
		
	err_quad0 =  np.reshape(err_quad0pix, (64,64), 'F')
	err_quad1 =  np.reshape(err_quad1pix, (64,64), 'F')
	err_quad2 =  np.reshape(err_quad2pix, (64,64), 'F')
	err_quad3 =  np.reshape(err_quad3pix, (64,64), 'F')
	
	sim_DPH = np.zeros((128,128), float)
	sim_err_DPH = np.zeros((128,128), float)
	
	sim_DPH[:64,:64] = np.flip(quad0, 0)
	sim_DPH[:64,64:] = np.flip(quad1, 0)
	sim_DPH[64:,64:] = np.flip(quad2, 0)
	sim_DPH[64:,:64] = np.flip(quad3, 0)
	
	sim_err_DPH[:64,:64] = np.flip(err_quad0, 0)
	sim_err_DPH[:64,64:] = np.flip(err_quad1, 0)
	sim_err_DPH[64:,64:] = np.flip(err_quad2, 0)
	sim_err_DPH[64:,:64] = np.flip(err_quad3, 0)

	badpix = fits.open(badpixfile)
	dphmask = np.ones((128,128))
	
	badq0 = badpix[1].data # Quadrant 0
	badpixmask = (badq0['PIX_FLAG']!=0)
	dphmask[(63 - badq0['PixY'][badpixmask]) ,badq0['PixX'][badpixmask]] = 0

	badq1 = badpix[2].data # Quadrant 1
	badpixmask = (badq1['PIX_FLAG']!=0)
	dphmask[(63 - badq1['PixY'][badpixmask]), (badq1['PixX'][badpixmask]+64)] = 0

	badq2 = badpix[3].data # Quadrant 2
	badpixmask = (badq2['PIX_FLAG']!=0)
	dphmask[(127 - badq2['PixY'][badpixmask]), (badq2['PixX'][badpixmask]+64)] = 0

	badq3 = badpix[4].data # Quadrant 3
	badpixmask = (badq3['PIX_FLAG']!=0)
	dphmask[(127 - badq3['PixY'][badpixmask]), badq3['PixX'][badpixmask]] = 0
			
	oneD_sim = (sim_DPH*dphmask).flatten()

	return oneD_sim*t_src,sim_DPH*t_src,dphmask,sim_err_DPH*t_src

def data_bkgd_image(grbdir,pre_tstart,pre_tend,grb_tstart,grb_tend,post_tstart,post_tend):
	"""
	Creates source and background dph.
	"""
	infile = glob.glob(grbdir + "/*quad_clean.evt")[0] 
	predph = evt2image(infile,pre_tstart,pre_tend)
	grbdph = evt2image(infile,grb_tstart,grb_tend)
	postdph = evt2image(infile,post_tstart,post_tend)

	bkgddph = predph+postdph

	oneD_grbdph = grbdph.flatten()
	oneD_bkgddph = bkgddph.flatten()
	t_src = grb_tend - grb_tstart
	t_total = (pre_tend-pre_tstart)+(post_tend-post_tstart)

	return oneD_grbdph,oneD_bkgddph,grbdph,bkgddph,t_src,t_total

def fit_line(model,scaling):
	"""
	returns the scaled model by a factor
	"""
	return scaling*model

def fit_line_int(model,scaling,intercept):
	"""
	returns a scaled model added to an intercept
	"""
	return scaling*model + intercept

def plot_vis_test(plotfile,pdf_file):
	"""
	Returns a document with all visual test plots required
	"""
	# First some parameters looked up from configfile---------------------------------
	
	grbdir = runconf['l2file'][0:10]
	pre_tstart = runconf['bkg1start']
	pre_tend = runconf['bkg1end']
	trigtime = runconf['trigtime']
	grb_tstart = runconf['transtart']
	grb_tend = runconf['tranend']
	post_tstart = runconf['bkg2start']
	post_tend = runconf['bkg2end']
	t_src = grb_tend - grb_tstart 
	t_tot = (pre_tend-pre_tstart)+(post_tend-post_tstart)
	ra_tran = runconf['ra']
	dec_tran = runconf['dec']
	lc_bin = runconf['lc_bin']
	alpha = runconf['alpha']
	beta = runconf['beta']
	E0 = runconf['E0']
	A = runconf['A']
	sim_scale = t_src
	pixbin = int(runconf['pixsize'])
	comp_bin = int(runconf['comp_bin'])
	typ = runconf['typ']

	# Calling txy to calculate thetax thetay and the coordinates----------------------
	
	thetax,thetay,x,y,z,t = txy(runconf['mkffile'], trigtime, ra_tran, dec_tran)
	
	# Plot the 3d visualisation for the position of the transient---------------------
	plt.figure()
	fig = visualize_3d(grbdir,x,y,z, t, thetax, thetay, grbdir)	
	pdf_file.savefig(fig)
	
	# Plotting the lightcurves for the four quadrants---------------------------------
	fig = plt.figure()
	clean_file = fits.open(runconf['infile'])
	plt.title('Light curves for '+grbdir + "\n" + r"$\theta_x$={tx:0.1f} and $\theta_y$={ty:0.1f} ".format(tx=thetax,ty=thetay))
	
	quad0 = clean_file[1].data
	data0,bin_edge = np.histogram(quad0['time'], bins=np.arange(quad0['time'][0],quad0['time'][-1],lc_bin))
	plt.plot((bin_edge[:-1]+bin_edge[1:])/2.0,data0,label='Quad 0',lw=0.7)
        quad1 = clean_file[2].data
	data1,bin_edge = np.histogram(quad1['time'], bins=np.arange(quad1['time'][0],quad1['time'][-1],lc_bin))
	plt.plot((bin_edge[:-1]+bin_edge[1:])/2.0,data1,label='Quad 1',lw=0.7)        
	quad2 = clean_file[3].data
	data2,bin_edge = np.histogram(quad2['time'], bins=np.arange(quad2['time'][0],quad2['time'][-1],lc_bin))
	plt.plot((bin_edge[:-1]+bin_edge[1:])/2.0,data2,label='Quad 2',lw=0.7)
        quad3 = clean_file[4].data
        data3,bin_edge = np.histogram(quad3['time'], bins=np.arange(quad3['time'][0],quad3['time'][-1],lc_bin))
        plt.plot((bin_edge[:-1]+bin_edge[1:])/2.0,data3,label='Quad 3',lw=0.7)
	plt.axvspan(grb_tstart,grb_tend,color='blue',alpha=0.1,label='GRB')
	plt.axvspan(pre_tstart,pre_tend,color='orange',alpha=0.2)
	plt.axvspan(post_tstart,post_tend,color='orange',alpha=0.2,label='Background')
	plt.legend(prop={'size':6})
	plt.xlim(pre_tstart-100,post_tend+100)
	pdf_file.savefig(fig)
	
	# Calling the sim_dph--------------------------------------------------------------
	
	grb_flat,bkgd_flat,grb_dph,bkgd_dph,t_src,t_total = data_bkgd_image(grbdir,pre_tstart,pre_tend,grb_tstart,grb_tend,post_tstart,post_tend)

	sim_flat,sim_dph,badpix_mask,sim_err_dph = simulated_dph(grbdir,typ,t_src,alpha,beta,E0,A)

	src_dph = grb_dph-bkgd_dph*t_src/t_tot

        print "Total counts in simulated dph: ",(sim_dph).sum()
        print "Total counts after badpix mask is applied: ",(sim_dph*badpix_mask).sum()
	print "Excess counts in badpix masked src dph: ",(src_dph*badpix_mask).sum()
 
	# Plotting the DPHs before badpix correction---------------------------------------
	
	f,((ax1,ax2),(ax3,ax4)) = plt.subplots(2,2)
	plt.suptitle('DPHs before badpix correction for '+grbdir + "\n" + r"$\theta_x$={tx:0.1f} and $\theta_y$={ty:0.1f} ".format(tx=thetax,ty=thetay))
        	# Sim
	im = ax3.imshow(sim_dph,interpolation='none')
	ax3.set_title('Sim DPH',fontsize=8)
	ax3.set_xlim(-1,128 - 0.5)
	ax3.axvline(x=-0.75,ymin=0,ymax=64,linewidth=5,color='k')
	ax3.spines['left'].set_position(('data',-0.5))
	ax3.set_yticklabels([])
	ax3.xaxis.set_ticks(np.arange(0,128,16))
	f.colorbar(im,ax=ax3,fraction=0.046, pad=0.04)
	
	        # Source 
	im = ax4.imshow(src_dph,interpolation='none',vmin=0)
	ax4.set_title('Src DPH (bkg subtracted)',fontsize=8)
	ax4.set_xlim(-1,128 -0.5)
	ax4.axvline(x=-0.75,ymin=0,ymax=64,linewidth=5,color='k')
	ax4.spines['left'].set_position(('data',-0.5))
	ax4.set_yticklabels([])
	ax4.xaxis.set_ticks(np.arange(0,128,16))
	f.colorbar(im,ax=ax4,fraction=0.046, pad=0.04)

        	# Source + Background
	im = ax1.imshow(grb_dph,interpolation='none')
	ax1.set_title('Src + Bkg DPH',fontsize=8)
	ax1.set_xlim(-1,128 -0.5)
	ax1.axvline(x=-0.75,ymin=0,ymax=64,linewidth=5,color='k')
	ax1.spines['left'].set_position(('data',-0.5))
	ax1.set_yticklabels([])
	ax1.xaxis.set_ticks(np.arange(0,128,16))
	f.colorbar(im,ax=ax1,fraction=0.046, pad=0.04)

        	# Background
	im = ax2.imshow(bkgd_dph*t_src/t_total,interpolation='none')
	ax2.set_title('Bkg DPH',fontsize=8)
	ax2.set_xlim(-1,128 -0.5)
	ax2.axvline(x=-0.75,ymin=0,ymax=64,linewidth=5,color='k')
	ax2.spines['left'].set_position(('data',-0.5))
	ax2.set_yticklabels([])
	ax2.xaxis.set_ticks(np.arange(0,128,16))
	f.colorbar(im,ax=ax2,fraction=0.046, pad=0.04)
	f.set_size_inches([6.5,6.5])
	pdf_file.savefig(f)  # saves the current figure into a pdf_file page
	
	# Plotting the Badpix mask---------------------------------------------

	fig = plt.figure()
	ax = plt.subplot(111)
	plt.title('Badpix Mask for '+grbdir+ "\n" + r"$\theta_x$={tx:0.1f} and $\theta_y$={ty:0.1f} ".format(tx=thetax,ty=thetay))
	im = ax.imshow(badpix_mask,interpolation='none')
	ax.set_xlim(-9,128 -0.5)
	ax.axvline(x=-5.,ymin=0,ymax=64,linewidth=5,color='k')
	ax.spines['left'].set_position(('data',-0.5))
	ax.xaxis.set_ticks(np.arange(0,128,16))
	ax.yaxis.set_ticks(np.arange(0,128,16))
	fig.colorbar(im,ax=ax,fraction=0.046, pad=0.04)
	
	pdf_file.savefig(fig)  # saves the current figure into a pdf_file page

	# Plotting badpix masked graphs--------------------------------------------
	f,((ax1,ax2),(ax3,ax4)) = plt.subplots(2,2)
	plt.suptitle('DPHs after badpix correction for '+grbdir+ "\n" + r"$\theta_x$={tx:0.1f} and $\theta_y$={ty:0.1f} ".format(tx=thetax,ty=thetay))
        	# Sim
	im = ax3.imshow(sim_dph*badpix_mask,interpolation='none')
	ax3.set_title('Sim DPH',fontsize=8)
	ax3.set_xlim(-1,128 -0.5)
	ax3.axvline(x=-0.75,ymin=0,ymax=64,linewidth=5,color='k')
	ax3.spines['left'].set_position(('data',-0.5))
	ax3.set_yticklabels([])
	ax3.xaxis.set_ticks(np.arange(0,128,16))
	f.colorbar(im,ax=ax3,fraction=0.046, pad=0.04)

	        # Source 
	im = ax4.imshow(src_dph*badpix_mask,interpolation='none',vmin=0)
	ax4.set_title('Src DPH (bkg subtracted)',fontsize=8)
	ax4.set_xlim(-1,128 -0.5)
	ax4.axvline(x=-0.75,ymin=0,ymax=64,linewidth=5,color='k')
	ax4.spines['left'].set_position(('data',-0.5))
	ax4.set_yticklabels([])
	ax4.xaxis.set_ticks(np.arange(0,128,16))
	f.colorbar(im,ax=ax4,fraction=0.046, pad=0.04)

	        # Source + Background
	im = ax1.imshow(grb_dph*badpix_mask,interpolation='none')
	ax1.set_title('Src + Bkg DPH',fontsize=8)
	ax1.set_xlim(-1,128 -0.5)
	ax1.axvline(x=-0.75,ymin=0,ymax=64,linewidth=5,color='k')
	ax1.spines['left'].set_position(('data',-0.5))
	ax1.set_yticklabels([])
	ax1.xaxis.set_ticks(np.arange(0,128,16))
	f.colorbar(im,ax=ax1,fraction=0.046, pad=0.04)
	
	        # Background
	im = ax2.imshow(bkgd_dph*badpix_mask*t_src/t_total,interpolation='none')
	ax2.set_title('Bkg DPH',fontsize=8)
	ax2.set_xlim(-1,128 -0.5)
	ax2.axvline(x=-0.75,ymin=0,ymax=64,linewidth=5,color='k')
	ax2.spines['left'].set_position(('data',-0.5))
	ax2.set_yticklabels([])
	ax2.xaxis.set_ticks(np.arange(0,128,16))
	f.colorbar(im,ax=ax2,fraction=0.046, pad=0.04)
	f.set_size_inches([6.5,6.5])
	pdf_file.savefig(f)  # saves the current figure into a pdf_file page

	# Plotting badpix masked graphs (Binned) ----------------------------------------------------
	for p in [4,8,16]:
		f,((ax1,ax2),(ax3,ax4)) = plt.subplots(2,2)
		plt.suptitle('DPHs after badpix correction for '+grbdir+ "\n" + r"$\theta_x$={tx:0.1f} and $\theta_y$={ty:0.1f} ".format(tx=thetax,ty=thetay)+ "pixsize="+str(p))
		        # Sim
		im = ax3.imshow(resample(sim_dph*badpix_mask,p),interpolation='none')
		ax3.set_title('Sim DPH',fontsize=8)
		ax3.set_xlim(-1,128/p -0.5)
		ax3.axvline(x=-0.75,ymin=0,ymax=64,linewidth=5,color='k')
		ax3.spines['left'].set_position(('data',-0.5))
		ax3.set_yticklabels([])
		ax3.xaxis.set_ticks(np.arange(0,(128/p),16/p))
		ax3.set_xticklabels(np.arange(0,128,16))
		f.colorbar(im,ax=ax3,fraction=0.046, pad=0.04)
		
		        # Source 
		im = ax4.imshow(resample(src_dph*badpix_mask,p),interpolation='none',vmin=0)
		ax4.set_title('Src DPH (bkg subtracted)',fontsize=8)
		ax4.set_xlim(-1,128/p -0.5)
		ax4.axvline(x=-0.75,ymin=0,ymax=64,linewidth=5,color='k')
		ax4.spines['left'].set_position(('data',-0.5))
		ax4.set_yticklabels([])
                ax4.xaxis.set_ticks(np.arange(0,(128/p),16/p))
                ax4.set_xticklabels(np.arange(0,128,16))		
		f.colorbar(im,ax=ax4,fraction=0.046, pad=0.04)
		
		        # Source + Background
		im = ax1.imshow(resample(grb_dph*badpix_mask,p),interpolation='none')
		ax1.set_title('Src + Bkg DPH',fontsize=10)
		ax1.set_xlim(-1,128/p -0.5)
		ax1.axvline(x=-0.75,ymin=0,ymax=64,linewidth=5,color='k')
		ax1.spines['left'].set_position(('data',-0.5))
		ax1.set_yticklabels([])
                ax1.xaxis.set_ticks(np.arange(0,(128/p),16/p))
                ax1.set_xticklabels(np.arange(0,128,16))		
		f.colorbar(im,ax=ax1,fraction=0.046, pad=0.04)
		
		        # Background
		im = ax2.imshow(resample(bkgd_dph*badpix_mask*t_src/t_total,p),interpolation='none')
		ax2.set_title('Bkg DPH',fontsize=8)
		ax2.set_xlim(-1,128/p -0.5)
		ax2.axvline(x=-0.75,ymin=0,ymax=64,linewidth=5,color='k')
		ax2.spines['left'].set_position(('data',-0.5))
		ax2.set_yticklabels([])
                ax2.xaxis.set_ticks(np.arange(0,(128/p),16/p))
                ax2.set_xticklabels(np.arange(0,128,16))		
		f.colorbar(im,ax=ax2,fraction=0.046, pad=0.04)
		f.set_size_inches([6.5,6.5])
		
		pdf_file.savefig(f)  # saves the current figure into a pdf_file page


	# Plotting the comparison graphs with equal bins ---------------------------------------
	print "No. of pixels with zero counts in sim_dph: ",sim_dph[sim_dph==0].size
	print "No. of pixels with zero counts in grb_dph(no bkg subtration): ",grb_dph[grb_dph==0].size
	
	# Generating the array for module number ------------------------------------------------
	A = ['A'+str(i) for i in range(16)]
	B = np.flip(['B'+str(i) for i in range(16)],0)
	C = np.flip(['C'+str(i) for i in range(16)],0)
	D = ['D'+str(i) for i in range(16)]
	quad_a = np.reshape(A,(4,4))
	quad_b = np.reshape(B,(4,4))
	quad_c = np.reshape(C,(4,4))
	quad_d = np.reshape(D,(4,4))
	Mod_arr = np.ndarray((8,8),dtype='|S3')
	Mod_arr[:4,:4] = quad_a
	Mod_arr[:4,4:] = quad_b
	Mod_arr[4:,4:] = quad_c
	Mod_arr[4:,:4] = quad_d
	Mod_names = Mod_arr.flatten()
	#print "Module name array : ",Mod_names
	#-----------------------------------------------------------------------------------------
		
	sim_dph = sim_dph*badpix_mask
	sim_err_dph = sim_err_dph*badpix_mask
        grb_dph = grb_dph*badpix_mask
        bkgd_dph = bkgd_dph*badpix_mask
	grb_err_dph = np.sqrt(grb_dph)*badpix_mask
	bkgd_err_dph = np.sqrt(bkgd_dph)*badpix_mask

	sim_bin = resample(sim_dph,pixbin)
	sim_err_bin = np.sqrt(resample(sim_err_dph**2,pixbin))	
	grb_bin = resample(grb_dph,pixbin)
	bkgd_bin = resample(bkgd_dph,pixbin)
	grb_err_bin = np.sqrt(resample(grb_err_dph,pixbin))	
	bkgd_err_bin = np.sqrt(resample(bkgd_err_dph,pixbin))	

	sim_flat_bin = sim_bin.flatten()
	sim_err_flat_bin = sim_err_bin.flatten()
	grb_flat_bin = grb_bin.flatten()
	bkgd_flat_bin = bkgd_bin.flatten()
	grb_err_flat_bin = grb_err_bin.flatten()
	bkgd_err_flat_bin = bkgd_err_bin.flatten()
	

	        # Defining model background and data
	model = sim_flat_bin
	model_copy = np.copy(model)
	bkgd = bkgd_flat_bin*t_src/t_tot
	src = grb_flat_bin
	
	data = src - bkgd
	data_copy = np.copy(data)
	
	err_src = grb_err_flat_bin
	err_bkgd = bkgd_err_flat_bin
	err_model = sim_err_flat_bin
	err_model_copy = np.copy(err_model)
	err_data = np.sqrt(((err_src)**2) + ((err_bkgd)**2)*(t_src/t_total)**2)
	err_data_copy = np.copy(err_data)
	
	ratio = data/model
	err_ratio = ratio*np.sqrt(((err_data/data)**2) + ((err_model/model)**2))
	
	chi_sq = (((model-data)**2)/((err_model)**2 + (err_data)**2)).sum()
	
	        # PLotting the comparison plots
	f,(ax1,ax2) = plt.subplots(2,gridspec_kw={'height_ratios':[2,1]},sharex='row')
	
	ax1.set_title("Comparison between simulated and real data for "+grbdir+ "\n" + r"$\theta_x$={tx:0.1f} and $\theta_y$={ty:0.1f} $\chi^2$={c:0.1f}".format(tx=thetax,ty=thetay,c=chi_sq))
	ax1.errorbar(np.arange(0,(len(data))),data,yerr=err_data,fmt='.',markersize=2,label="Data",elinewidth=0.5)
	ax1.errorbar(np.arange(0,(len(model))),model,yerr=err_model,fmt='.',markersize=2,label="Simulation",elinewidth=0.5)
	ax1.legend()
        ax1.xaxis.set_ticks(np.arange(0,len(data)))
	ax1.set_ylabel('Counts')
	ax1.xaxis.grid(linewidth=0.5,alpha=0.3)
        ax1.set_xticklabels(Mod_names,rotation=90,fontsize=5)

	ax2.errorbar(np.arange(0,(len(ratio))),ratio,yerr=err_ratio,fmt='.',markersize=2,label="Ratio = Data/Model",elinewidth=0.5)
	ax2.xaxis.set_ticks(np.arange(0,len(data)))
        ax2.set_xticklabels(Mod_names,rotation=90,fontsize=5)
        ax2.yaxis.set_ticks(np.arange(int(min(ratio-err_ratio)-1),int(max(ratio+err_ratio)+2),1))
	ax2.tick_params(labelsize=5)
	ax2.axhline(y=1,linewidth=0.5,color='k')
	ax2.legend()
	ax2.set_xlabel('CZT Modules')
	ax2.set_ylabel('Ratio of counts')
	ax2.xaxis.grid(linewidth=0.5,alpha=0.3)
	plt.tight_layout(h_pad=0.0)
	f.set_size_inches([6.5,10])
	pdf_file.savefig(f,orientation='portrait')  # saves the current figure into a pdf_file page

	# Plotting comparison graphs with random binning------------------------------
	
        sim_flat = sim_dph.flatten()
	sim_err_flat = sim_err_dph.flatten()
        grb_flat = grb_dph.flatten()
        bkgd_flat = bkgd_dph.flatten()
	src_flat = src_dph.flatten()
	
	order = np.random.permutation(np.arange(0,len(sim_flat)))
	
        sim_flat = sim_flat[order]
	sim_err_flat = sim_err_flat[order]
	grb_flat = grb_flat[order]
	bkgd_flat = bkgd_flat[order]
	src_flat = src_flat[order]
	
	print "No. of pixels with zero counts in sim_flat: ",sim_flat[sim_flat==0].size
	print "No. of pixels with zero counts in src_flat: ",src_flat[src_flat==0].size
	
	bins = np.array(np.sort(np.random.uniform(0,1,comp_bin)*len(sim_flat)),dtype=np.int64)
	x = np.zeros(len(bins)+2,dtype=np.int64)
	x[0] = 0
	x[-1] = len(sim_flat)
	x[1:-1] = bins
	
	#print "The bin edges: ",x # ---------------------------------------------------------------
	
	sim_flat_bin = np.array([sim_flat[x[i]:x[i+1]].sum() for i in range(comp_bin+1)])
	sim_err_flat_bin = np.sqrt(np.array([(sim_err_flat[x[i]:x[i+1]]**2).sum() for i in range(comp_bin+1)]))
	grb_flat_bin = np.array([grb_flat[x[i]:x[i+1]].sum() for i in range(comp_bin+1)])
	grb_err_flat_bin = np.sqrt(np.array([grb_flat[x[i]:x[i+1]].sum() for i in range(comp_bin+1)]))
	bkgd_flat_bin = np.array([bkgd_flat[x[i]:x[i+1]].sum() for i in range(comp_bin+1)])
	bkgd_err_flat_bin = np.sqrt(np.array([bkgd_flat[x[i]:x[i+1]].sum() for i in range(comp_bin+1)]))
	src_flat_bin = np.array([src_flat[x[i]:x[i+1]].sum() for i in range(comp_bin+1)])
	
	print "Total sim_flat_bin : ",sim_flat_bin.sum() #-----------------------------------------
	#print " Max(cumsum) : ",max(np.cumsum(sim_flat)) #-----------------------------------------

                # Defining model background and data
        model = sim_flat_bin #avg_flat_bin
        bkgd = bkgd_flat_bin*t_src/t_tot
        src = grb_flat_bin
	
        data = src - bkgd

        err_src = np.sqrt(src)
        err_bkgd = np.sqrt(bkgd_flat_bin)
        err_model = sim_err_flat_bin
        err_data = np.sqrt(((err_src)**2) + ((err_bkgd)**2)*(t_src/t_total)**2)
	
	chi_sq_new = (((model-data)**2)/((err_model)**2 + (err_data)**2)).sum()
                # PLotting the comparison plots
        fig = plt.figure()
        plt.title("Comparison between simulated and real data for "+grbdir+ "\n" + r"$\theta_x$={tx:0.1f} and $\theta_y$={ty:0.1f} $\chi^2$={c:0.1f} ".format(tx=thetax,ty=thetay,c=chi_sq_new))
        plt.errorbar(np.arange(0,(len(data))),data,yerr=err_data,fmt='.',markersize=2,label="Data",elinewidth=0.5)
        plt.errorbar(np.arange(0,(len(model))),model,yerr=err_model,fmt='.',markersize=2,label="Simulation",elinewidth=0.5)
        plt.ylabel('Counts')
	plt.xlabel('Random Bins')
	plt.xticks(np.arange(0,(len(data)),1))
	plt.legend()
        pdf_file.savefig(fig)   #saves the current figure into a pdf_file page

	# Plotting observed vs predicted counts------------------------------------------------------

	fig = plt.figure()
	plt.title(grbdir + r" : Observed vs Predicted counts with $\chi^2$={cs:0.1f}".format(cs=chi_sq))
	plt.errorbar(model_copy,data_copy,xerr=err_model_copy,yerr=err_data_copy,fmt='g.',markersize=2,elinewidth=0.5)
	for i in range(len(model_copy)):
		plt.text(model_copy[i],data_copy[i],Mod_names[i],fontsize=5)
	plt.plot(np.arange(-1000,1000),np.arange(-1000,1000),'k',linewidth=0.5)
	plt.xlim(min(model_copy)-5,max(model_copy)+5)
	plt.ylim(min(data_copy)-5,max(data_copy)+5)
	plt.xlabel('Predicted Counts')
	plt.ylabel('Observed Counts')
	plt.legend()
	plt.grid()
	pdf_file.savefig(fig)

	# Scaling the model using curve fit =============================================================== 
	
	param,pcov = curve_fit(fit_line_int,model_copy,data_copy)
	scaling = param[0]
	intercept = param[1]
	
	# Plotting the scaled plots ===================================================================
	# Plotting the comparison graphs with equal bins ---------------------------------------

	sim_dph = sim_dph*badpix_mask
	sim_err_dph = sim_err_dph*badpix_mask
        grb_dph = grb_dph*badpix_mask
        bkgd_dph = bkgd_dph*badpix_mask
	grb_err_dph = np.sqrt(grb_dph)*badpix_mask
	bkgd_err_dph = np.sqrt(bkgd_dph)*badpix_mask

	sim_bin = resample(sim_dph,pixbin)
	sim_err_bin = np.sqrt(resample(sim_err_dph**2,pixbin))	
	grb_bin = resample(grb_dph,pixbin)
	bkgd_bin = resample(bkgd_dph,pixbin)
	grb_err_bin = np.sqrt(resample(grb_err_dph,pixbin))	
	bkgd_err_bin = np.sqrt(resample(bkgd_err_dph,pixbin))	

	sim_flat_bin = sim_bin.flatten()
	sim_err_flat_bin = sim_err_bin.flatten()
	grb_flat_bin = grb_bin.flatten()
	bkgd_flat_bin = bkgd_bin.flatten()
	grb_err_flat_bin = grb_err_bin.flatten()
	bkgd_err_flat_bin = bkgd_err_bin.flatten()
	

	        # Defining model background and data
	#model = sim_flat_bin*scaling
	model = sim_flat_bin*scaling + intercept
	bkgd = bkgd_flat_bin*t_src/t_tot
	src = grb_flat_bin
	
	data = src - bkgd
	
	err_src = grb_err_flat_bin
	err_bkgd = bkgd_err_flat_bin
	#err_model = sim_err_flat_bin*scaling
	err_model = sim_err_flat_bin*scaling
	err_data = np.sqrt(((err_src)**2) + ((err_bkgd)**2)*(t_src/t_total)**2)
	
	ratio = data/model
	err_ratio = ratio*np.sqrt(((err_data/data)**2) + ((err_model/model)**2))
	
	chi_sq = (((model-data)**2)/((err_model)**2 + (err_data)**2)).sum()
	
	        # PLotting the comparison plots
	f,(ax1,ax2) = plt.subplots(2,gridspec_kw={'height_ratios':[2,1]},sharex='row')
	
	ax1.set_title("Comparison between simulated (scaled) and real data for "+grbdir+ "\n" + r"$\theta_x$={tx:0.1f} and $\theta_y$={ty:0.1f} $\chi^2$={c:0.1f} ".format(tx=thetax,ty=thetay,c=chi_sq))
	ax1.errorbar(np.arange(0,(len(data))),data,yerr=err_data,fmt='.',markersize=2,label="Data",elinewidth=0.5)
	ax1.errorbar(np.arange(0,(len(model))),model,yerr=err_model,fmt='.',markersize=2,label="Simulation (scaling = {s:0.2f},offset = {o:0.2f})".format(s=scaling,o=intercept),elinewidth=0.5)
	ax1.legend()
        ax1.xaxis.set_ticks(np.arange(0,len(data)))
	ax1.set_ylabel('Counts')
	ax1.xaxis.grid(linewidth=0.5,alpha=0.3)
        ax1.set_xticklabels(Mod_names,rotation=90,fontsize=5)
	
	ax2.errorbar(np.arange(0,(len(ratio))),ratio,yerr=err_ratio,fmt='.',markersize=2,label="Ratio = Data/Model(scaling = {s:0.2f}, offset={o:0.2f})".format(s=scaling,o=intercept),elinewidth=0.5)
	ax2.xaxis.set_ticks(np.arange(0,len(data)))
        ax2.set_xticklabels(Mod_names,rotation=90,fontsize=5)
        ax2.yaxis.set_ticks(np.arange(int(min(ratio-err_ratio)-1),int(max(ratio+err_ratio)+2),1))
	ax2.tick_params(labelsize=5)
	ax2.axhline(y=1,linewidth=0.5,color='k')
	ax2.legend()
	ax2.set_xlabel('CZT Modules')
	ax2.set_ylabel('Ratio of counts')
	ax2.xaxis.grid(linewidth=0.5,alpha=0.3)
	plt.tight_layout(h_pad=0.0)
	f.set_size_inches([6.5,10])
	pdf_file.savefig(f,orientation='portrait')  # saves the current figure into a pdf_file page

	# Plotting comparison graphs with random binning------------------------------
	
        sim_flat = sim_dph.flatten()
	sim_err_flat = sim_err_dph.flatten()
        grb_flat = grb_dph.flatten()
        bkgd_flat = bkgd_dph.flatten()
	src_flat = src_dph.flatten()
	
	order = np.random.permutation(np.arange(0,len(sim_flat)))
	
        sim_flat = sim_flat[order]
	sim_err_flat = sim_err_flat[order]
	grb_flat = grb_flat[order]
	bkgd_flat = bkgd_flat[order]
	src_flat = src_flat[order]
	
	bins = np.array(np.sort(np.random.uniform(0,1,comp_bin)*len(sim_flat)),dtype=np.int64)
	x = np.zeros(len(bins)+2,dtype=np.int64)
	x[0] = 0
	x[-1] = len(sim_flat)
	x[1:-1] = bins
	
	sim_flat_bin = np.array([sim_flat[x[i]:x[i+1]].sum() for i in range(comp_bin+1)])
	sim_err_flat_bin = np.sqrt(np.array([(sim_err_flat[x[i]:x[i+1]]**2).sum() for i in range(comp_bin+1)]))
	grb_flat_bin = np.array([grb_flat[x[i]:x[i+1]].sum() for i in range(comp_bin+1)])
	grb_err_flat_bin = np.sqrt(np.array([grb_flat[x[i]:x[i+1]].sum() for i in range(comp_bin+1)]))
	bkgd_flat_bin = np.array([bkgd_flat[x[i]:x[i+1]].sum() for i in range(comp_bin+1)])
	bkgd_err_flat_bin = np.sqrt(np.array([bkgd_flat[x[i]:x[i+1]].sum() for i in range(comp_bin+1)]))
	src_flat_bin = np.array([src_flat[x[i]:x[i+1]].sum() for i in range(comp_bin+1)])
	
                # Defining model background and data
        #model = sim_flat_bin*scaling
	model = sim_flat_bin*scaling + intercept
        bkgd = bkgd_flat_bin*t_src/t_tot
        src = grb_flat_bin
	
        data = src - bkgd

        err_src = np.sqrt(src)
        err_bkgd = np.sqrt(bkgd_flat_bin)
        #err_model = sim_err_flat_bin*scaling
	err_model = sim_err_flat_bin*scaling
        err_data = np.sqrt(((err_src)**2) + ((err_bkgd)**2)*(t_src/t_total)**2)
	
	chi_sq_new = (((model-data)**2)/((err_model)**2 + (err_data)**2)).sum()
                # PLotting the comparison plots
        fig = plt.figure()
        plt.title("Comparison between simulated(scaled) and real data for "+grbdir+ "\n" + r"$\theta_x$={tx:0.1f} and $\theta_y$={ty:0.1f} $\chi^2$={c:0.1f} ".format(tx=thetax,ty=thetay,c=chi_sq_new))
        plt.errorbar(np.arange(0,(len(data))),data,yerr=err_data,fmt='.',markersize=2,label="Data",elinewidth=0.5)
        plt.errorbar(np.arange(0,(len(model))),model,yerr=err_model,fmt='.',markersize=2,label="Simulation (scaling = {s:0.2f}, offset = {o:0.2f})".format(s=scaling,o=intercept),elinewidth=0.5)
        plt.ylabel('Counts')
	plt.xlabel('Random Bins')
	plt.xticks(np.arange(0,(len(data)),1))
	plt.legend()
        pdf_file.savefig(fig)   #saves the current figure into a pdf_file page


	# Plotting observed vs predicted counts--------------------------------------------------------

	fig = plt.figure()
        plt.title(grbdir + r" : Observed vs Predicted counts with $\chi^2$ = {cs:0.1f}".format(cs=chi_sq))
        plt.errorbar(model_copy,data_copy,xerr=err_model_copy,yerr=err_data_copy,fmt='g.',markersize=2,elinewidth=0.5)
	for i in range(len(model_copy)):	
		plt.text(model_copy[i],data_copy[i],Mod_names[i],fontsize=5)
        #plt.plot(np.arange(-1000,1000),fit_line(np.arange(-1000,1000),scaling),'k',linewidth=0.5,label='m = {s:0.2f}'.format(s=scaling))
	plt.plot(np.arange(-1000,1000),fit_line_int(np.arange(-1000,1000),scaling,intercept),'k',linewidth=0.5,label='scaling = {s:0.2f}, offset = {i:0.2f}'.format(s=scaling,i=intercept))
	plt.plot(np.arange(min(model_copy)-5,max(model_copy)+5),np.ones(len(np.arange(min(model_copy)-5,max(model_copy)+5)))*intercept,'r-',label='intercept',linewidth=0.5)
        plt.xlim(min(model_copy)-5,max(model_copy)+5)
        plt.ylim(min(data_copy)-5,max(data_copy)+5)
        plt.xlabel('Predicted Counts')
        plt.ylabel('Observed Counts')
	plt.legend()
	plt.grid()
        pdf_file.savefig(fig)
		
	print "==============================================================================================="
	
	return  

#------------------------------------------------------------------------

#energies = [70]
#thetas_x = [25, 35, 45]
#thetas_y = [40, 50, 60, 70]
#thetas_x = [25, 30, 35, 40, 45]
#thetas_y = [50, 55, 60, 65, 70]

#thetas_x = np.arange(15, 56, 3)
#thetas_y = np.arange(35, 76, 3)
#thetas_x = np.arange(25, 45, 3)
#thetas_y = np.arange(48, 68, 3)

# GRB160119A
#     runconf['infile'] = 'GRB160119A.evt'
#     #runconf['infile'] = "test_dx30_dy60.evt"
#     src_image = evt2image(runconf['infile'], 190868870., 190868970.)   # t_trig+100 to +200 - note that peak is trig+150
#     crude_source_image, mask_src = clean_mask(src_image)
#     print "Source photons    : {sp:0d}".format(sp=int(np.sum(crude_source_image)))
#     
#     # Background image
#     bkg_image1 = evt2image(runconf['infile'], 190868970., 190868670.)   # -100 to -400
#     bkg_image2 = evt2image(runconf['infile'], 190869270., 190869570.)   # +500 to +800
#     
# GRB160119A is at approx 40, -30, peak is at 190868770+150 sec
# thetas_x = np.arange(20, 60, 5)
# thetas_y = np.arange(-50, -10, 5)
#
# source_image = inject(60, 50, 40, 15, 5)
# GRB151006A is at 34, 58
#   # Source image:
#   infile = 'AS1P01_003T01_9000000002cztM0_level2.evt_grb_gt60kev'
#   #infile = "test_dx30_dy60.evt"
#   src_image = evt2image(runconf['infile'], 181821200., 181821400.)
#   crude_source_image, mask_src = clean_mask(src_image)
#   print "Source photons    : {sp:0d}".format(sp=int(np.sum(crude_source_image)))
#   
#   # Background image
#   bkg_image1 = evt2image(runconf['infile'], 181821000., 181821200.)
#   bkg_image2 = evt2image(runconf['infile'], 181821500., 181821700.)
#   bkg_image0 = (bkg_image1 + bkg_image2) / 2.0
#   crude_bkg_image, mask_bkg = clean_mask(bkg_image0)
#   print "Background photons: {bp:0d}".format(bp=int(np.sum(crude_bkg_image)))


#---------crude_bkg_image---------------------------------------------------------------
# Main code begins:

if __name__ == "__main__":
    # Parse arguments to get config file
	args = parser.parse_args()
	runconf = get_configuration(args)
	imsize = 128/runconf['pixsize']
	plotfile = runconf['plotfile']

	pdf_file = PdfPages(plotfile)

	plot_vis_test(plotfile,pdf_file)
	
	pdf_file.close()    
            
    #for     ra in ra0 + runconf['radius'] * 0.9 / np.cos(np.deg2rad(runconf['dec'])) * np.arange(-1, 1.1):
    #        hp.projtext(ra, dec0, "{ra:0.1f},{dec:0.1f}".format(ra=ra,dec=dec0), lonlat=True)
    #        hp.projscatter(ra, dec0, lonlat=True)
    #for     dec in dec0 + runconf['radius'] * np.arange(-1, 1.1):
    #        hp.projtext(ra0, dec, "{ra:0.1f},{dec:0.1f}".format(ra=ra0,dec=dec), lonlat=True)
    #        hp.projscatter(ra0, dec, lonlat=True)
            
            
    #  f    ig2 = plt.figure()
    #  p    lt.contourf(thetas_y, thetas_x, redchi*dof)
    #  p    lt.xlabel(r"$\theta_y$")
    #  p    lt.ylabel(r"$\theta_x$")
    #  p    lt.title(r"$\chi^2$")
    #  p    lt.colorbar()
    #  x    lims = plt.xlim()
    #  y    lims = plt.ylim()
    #  p    lt.scatter(58, 34, s=30, color='black', marker='x', linewidths=3)
    #  p    lt.xlim(xlims)
    #  p    lt.ylim(ylims)
    #  p    lt.show()
            
    #  f    ig3 = plt.figure()
    #  p    lt.contourf(thetas_y, thetas_x, fluxes)
    #  p    lt.xlabel(r"$\theta_y$")
    #  p    lt.ylabel(r"$\theta_x$")
    #  p    lt.title(r"Flux")
    #  p    lt.colorbar()
    #  p    lt.show()` 
