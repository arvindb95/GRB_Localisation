"""
    Code for localising grb's from mass model grid data and pipeline cleaned data files.
    Returns chi_sq contour plots and dph's.

    Version 1.1
    January 12 2018
    Arvind Balasubramanian

"""
########################### Importing required libraries ########################################

import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.table import Table
import glob
import ConfigParser, argparse
from matplotlib.backends.backend_pdf import PdfPages
import astropy.coordinates as coo
import astropy.units as u
import esutil as es
from astropy.stats import sigma_clip, sigma_clipped_stats
from scipy.integrate import simps
from scipy.optimize import curve_fit
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import griddata
from matplotlib import cm
import pylab as pl
from mpl_toolkits.basemap import Basemap

import time

########################### Defining all the required functions ################################

t0 = time.time()

# 1. Functions to read the configfile

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


# 2. Function to calculate the theta phi from the ra dec in the conf file

def get_angles(mkffile, trigtime, ra_tran, dec_tran, window=10):
    """
    Calculate thetax, thetay using astropy
    Use pitch, roll and yaw information from the MKF file
    """
    # x = -yaw
    # y = +pitch
    # z = +roll

    # Read in the MKF file
    mkfdata = fits.getdata(mkffile, 1)
    sel = abs(mkfdata['time'] - trigtime) < window

    # Get pitch, roll, yaw
    # yaw is minus x
    pitch = coo.SkyCoord( np.median(mkfdata['pitch_ra'][sel]) * u.deg,np.median(mkfdata['pitch_dec'][sel]) * u.deg )
    roll = coo.SkyCoord( np.median(mkfdata['roll_ra'][sel]) * u.deg,np.median(mkfdata['roll_dec'][sel]) * u.deg )
    yaw_ra  = (180.0 + np.median(mkfdata['yaw_ra'][sel]) ) % 360
    yaw_dec = -np.median(mkfdata['yaw_dec'][sel])
    minus_yaw = coo.SkyCoord( yaw_ra * u.deg, yaw_dec * u.deg )

    # Earth - the mkffile has satellite xyz 
    earthx = np.median(mkfdata['posx'][sel]) * u.km
    earthy = np.median(mkfdata['posy'][sel]) * u.km
    earthz = np.median(mkfdata['posz'][sel]) * u.km
    earth = coo.SkyCoord(-earthx, -earthy, -earthz, frame='icrs',representation='cartesian')

    # Sun coordinates:
    #sunra = np.median(mkfdata['sun_ra'][sel]) * u.deg
    #sundec = np.median(mkfdata['sun_dec'][sel]) * u.deg
    #sun = coo.SkyCoord(sunra, sundec)
    # Transient:
    transient = coo.SkyCoord(ra_tran * u.deg, dec_tran * u.deg)

    # Angles from x, y, z axes are:
    ax = minus_yaw.separation(transient)
    ay = pitch.separation(transient)
    az = roll.separation(transient)

    theta = az.value # Theta is the angle of the transient from z axis

    # the components are:
    cx = np.cos(ax.radian) # The .radian is not really needed, but anyway...
    cy = np.cos(ay.radian)
    cz = np.cos(az.radian)

    # Thetax = angle from z axis in ZX plane
    # lets use arctan2(ycoord, xcoord) for this
    thetax = u.rad * np.arctan2(cx, cz)
    thetay = u.rad * np.arctan2(cy, cz)

    phi = u.rad * np.arctan2(cy, cx) # phi is the angle of the transient from the x axis

    if (phi.to(u.deg).value < 0 ):
        phi_new = 360 + phi.to(u.deg).value
    else:
        phi_new = phi.to(u.deg).value
    return theta, phi_new, thetax.to(u.deg).value,thetay.to(u.deg).value, minus_yaw, pitch, roll, transient, earth

# 3. Function to get the coordinates of the center of the trixel in HTM grid in which the transient point falls

def get_center(depth,trans_theta,trans_phi):
    """
    Takes transient coordinates (trans_theta,trans_phi) and returns the coordinates of the 
    pixel-center (pix_theta,pix_phi) in which the transient point lies.

    Inputs:
    depth = The depth of the grid 
    trans_theta = theta of transient (deg)
    trans_phi = phi of transient (deg)

    Returns:
    pix_id, pix_theta (deg) and pix_phi (deg)
    """

    htm_grid = es.htm.HTM(depth=depth)
    pix_id = htm_grid.lookup_id(trans_phi,trans_theta)[0]

    v1,v2,v3 = htm_grid.get_vertices(pix_id)
    center = (v1+v2+v3)/3.0
    center = center/np.sqrt(center[0]**2.0 + + center[1]**2.0 + center[2]**2.0)
    pix_theta =  np.rad2deg(np.arccos(center[2]))
    pix_phi = np.rad2deg(np.arctan2(center[0],center[1]))
    if (pix_phi < 0):
        pix_phi = 360.0 + pix_phi

    return pix_id, pix_theta, pix_phi

# 4. Function to get pix ids of neighbouring pixels

def get_neighbours(pix_theta,pix_phi,theta_arr,phi_arr,sep_angle=20.0):
    """
    Takes a pix_theta and pix_phi as input and gives coordinates of the neighbouring pixels

    Inputs :
    pix_theta = theta for the central pixel (deg)
    pix_phi = phi for the central pixel (deg)
    theta_arr = array of all thetas in the grid (deg)
    phi_arr = array of all phis in the grid (deg)
    sep_angle = angle around the central pixel within which the neighbouring pixels lie (deg)

    Returns:
    2 arrays of neighbouring pixel theta and phi (including central pixel so that these arrays can directly be fed to the sim dph maker)
    """
    center = coo.SkyCoord(pix_phi,pix_theta-90,unit="deg")

    neigh_theta_lst = []
    neigh_phi_lst = []

    for i in range(len(theta_arr)):
        coord = coo.SkyCoord(phi_arr[i],theta_arr[i]-90,unit="deg")
        sep = abs(center.separation(coord).value)
        if (sep <= sep_angle):
            neigh_theta_lst.append(theta_arr[i])
            neigh_phi_lst.append(phi_arr[i])
    
    sel_theta_arr = np.array(neigh_theta_lst)
    sel_phi_arr = np.array(neigh_phi_lst)

    return sel_theta_arr, sel_phi_arr

# 5. Functions to decide the spectrum of the source 

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


def powerlaw(E, alpha=-1.0,E_peak=250.0*u.keV,norm=1.0):
    """
    Returns PHOTONS/s/cm^2/keV from powerlaw using given parameters

    Required inputs:
    E = energy array in keV
    Optional inputs:
    alpha = power law index , default = -1.0
    norm = normalisation constant for the powerlaw , default = 1
    
    Returns:
    Total number of PHOTONS/s/cm^2/keV
    """
    return norm*E**alpha*np.exp(-E/E_peak)

def model(E, alpha=-1.0, beta=-2.5, E_peak=250.0*u.keV, norm=1.0,typ="band"):
    """
    Returns PHOTONS/s/cm^2/keV from band or powerlaw based on input typ

    Required inputs:
    E = energy array (keV)
    Optional inputs:
    alpha = first power law index of the band function, default = -1.0
    beta = second powerlaw index of the band function, default = -2.5
    E_peak = characteristic energy in keV, default = 250  (keV)
    norm = normalisation constant for the band function, default = 1 
    typ = string to set the function to be used band or powerlaw, default = "band" 

    Returns:
    PHOTONS/s/cm^2/keV
    """
    if (typ=="powerlaw"):
        return powerlaw(E,alpha,norm)
    else:
        return band(E,alpha,beta,E_peak,norm)

# 6. Function to calculate the simulated dph

def simulated_dph(grbdir,grid_dir,run_theta,run_phi,typ,t_src,alpha,beta,E0,A):
        """
        Function that creates simulated dph and badpixmap
        from given simulated data and badpix files respectively.
        """
        filenames = glob.glob(grid_dir+ "/T{th:06.2f}_P{ph:06.2f}/*.fits.gz".format(th=run_theta,ph=run_phi))
        badpixfile = glob.glob(grbdir + "/*badpix*.fits")[0]
	print "No. of fits files for  this direction :",len(filenames)
        filenames.sort()
        pix_cnts = np.zeros((16384,len(filenames)))
        err_pix_cnts = np.zeros((16384,len(filenames)))
        en = np.arange(5, 261., .5)
        sel  = (en>=100) & (en <= 150)
        en_range = np.zeros(len(filenames))
        for f in range(len(filenames)):
                #en_range[f] = filenames[f][-31:-24] Older way. Very clumsy if you change the path!!!
                fits_file = fits.open(filenames[f])
                hdr = fits_file[0].header
                en_range[f] = hdr["ENERGY"]
        err_100_500 = (100.0 <= en_range.astype(np.float)) & (en_range.astype(np.float) <= 500.0)
        err_500_1000 = (500.0 < en_range.astype(np.float)) & (en_range.astype(np.float) <= 1000.0)
        err_1000_2000 = (1000.0 < en_range.astype(np.float)) & (en_range.astype(np.float) <= 2000.0)
        exist_1000_2000 = np.where(err_1000_2000 == True)
        E = np.array([])

        print "Indices where energy is in between 1000 and 2000 :",exist_1000_2000[0]

        for i,f in enumerate(filenames):
                        print "---------------------------------------------------------"
			print "Reading file : ",f
                        print "---------------------------------------------------------"
                        data = fits.getdata(f)
                        fits_file = fits.open(f)
                        hdr = fits_file[0].header
                        E = np.append(E,float(hdr["ENERGY"]))
                        # E = np.append(E, float(f[-31:-24])) Older way. Very clumsy if you change the path!!!
                        error = np.sqrt(data)
                        data[:,~sel] = 0.
                        error[:,~sel] = 0.
                        pix_cnts[:,i] = data.sum(1)*model(E[i], alpha, beta, E0, A,typ)/55.5
                        err_pix_cnts[:,i] = np.sqrt(((error*model(E[i], alpha, beta, E0, A,typ)/55.5)**2).sum(1))
	
	print "Energies in the directory are : ",E
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

# 7. Function to calculate the dph from data

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


def data_bkgd_image(grbdir,infile,pre_tstart,pre_tend,grb_tstart,grb_tend,post_tstart,post_tend):
    """
    Creates source and background dph.
    """
    predph = evt2image(infile,pre_tstart,pre_tend)
    grbdph = evt2image(infile,grb_tstart,grb_tend)
    postdph = evt2image(infile,post_tstart,post_tend)

    bkgddph = predph+postdph

    oneD_grbdph = grbdph.flatten()
    oneD_bkgddph = bkgddph.flatten()
    t_src = grb_tend - grb_tstart
    t_total = (pre_tend-pre_tstart)+(post_tend-post_tstart)

    return oneD_grbdph,oneD_bkgddph,grbdph,bkgddph,t_src,t_total


# 8. Function for resampling the dph (module wise or as you please) 

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

# 9. Function to calculate the chi_sq before and after scaling

def fit_line_int(model,scaling,intercept):
    """
    returns a scaled model added to an intercept
    """
    return scaling*model + intercept


def calc_chi_sq(tab_filename,pdf_file,grbdir,grid_dir,sel_theta_arr,sel_phi_arr,typ,t_src,alpha=-1.0,beta=-2.5,E0=250,A=1):
    """
    Calculates chi_sq with and without scaling and writes in txt table. Plots all the sim dphs in the pdf_file.

    Inputs:
    pdf_file = file to which the output has to be written
    grbdir = directory containing the data for the GRB
    grid_dir = directory in which the grid outputs are stored
    sel_theta_arr = array of theta's of the grid points around the GRB location (deg)
    sel_phi_arr = array of phi's of the grid points around the GRB location (deg)
    typ = function type to use for the spectrum "band" or "powerlaw", default = "band"
    t_src = t_90 of the source (s)
    alpha = first power law index of the band function, default = -1.0
    beta = second powerlaw index of the band function, default = -2.5
    E0 = characteristic energy in keV, default = 250  (keV)
    A = normalisation constant for the band function, default = 1
    
    Returns:
    Saves table of theta, phi, chi_sq_wo_sca and chi_sq_w_sca in tab_filename
    """
    # Defining arrays to store direction wise information (chi_sq_wo_sca,chi_sq_sca)
    
    chi_sq_wo_sca_arr = np.zeros(len(sel_theta_arr))
    chi_sq_sca_arr = np.zeros(len(sel_theta_arr))
    scaling_arr = np.zeros(len(sel_theta_arr))
    intercept_arr = np.zeros(len(sel_theta_arr))

    no_dphs = len(sel_theta_arr)
    #fig = plt.figure()
    for loc in range(no_dphs):
        loc_sim_flat,loc_sim_dph,badpix_mask,loc_sim_err_dph = simulated_dph(grbdir,grid_dir,sel_theta_arr[loc],sel_phi_arr[loc],typ,t_src,alpha,beta,E0,A)

        #ax = fig.add_subplot(np.ceil(no_dphs/3.0),3,loc+1)
	#plot_binned_dph(fig,ax,"",loc_sim_dph*badpix_mask,pixbin)
        #ax.set_xticklabels([])
        #ax.set_ylabel(r"$\theta$ = {th:0.1f}, $\phi$ = {ph:0.1f}".format(th=sel_theta_arr[loc],ph=sel_phi_arr[loc]),fontsize=5,rotation=90)
        
	final_sim_dph = loc_sim_dph*badpix_mask
        final_sim_err_dph = loc_sim_err_dph*badpix_mask
        final_grb_dph = grb_dph*badpix_mask
        final_bkgd_dph = bkgd_dph*badpix_mask
        final_grb_err_dph = np.sqrt(grb_dph)*badpix_mask
        final_bkgd_err_dph = np.sqrt(bkgd_dph)*badpix_mask

        sim_bin = resample(final_sim_dph,pixbin)
        sim_err_bin = np.sqrt(resample(final_sim_err_dph**2,pixbin))
        grb_bin = resample(final_grb_dph,pixbin)
        bkgd_bin = resample(final_bkgd_dph,pixbin)
        grb_err_bin = np.sqrt(resample(final_grb_err_dph,pixbin))
        bkgd_err_bin = np.sqrt(resample(final_bkgd_err_dph,pixbin))

        sim_flat_bin = sim_bin.flatten()
        sim_err_flat_bin = sim_err_bin.flatten()
        grb_flat_bin = grb_bin.flatten()
        bkgd_flat_bin = bkgd_bin.flatten()
        grb_err_flat_bin = grb_err_bin.flatten()
        bkgd_err_flat_bin = bkgd_err_bin.flatten()

        # Defining model and data to calculate chi_sq_wo_sca

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
        err_data = np.sqrt(((err_src)**2) + ((err_bkgd)**2)*(t_src/t_tot)**2)
        err_data_copy = np.copy(err_data)

        chi_sq_wo_sca = (((model-data)**2)/((err_model)**2 + (err_data)**2)).sum()

        chi_sq_wo_sca_arr[loc] = chi_sq_wo_sca ################## Yaaaay !!! ####################

	# Calculating the scaling and offset 

        param,pcov = curve_fit(fit_line_int,model_copy,data_copy)
        scaling = param[0]
        intercept = param[1]
	
	scaling_arr[loc] = scaling
	intercept_arr[loc] = intercept

	# Redefining model and data to calculate chi_sq_w_sca

	model_sca = sim_flat_bin*scaling + intercept
        bkgd = bkgd_flat_bin*t_src/t_tot
        src = grb_flat_bin

        data = src - bkgd

        err_src = grb_err_flat_bin
        err_bkgd = bkgd_err_flat_bin
        err_model_sca = sim_err_flat_bin*scaling
        err_data = np.sqrt(((err_src)**2) + ((err_bkgd)**2)*(t_src/t_tot)**2)

        chi_sq_sca = (((model_sca-data)**2)/((err_model_sca)**2 + (err_data)**2)).sum()

	chi_sq_sca_arr[loc] = chi_sq_sca ######################## Yaaaay !!! ######################

    #fig.set_size_inches([6.5,6.5])
    #pdf_file.savefig(fig)

    table_file = open(tab_filename,"w")
    loc_table = Table([np.around(sel_theta_arr,2),np.around(sel_phi_arr,decimals=2),np.around(chi_sq_wo_sca_arr,decimals=2),np.around(scaling_arr,decimals=2),np.around(intercept_arr,decimals=2),np.around(chi_sq_sca_arr,decimals=2)],names=['theta','phi','chi_sq_wo_sca','scaling','intercept','chi_sq_sca'])
    loc_table.write(tab_filename,format="ascii",overwrite=True)
    table_file.close()
    
    return chi_sq_wo_sca_arr,chi_sq_sca_arr

# 10. Plotting functions

def plot_lc(grb_name,clean_file,theta,phi,grb_tstart,grb_tend,pre_tstart,pre_tend,post_tstart,post_tend,pdf_file):
    """
    Plots lightcurves for all the quadrants and highlights the portions taken as the grb and background durations

    Inputs:
    grb_name = name of the trigger
    clean_file = quad_clean file for the relevent orbit
    theta = theta of the grb (deg)
    phi = phi of the grb (deg)
    grb_tstart = time of start of the trigger (mission time s)
    grb_tend = time of end of the trigger (mission time s)
    pre_tstart = time of start of the background before trigger (mission time s)
    pre_tend = time of end of the background before trigger (mission time s)
    post_tstart = time of start of the background after the trigger (mission time s)
    post_tend = time of end of the background after the trigger (mission time s)
    pdf_file = file in which the plot should go

    Retrurns:
    Plots the lightcurves for all the quadrants in the pdf_file
    """

    fig = plt.figure()
    clean_file = fits.open(runconf['infile'])
    plt.title('Light curves for '+grb_name + "\n" + r"$\theta$={t:0.1f} and $\phi$={p:0.1f} ".format(t=theta,p=phi))

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

    return 0

def plot_grid(pdf_file,grb_name,trans_theta,trans_phi,sel_theta_arr,sel_phi_arr,search_radius):
    """
    Plots the grid for which the chi-sq analysis will be done along with the known value of theta phi
    
    Inputs:
    pdf_file = file in which the output is to be written
    grb_name = name of the grb 
    trans_theta = theta of transient (deg)
    trans_phi = phi of transient (deg)
    sel_theta_arr = array of theta's of pixels around the given direction (deg)
    sel_phi_arr = array of phi's of pixels around the given direction (deg)
    search_radius = radius (deg) around the central pixel to do the analysis

    Returns:
    saves figure containing the grid to the pdf_file
    """
    fig = plt.figure()
    grid_map = Basemap(projection="ortho",llcrnrx=-2.5*search_radius*10**5,llcrnry=-2.5*search_radius*10**5,urcrnrx=2.5*search_radius*10**5,urcrnry=2.5*search_radius*10**5, lat_0=90-trans_theta,lon_0=trans_phi)
    
    x_trans, y_trans = grid_map(trans_phi, 90-trans_theta)
    grid_map.plot(x_trans,y_trans,"k+")
    plt.text(x_trans,y_trans,grb_name)

    x_neigh, y_neigh = grid_map(sel_phi_arr, 90 - sel_theta_arr)
    grid_map.plot(x_neigh, y_neigh, "C0o")
    grid_map.drawparallels(np.arange(-90,90,30), lables=np.arange(180,0,-30))
    grid_map.drawmeridians(np.arange(0,360,30), lables=np.arange(0,360,30))
    plt.title("The simulation grid for "+grb_name+ " (r = {r:0.1f})".format(r=search_radius))
    
    fig.set_size_inches([6,6])
    pdf_file.savefig(fig)
    
    return 0

def plot_binned_dph(fig,ax,ax_title,image,pixbin):
    """
    Plots a dph of a given binning
    
    Inputs:
    fig = figure to which the ax object belongs
    ax = axis object
    ax_title = title required 
    image = the image to be plotted 
    pixbin = the resampling bin size 
    """
    im = ax.imshow(resample(image,pixbin),interpolation='none')
    ax.set_title(ax_title,fontsize=8)
    ax.set_xlim(-1,128/pixbin - 0.5)
    ax.axvline(x=-0.75,ymin=0,ymax=64,linewidth=5,color='k')
    ax.spines['left'].set_position(('data',-0.5))
    ax.set_yticklabels([])
    ax.xaxis.set_ticks(np.arange(0,128/pixbin,16/pixbin))
    ax.set_xticklabels(np.arange(0,128,16))
    cb = fig.colorbar(im,ax=ax,fraction=0.046, pad=0.04)
    cb.ax.tick_params(labelsize=8)

    return 0

def plot_sim_data_dphs(pdf_file,grb_name,trans_theta,trans_phi,pix_theta,pix_phi,sim_dph,grb_dph,bkgd_dph,badpix_mask,t_src,t_tot,pixbin):
    """
    Plots the binned dphs for the src + bkgd, bkgd , src (bkgd subtracted) and the sim dph (of the pixel in which 
    the transient lies
    
    Inputs:
    pdf_file = file in which the output is to be written
    grb_name = name of grb
    trans_theta = theta of transient (deg)
    trans_phi = phi of transient (deg)
    pix_theta = theta coordinate of center of the pixel containing the transient (deg)
    pix_phi = phi coordinate of the center of the pixel containing the transient (deg)
    sim_dph = dph from simulation at (pix_theta,pix_phi)
    grb_dph = dph of src + bkgd 
    bkgd_dph = only bkgd
    badpix_mask = badpix_mask to be multiplied 
    t_src = duration of the burst (t90) (s)
    t_tot = total background time to consider (s)
    pixbin = binsize to resample the dphs 

    Returns:
    saves the figure (containing all the dphs in the pdf_file)
    """
    # We need to first do the badpix correction 
    
    sim_dph_corr = sim_dph*badpix_mask
    grb_dph_corr = grb_dph*badpix_mask
    src_dph_corr = (grb_dph - bkgd_dph*t_src/t_tot)*badpix_mask
    bkgd_dph_corr = bkgd_dph*badpix_mask*t_src/t_tot
	
    total_counts = src_dph.sum()     

    f,((ax1,ax2),(ax3,ax4)) = plt.subplots(2,2)
    plt.suptitle('DPHs after badpix correction for '+grb_name + r" $\theta_{{grb}}$={tg:0.1f} and $\phi_{{grb}}$={pg:0.1f}".format(tg=trans_theta,pg=trans_phi)+"\n"+r"Pixel (for simulated dph) : $\theta$={t:0.1f} and $\phi$={p:0.1f}i".format(t=pix_theta,p=pix_phi)+"\n"+r"Total observed counts={c:0.2f}".format(c=total_counts))
            
            # Source + Background	
    plot_binned_dph(f,ax1,"Src + Bkgd DPH",grb_dph_corr,pixbin)
            # Background
    plot_binned_dph(f,ax2,"Bkgd DPH",bkgd_dph_corr,pixbin) 
            # Sim
    plot_binned_dph(f,ax3,"Sim DPH",sim_dph_corr,pixbin)
            # Source 
    plot_binned_dph(f,ax4,"Src DPH (bkgd subtracted)",src_dph_corr,pixbin)

    f.set_size_inches([6.5,6.5])
    pdf_file.savefig(f)  # saves the current figure into a pdf_file page
    return 0

def plot_all_sim_dphs(pdf_file,grbdir,grid_dir,sel_theta_arr,sel_phi_arr,typ,t_src,alpha=-1.0,beta=-2.5,E0=250,A=1):
    """
    Plots all the simulation dphs at all the points of the grid, which is the output of the plot_grid function

    Inputs:
    pdf_file = file to which the output has to be written
    grbdir = directory containing the data for the GRB
    grid_dir = directory in which the grid outputs are stored
    sel_theta_arr = array of theta's of the grid points around the GRB location (deg)
    sel_phi_arr = array of phi's of the grid points around the GRB location (deg)
    typ = function type to use for the spectrum "band" or "powerlaw", default = "band"
    t_src = t_90 of the source (s)
    alpha = first power law index of the band function, default = -1.0
    beta = second powerlaw index of the band function, default = -2.5
    E0 = characteristic energy in keV, default = 250  (keV)
    A = normalisation constant for the band function, default = 1
    
    Returns:
    Saves the figure with all the dph's in the pdf_file 
    """
    no_dphs = len(sel_theta_arr)
    fig = plt.figure()
    for loc in range(no_dphs):
        loc_sim_flat,loc_sim_dph,badpix_mask,loc_sim_err_dph = simulated_dph(grbdir,grid_dir,sel_theta_arr[loc],sel_phi_arr[loc],typ,t_src,alpha,beta,E0,A)

        ax = fig.add_subplot(np.ceil(no_dphs/3.0),3,loc+1)

        plot_binned_dph(fig,ax,"",loc_sim_dph*badpix_mask,pixbin)
        ax.set_xticklabels([])
        ax.set_ylabel(r"$\theta$ = {th:0.1f}, $\phi$ = {ph:0.1f}".format(th=sel_theta_arr[loc],ph=sel_phi_arr[loc]),fontsize=5,rotation=90)
    
    fig.set_size_inches([6.5,6.5])
    pdf_file.savefig(fig)
    
    return 0

def plot_loc_contour(grb_name,pdf_file,trans_theta,trans_phi,sel_theta_arr,sel_phi_arr,chi_sq_wo_sca_arr,chi_sq_sca_arr,search_radius):
    """
    Plots the 3d and contour plots for the localisation from the calculated chi_sq values at the selected locations

    Inputs :
    grb_name = name of the trigger
    pdf_file = file to which the plot should be saved
    trans_theta = theta of the grb (deg)
    trans_phi = phi of the grb (deg)
    sel_theta_arr = array of theta's for the points on the grid (deg)
    sel_phi_arr = array of phi's for the points on the grid (deg)
    chi_sq_wo_sca_arr = array containing the corresponding chi_sq values without scaling the sim 
    chi_sq_sca_arr = array containing the corresponding chi_sq values with scaling and intercept applied to the sim

    Returns :
    Saves the plot into the pdf_file
    """
    fig = plt.figure()

    plt.suptitle(r"$\chi^2$ plots for "+grb_name+"; Left: Without scaling, Right: With scaling")

## For flat plots
#    X = np.linspace(sel_theta_arr.min(),sel_theta_arr.max(),100)
#    Y = np.linspace(sel_phi_arr.min(),sel_phi_arr.max(),100)
    
#    Xi, Yi = np.meshgrid(X,Y)
#    Z1 = griddata((sel_theta_arr,sel_phi_arr),chi_sq_wo_sca_arr,(X[None,:],Y[:,None]),method="cubic")
#    Z2 = griddata((sel_theta_arr,sel_phi_arr),chi_sq_sca_arr,(X[None,:],Y[:,None]),method="cubic")

    X = np.linspace(sel_phi_arr.min(),sel_phi_arr.max(),100)
    Y = np.linspace(90 - sel_theta_arr.min(),90- sel_theta_arr.max(),100)

    Xi, Yi = np.meshgrid(X,Y)
    Z1 = griddata((sel_phi_arr,90-sel_theta_arr),chi_sq_wo_sca_arr,(X[None,:],Y[:,None]),method="cubic")
    Z2 = griddata((sel_phi_arr,90-sel_theta_arr),chi_sq_sca_arr,(X[None,:],Y[:,None]),method="cubic")
   
    sca_1 = np.nanmin(Z1)/62.0
    sca_2 = np.nanmin(Z2)/60.0

    mesh_grid_pix_area = (X[1]-X[0])*(Y[1]-Y[0])

    cont_pix_no = 0

    for i in range(len(X)):
	for j in range(len(Y)):
	    if (Z2[i,j] <= 74.397*sca_2):
		cont_pix_no +=1

    cont_area = cont_pix_no * mesh_grid_pix_area

    print "Area of the 90% contour is : ", abs(cont_area)

    
    ax3 = fig.add_subplot(221)
    #ax3.set_xlabel(r"$\theta [in^{o}]$")
    #ax3.set_ylabel(r"$\phi [in^{o}]$",rotation=90)
    #ax3.contour(Xi,Yi,Z1,[np.nanmin(Z1)+1*sca_1,np.nanmin(Z1)+2*sca_1,np.nanmin(Z1)+3*sca_1],colors=["C0","C1","C2"],linewidths=0.75)
    #grb = ax3.scatter(trans_theta,trans_phi,marker='+',color="k",label=grb_name)
    #ax3.set_xlim(trans_theta-45,trans_theta+45)
    #ax3.set_ylim(trans_phi-45,trans_phi+45)
    map = Basemap(projection="ortho",llcrnrx=-2.5*search_radius*10**5,llcrnry=-2.5*search_radius*10**5,urcrnrx=2.5*search_radius*10**5,urcrnry=2.5*search_radius*10**5, lat_0=90-trans_theta,lon_0=trans_phi)
    map_Xi, map_Yi = map(Xi,Yi)
    map.contour(map_Xi,map_Yi,Z1,[np.nanmin(Z1)+1*sca_1,np.nanmin(Z1)+2*sca_1,np.nanmin(Z1)+3*sca_1],colors=["C0","C1","C2"],linewidths=0.75)
    x_trans, y_trans = map(trans_phi, 90-trans_theta)
    grb = map.plot(x_trans,y_trans,"k+")
    map.drawmeridians(np.arange(0,360,30), lables=np.arange(0,360,30)) 
    map.drawparallels(np.arange(-90,90,30), lables=np.arange(180,0,-30))

    line1= pl.Line2D(range(10), range(10), marker='None', linestyle='-',linewidth=0.75, color="C0")
    line2= pl.Line2D(range(10), range(10), marker='None', linestyle='-',linewidth=0.75, color="C1")
    line3= pl.Line2D(range(10), range(10), marker='None', linestyle='-',linewidth=0.75, color="C2")
    ax3.legend(("k+",line1,line2,line3),(grb_name,r"1-$\sigma$",r"2-$\sigma$",r"3-$\sigma$"),numpoints=1,loc='upper right',prop={'size':6})

    ax4 = fig.add_subplot(222)
    #ax4.set_xlabel(r"$\theta [in^{o}]$")
    #ax4.contour(Xi,Yi, Z2, [np.nanmin(Z2)+1*sca_2,np.nanmin(Z2)+2*sca_2,np.nanmin(Z2)+3*sca_2],colors=["C0","C1","C2"],linewidths=0.75)
    #grb = ax4.scatter(trans_theta,trans_phi,marker='+',color="k",label=grb_name)
    #ax4.set_xlim(trans_theta-45,trans_theta+45)
    #ax4.set_ylim(trans_phi-45,trans_phi+45)
    map = Basemap(projection="ortho",llcrnrx=-2.5*search_radius*10**5,llcrnry=-2.5*search_radius*10**5,urcrnrx=2.5*search_radius*10**5,urcrnry=2.5*search_radius*10**5, lat_0=90-trans_theta,lon_0=trans_phi)
    map_Xi, map_Yi = map(Xi,Yi)
    map.contour(map_Xi,map_Yi,Z2,[np.nanmin(Z2)+1*sca_2,np.nanmin(Z2)+2*sca_2,np.nanmin(Z2)+3*sca_2],colors=["C0","C1","C2"],linewidths=0.75)
    x_trans, y_trans = map(trans_phi, 90-trans_theta)
    grb = map.plot(x_trans,y_trans,"k+")
    map.drawmeridians(np.arange(0,360,30), lables=np.arange(0,360,30)) 
    map.drawparallels(np.arange(-90,90,30), lables=np.arange(180,0,-30))

    line1= pl.Line2D(range(10), range(10), marker='None', linestyle='-',linewidth=0.75, color="C0")
    line2= pl.Line2D(range(10), range(10), marker='None', linestyle='-',linewidth=0.75, color="C1")
    line3= pl.Line2D(range(10), range(10), marker='None', linestyle='-',linewidth=0.75, color="C2")
    ax4.legend(("k+",line1,line2,line3),(grb_name,r"1-$\sigma$",r"2-$\sigma$",r"3-$\sigma$"),numpoints=1,loc='upper right',prop={'size':6})
    
    ax1 = fig.add_subplot(223)
    #ax1.set_xlabel(r"$\theta [in^{o}]$")
    #ax1.set_ylabel(r"$\phi [in^{o}]$",rotation=90)
    #ax1.contour(Xi, Yi, Z1, [74.397*sca_1,88.379*sca_1],colors=["C0","C1"],linewidths=0.75)
    #grb = ax1.scatter(trans_theta,trans_phi,marker='+',color="k",label=grb_name)
    #ax1.set_xlim(trans_theta-45,trans_theta+45)
    #ax1.set_ylim(trans_phi-45,trans_phi+45)

    map = Basemap(projection="ortho",llcrnrx=-2.5*search_radius*10**5,llcrnry=-2.5*search_radius*10**5,urcrnrx=2.5*search_radius*10**5,urcrnry=2.5*search_radius*10**5, lat_0=90-trans_theta,lon_0=trans_phi)
    map_Xi, map_Yi = map(Xi,Yi)
    map.contour(map_Xi,map_Yi,Z1,[74.397*sca_1,88.379*sca_1],colors=["C0","C1"],linewidths=0.75)
    x_trans, y_trans = map(trans_phi, 90-trans_theta)
    map.plot(x_trans,y_trans,"k+")
    map.drawmeridians(np.arange(0,360,30), lables=np.arange(0,360,30))
    map.drawparallels(np.arange(-90,90,30), lables=np.arange(180,0,-30))

    line1= pl.Line2D(range(10), range(10), marker='None', linestyle='-',linewidth=0.75, color="C0")
    line2= pl.Line2D(range(10), range(10), marker='None', linestyle='-',linewidth=0.75, color="C1")
    ax1.legend(("k+",line1,line2),(grb_name,"90 %","99 %"),numpoints=1,loc='upper right',prop={'size':6})

    ax2 = fig.add_subplot(224)
    #ax2.set_xlabel(r"$\theta [in^{o}]$")
    #cs = ax2.contour(Xi,Yi, Z2, [74.397*sca_2,88.379*sca_2],colors=["C0","C1"],linewidths=0.75)
    #grb = ax2.scatter(trans_theta,trans_phi,marker='+',color="k",label=grb_name)
    #ax2.set_xlim(trans_theta-45,trans_theta+45)
    #ax2.set_ylim(trans_phi-45,trans_phi+45)
    
    map = Basemap(projection="ortho",llcrnrx=-2.5*search_radius*10**5,llcrnry=-2.5*search_radius*10**5,urcrnrx=2.5*search_radius*10**5,urcrnry=2.5*search_radius*10**5, lat_0=90-trans_theta,lon_0=trans_phi)
    map_Xi, map_Yi = map(Xi,Yi)
    map.contour(map_Xi,map_Yi,Z2,[74.397*sca_2,88.379*sca_2],colors=["C0","C1"],linewidths=0.75)
    x_trans, y_trans = map(trans_phi, 90-trans_theta)
    grb = map.plot(x_trans,y_trans,"k+")
    map.drawmeridians(np.arange(0,360,30), lables=np.arange(0,360,30)) 
    map.drawparallels(np.arange(-90,90,30), lables=np.arange(180,0,-30))

    line1= pl.Line2D(range(10), range(10), marker='None', linestyle='-',linewidth=0.75, color="C0")
    line2= pl.Line2D(range(10), range(10), marker='None', linestyle='-',linewidth=0.75, color="C1")
    ax2.legend(("k+",line1,line2),(grb_name,"90 %","99 %"),numpoints=1,loc="upper right",prop={'size':6})
    ax2.text(x_trans,y_trans,r"{a:0.2f} deg$^{{2}}$".format(a=cont_area))
    
    fig.set_size_inches([8,8])
    pdf_file.savefig(fig)

    return 0


##################################### Main function begins #########################################

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("configfile", nargs="?", help="Path to configfile", type=str)
    parser.add_argument("radius", help="Radius (deg) around the central pixel to do chi_sq analysis",type=int)
    parser.add_argument("--calc_chi_sq_bool", help="Boolean to decide whether to do chi_sq analysis or not",type=bool)
    parser.add_argument('--noloc', dest='noloc', action='store_true')
    parser.set_defaults(calc_chi_sq_bool=True,noloc=False)
    args = parser.parse_args()
    runconf = get_configuration(args)

    # Getting all the info from the configfile. Will also print config for this run.

    grb_name = runconf['name']
    mkffile = runconf['mkffile']
    infile = runconf['infile']
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
    plotfile = runconf['plotfile']
    sim_scale = t_src
    pixbin = int(runconf['pixsize'])
    comp_bin = int(runconf['comp_bin'])
    typ = runconf['typ']
    imsize = 128/runconf['pixsize']

    path_to_conf = args.configfile.split("/") # Splitting to get the location of the data

    path_to_data = ""

    for i in range(1, len(path_to_conf)-1):
  	path_to_data += "/"+path_to_conf[i]

    grbdir = path_to_data
    
    path_to_plotfile = plotfile.split("/")[0]
    print path_to_plotfile



    grid_dir = "/mnt/nas_czti/massmodelgrid"
    depth = 4 # For now hard coded this. Let's see if we can make this better later
    
    pdf_file = PdfPages(plotfile)
    loc_txt_file = path_to_plotfile+"/"+grb_name+"_loc_table.txt"
    print "======================================================================================"

    # R eading files containing arrays of theta and phi for the grid

    theta_tab = Table.read("final_theta.txt",format="ascii")
    theta_arr = theta_tab["theta"].data
    phi_tab = Table.read("final_phi.txt", format="ascii")
    phi_arr = phi_tab["phi"].data
    
    # Converting the (RA,DEC) coordinates to (theta,phi).

    trans_theta, trans_phi, thetax,thetay, czti_x, czti_y, czti_z, transient, earth = get_angles(mkffile, trigtime, ra_tran, dec_tran, window=10)

    print "The coordinates of the transient in CZTI frame are : THETA = ",trans_theta,", PHI = ",trans_phi

    pix_id, pix_theta, pix_phi = get_center(depth,90-trans_theta,90-trans_phi)

    #print "The pixel in which the transient lies is : ",pix_id,"(for depth = ",depth,")" 
    print "The coordinates of the pixel are : THETA = ",pix_theta,", PHI = ",pix_phi
    
    sel_theta_arr, sel_phi_arr = get_neighbours(pix_theta,pix_phi,theta_arr,phi_arr,sep_angle=args.radius)

    print "The theta's of the selected pixels : ",sel_theta_arr
    print "The phi's of the selected pixels : ",sel_phi_arr

    # Plotting the lightcurves for all quadrants 

    plot_lc(grb_name,infile,trans_theta,trans_phi,grb_tstart,grb_tend,pre_tstart,pre_tend,post_tstart,post_tend,pdf_file)
    
    # Plotting the grid points around the known position of the transient

    plot_grid(pdf_file,grb_name,trans_theta,trans_phi,sel_theta_arr,sel_phi_arr,args.radius)

    # Now we have all the points. So, we can move on to calculating the dphs for all these points
    print "========================================================================================"

    # Calling the function to get the data_dph 
    flat_grb_dph,flat_bkgd_dph,grb_dph,bkgd_dph,t_src,t_tot = data_bkgd_image(grbdir,infile,pre_tstart,pre_tend,grb_tstart,grb_tend,post_tstart,post_tend)
    src_dph = grb_dph - bkgd_dph*t_src/t_tot

    sim_flat,sim_dph,badpix_mask,sim_err_dph = simulated_dph(grbdir,grid_dir,pix_theta,pix_phi,typ,t_src,alpha,beta,E0,A)

    # Plotting the badpix corrected dphs 
    
    plot_sim_data_dphs(pdf_file,grb_name,trans_theta,trans_phi,pix_theta,pix_phi,sim_dph,grb_dph,bkgd_dph,badpix_mask,t_src,t_tot,pixbin)
    
    print "========================================================================================"

    # Calculating chi_sq before and after scaling
    
    ##chi_sq_wo_sca_arr, chi_sq_sca_arr = calc_chi_sq(loc_txt_file,pdf_file,grbdir,grid_dir,sel_theta_arr,sel_phi_arr,typ,t_src,alpha,beta,E0,A)
    
    print "========================================================================================"

    # Plotting the contour plots (The data is read out from the files)
    
    tab = Table.read(loc_txt_file,format='ascii')

    chi_sq_wo_sca_arr = tab['chi_sq_wo_sca'].data
    chi_sq_sca_arr = tab['chi_sq_sca'].data
    scaling_arr = tab['scaling'].data
    intercept_arr = tab['intercept'].data

    avg_scaling = np.sum(scaling_arr)/len(scaling_arr)
    avg_intercept = np.sum(intercept_arr)/len(intercept_arr)


    plot_loc_contour(grb_name,pdf_file,trans_theta,trans_phi,sel_theta_arr,sel_phi_arr,chi_sq_wo_sca_arr,chi_sq_sca_arr,args.radius)

    t1 = time.time()

    t_final = t1-t0
    print "Time taken for the code to run (s) : ",t_final    
