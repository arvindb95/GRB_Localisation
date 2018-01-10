import numpy as np
import ConfigParser, argparse
from matplotlib.backends.backend_pdf import PdfPages
import astropy.units as u
from scipy.integrate import quad

from offaxispos import powerlaw
from offaxispos import band
from offaxispos import model
from offaxispos import simulated_dph
from offaxispos import data_bkgd_image
from offaxispos import get_configuration
from offaxispos import resample

parser = argparse.ArgumentParser()
parser.add_argument("configfile", nargs="?", help="Name of configuration file", type=str)
parser.add_argument("emin",help="Min energy to start integration for calculating norm",type=float)
parser.add_argument("emax",help="Max energy to start integration for calculating norm",type=float)
parser.add_argument("fluence",help="Fluence for calculating norm",type=float)
parser.add_argument('--noloc', dest='noloc', action='store_true')
parser.set_defaults(noloc=False)
args = parser.parse_args()
runconf = get_configuration(args)

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
sim_scale = t_src
pixbin = int(runconf['pixsize'])
comp_bin = int(runconf['comp_bin'])
typ = runconf['typ']
plotfile = runconf['plotfile']


def f(E,alpha,beta,E0,A,typ):
	return model(E,alpha,beta,E0,A,typ)*E

emin = args.emin
emax = args.emax
fluence = args.fluence

I = quad(f,emin,emax,args=(alpha,beta,E0,1,typ))[0]*t_src*u.keV.to(u.erg)

A = fluence/I
print "================================================="
print "The Norm for "+grbdir+" is ",A
print "=================================================="
