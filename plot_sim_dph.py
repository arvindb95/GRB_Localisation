import numpy as np
import matplotlib.pyplot as plt


from trans_loc import *

theta_tab = Table.read("new_final_theta.txt",format="ascii")
theta_arr = theta_tab["theta"].data
phi_tab = Table.read("new_final_phi.txt", format="ascii")
phi_arr = phi_tab["phi"].data

soft_alpha = -0.7
soft_beta = -3.2
hard_alpha = -0.5
hard_beta = -1.8
E0 = 120
A = 1
t_src = 10
e_low = 20
e_high = 200
typ = "band"


grbdir="./"

grid_dir = "/media/arvind/Elements/5th_Year_Project/massmodelgrid"

theta = theta_arr[1]

phi = phi_arr[1]

fig = plt.figure()
ax = fig.add_subplot(111)

sim_flat,sim_dph,badpix_mask,sim_err_dph = simulated_dph(grbdir,grid_dir,theta,phi,typ,t_src,soft_alpha,soft_beta,E0,A,e_low,e_high)

plot_binned_dph(fig,ax,"Mass Model DPH for theta = {t:0.2f} phi = {p:0.2f} alpha = {a:0.2f} , beta = {b:0.2f} , epeak = {e:0.2f} ".format(t=theta,p=phi,a=soft_alpha,b=soft_beta,e=E0),sim_dph,16,cm.viridis)

plt.savefig("T_{t:0.2f}P_{p:0.2f}a_{a:0.2f}b_{b:0.2f}e_{e:0.2f}.png".format(t=theta,p=phi,a=soft_alpha,b=soft_beta,e=E0))

plt.show()
