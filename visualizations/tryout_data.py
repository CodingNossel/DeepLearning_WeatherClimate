from pylab import *
import h5netcdf

a = h5netcdf.File('../rawdata/openmars_my34_ls342_my34_ls358.nc', 'r')
lats = a.variables['lat'][:]  # Latitude
lons = a.variables['lon'][:]  # Longitude
temp = a.variables['temp'][:, 10, :, :]  # Water vapour vmr at eleventh model level

fig = plt.figure()
CS1 = plt.contourf(lons, lats, temp[719, :, :], 30, cmap=plt.cm.Blues)
cbar = plt.colorbar(CS1)
cbar.set_label('Water vapour / ppmv', fontsize=20)
for t in cbar.ax.get_yticklabels():
    t.set_fontsize(18)
plt.xticks(arange(-135, 180, 45),
           ('135$^\circ$W', '90$^\circ$W', '45$^\circ$W', '0$^\circ$', '45$^\circ$E', '90$^\circ$E', '135$^\circ$E'),
           fontsize=16)
plt.xlabel(r'Longitude', fontsize=18)
plt.yticks(arange(-60, 90, 30), ('60$^\circ$S', '30$^\circ$S', '0$^\circ$', '30$^\circ$N', '60$^\circ$N'), fontsize=16)
plt.ylabel('Latitude', fontsize=18)
plt.axis([-180., 175., -87.5, 87.5])
fig.set_size_inches(14, 8)
plt.savefig('my34_ls176_temp.png', dpi=200, bbox_inches='tight', pad_inches=0.3)
plt.close()
