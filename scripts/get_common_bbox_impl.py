import glob
from osgeo import gdal
import numpy as np

files = sorted(glob.glob("sbas/*/*_dem.tif"))
if not files:
    files = sorted(glob.glob("sbas/*/*_unw_phase.tif"))

print(f"共找到 {len(files)} 个影像")

bounds = []
for f in files:
    ds = gdal.Open(f)
    gt = ds.GetGeoTransform()
    xsize = ds.RasterXSize
    ysize = ds.RasterYSize
    minx = gt[0]
    maxy = gt[3]
    maxx = minx + gt[1] * xsize
    miny = maxy + gt[5] * ysize
    bounds.append([miny, maxy, minx, maxx])

bounds = np.array(bounds)
common_S = bounds[:,0].max()
common_N = bounds[:,1].min()
common_W = bounds[:,2].max()
common_E = bounds[:,3].min()

print(f"\n✅ 公共重叠区域：")
print(f"mintpy.subset.lalo = {common_S:.6f}:{common_N:.6f},{common_W:.6f}:{common_E:.6f}")
