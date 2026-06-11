import os
import sys
import time

import numpy as np
import rasterio
from rasterio.windows import Window

sys.path.insert(0, os.getcwd())

from data_loader.date_loader import load_and_clean_data
from data_loader.rainfall_loader import build_chirps_cog_url, chirps_crop_window, infer_bounds


def main():
    _, coords, _, _, _ = load_and_clean_data("dataset/inter228_5241.csv")
    bounds = infer_bounds(coords, padding=0.0)
    crop_bounds = (
        bounds[0] - 0.25,
        bounds[1] + 0.25,
        bounds[2] - 0.25,
        bounds[3] + 0.25,
    )
    r0, r1, c0, c1 = chirps_crop_window(crop_bounds)
    url = build_chirps_cog_url("2017-02-07")
    print(f"url={url}")
    print(f"window={(r0, r1, c0, c1)} size={(r1 - r0, c1 - c0)}")
    start = time.time()
    with rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR", CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".cog"):
        with rasterio.open(url) as ds:
            print(f"dataset width={ds.width} height={ds.height} count={ds.count} dtype={ds.dtypes[0]}")
            data = ds.read(1, window=Window(c0, r0, c1 - c0, r1 - r0)).astype(np.float32)
    elapsed = time.time() - start
    data[data < -9000] = np.nan
    print(
        f"elapsed={elapsed:.2f}s shape={data.shape} "
        f"min={float(np.nanmin(data)):.3f} max={float(np.nanmax(data)):.3f} "
        f"mean={float(np.nanmean(data)):.3f}"
    )


if __name__ == "__main__":
    main()
