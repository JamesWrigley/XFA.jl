#!/usr/bin/env python

"""Generate a detector-assembly lookup table.

The geometry comes from a CrystFEL .geom file, or for detectors with a fixed
single-module layout it can be built without one: ePix100 uses its two-sensor
pair geometry, and JUNGFRAU with --n_modules 1 uses a single module at the
origin.

The LUT maps each input pixel of a single detector frame to the flat index of
the pixel it occupies in the assembled 2D image, using EXtra-geom's own
(snapped) assembly. The LUT uses 1-based indexes so it can be consumed by Julia.

Output is written as raw bytes to stdout:
  int64  nx
  int64  ny
  uint64 lut[n_modules * slow_scan * fast_scan]
"""

import argparse
import sys

import numpy as np
import extra_geom


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("geom_file", nargs="?",
                        help="path to the CrystFEL .geom file; optional for "
                             "detectors with a fixed single-module layout")
    parser.add_argument("--detector", default="AGIPD_1MGeometry",
                        help="EXtra-geom geometry class (default: AGIPD_1MGeometry)")
    parser.add_argument("--n_modules", type=int, default=1,
                        help="number of modules, for JUNGFRAU without a geom file")
    args = parser.parse_args()

    geom_cls = getattr(extra_geom, args.detector)
    if args.geom_file is not None:
        geom = geom_cls.from_crystfel_geom(args.geom_file)
    elif args.detector == "Epix100Geometry":
        geom = geom_cls.pair_geometry()
    elif args.detector == "JUNGFRAUGeometry" and args.n_modules == 1:
        geom = geom_cls.from_module_positions()
    else:
        parser.error(f"a geom file is required for {args.detector}")

    nmod, nss, nfs = geom.expected_data_shape
    npix = nmod * nss * nfs

    # Assemble an array of flat indices: the value left at each output pixel is
    # the input pixel that landed there (gaps stay NaN). Invert that into an
    # input -> output map keyed by the input's flat index.
    idx = np.arange(npix, dtype=np.float64).reshape(nmod, nss, nfs)
    assembled, _ = geom.position_modules_fast(idx)
    ny, nx = assembled.shape

    flat = assembled.ravel()
    filled = ~np.isnan(flat)
    out_pos = np.nonzero(filled)[0]                   # output C-flat indices
    in_idx = flat[filled].astype(np.int64)            # input C-flat index per output
    lut = np.empty(npix, dtype=np.uint64)
    lut[in_idx] = out_pos + 1                         # 1-based for Julia

    out = sys.stdout.buffer
    out.write(np.array([nx, ny], dtype=np.int64).tobytes())
    out.write(lut.tobytes())
    out.flush()


if __name__ == "__main__":
    main()
