#!/usr/bin/env python3
"""
ParaView single-frame visualization for 3D nematic data with disclination lines.

Usage (script arguments go after '--'):

  # Interactive — open in ParaView GUI:
  paraview --script=visualize_frame_in_paraview.py -- --frame 42

  # Headless — save a PNG:
  pvpython visualize_frame_in_paraview.py -- --frame 42 --save

  # Last available frame (default):
  pvpython visualize_frame_in_paraview.py -- --save

Requires ParaView >= 5.11 for VTKHDF reader support.
"""

import argparse
import glob
import os
import sys

from paraview.simple import (
    Calculator, ComputeDerivatives, GetActiveViewOrCreate,
    GetColorTransferFunction,
    Glyph, Hide, IsoVolume, OpenDataFile,
    RenameSource, Render, ResetCamera,
    SaveScreenshot, Show, Slice, ColorBy,
)


# ===========================================================================
#  Configuration — adjust to match your simulation
# ===========================================================================

DATA_DIR   = './data'
OUTPUT_DIR = os.path.join(DATA_DIR, 'frames')

NX, NY, NZ = 64, 32, 32
LX, LY, LZ = NX, NY, NZ
ORIGIN      = (0.0, 0.0, 0.0)

MID_PT_X = ORIGIN[0] + LX / 2
MID_PT_Y = ORIGIN[1] + LY / 2
MID_PT_Z = ORIGIN[2] + LZ / 2
MID_PT   = [MID_PT_X, MID_PT_Y, MID_PT_Z]

GLYPH_STRIDE = 4
DEFECT_S_MAX = 0.2
S_RANGE      = [0.0, 0.7]

IMAGE_WIDTH  = 1920
IMAGE_HEIGHT = 1080

# ===========================================================================


def build_pipeline(reader):
    vel = Calculator(Input=reader)
    vel.AttributeType   = 'Point Data'
    vel.Function        = 'iHat*ux + jHat*uy + kHat*uz'
    vel.ResultArrayName = 'velocity'
    RenameSource('velocity', vel)

    deriv = ComputeDerivatives(Input=vel)
    deriv.Vectors          = ['POINTS', 'velocity']
    deriv.OutputVectorType = 'Vorticity'
    deriv.OutputTensorType = 'Nothing'
    RenameSource('vorticity_pipeline', deriv)

    slice_yz = Slice(Input=deriv)
    slice_yz.SliceType        = 'Plane'
    slice_yz.SliceType.Normal = [1, 0, 0]
    slice_yz.SliceType.Origin = MID_PT
    RenameSource('slice_YZ', slice_yz)

    slice_xz = Slice(Input=deriv)
    slice_xz.SliceType        = 'Plane'
    slice_xz.SliceType.Normal = [0, 1, 0]
    slice_xz.SliceType.Origin = MID_PT
    RenameSource('slice_XZ', slice_xz)

    glyph_v_yz = Glyph(Input=slice_yz, GlyphType='Arrow')
    glyph_v_yz.OrientationArray = ['POINTS', 'velocity']
    glyph_v_yz.ScaleFactor      = (LX / NX) * GLYPH_STRIDE * 0.9
    glyph_v_yz.GlyphMode        = 'Uniform Spatial Distribution (Surface Sampling)'
    glyph_v_yz.ScaleArray       = ['POINTS', 'velocity']
    RenameSource('velocity_glyph_slice_yz', glyph_v_yz)

    glyph_v_xz = Glyph(Input=slice_xz, GlyphType='Arrow')
    glyph_v_xz.OrientationArray = ['POINTS', 'velocity']
    glyph_v_xz.ScaleFactor      = (LX / NX) * GLYPH_STRIDE * 0.9
    glyph_v_xz.GlyphMode        = 'Uniform Spatial Distribution (Surface Sampling)'
    glyph_v_xz.ScaleArray       = ['POINTS', 'velocity']
    RenameSource('velocity_glyph_slice_xz', glyph_v_xz)

    glyph = Glyph(Input=reader, GlyphType='Line')
    glyph.OrientationArray = ['POINTS', 'director']
    glyph.ScaleFactor      = (LX / NX) * GLYPH_STRIDE * 0.9
    glyph.GlyphMode        = 'Uniform Spatial Distribution (Volume Sampling)'
    RenameSource('director_glyph', glyph)

    defects = IsoVolume(Input=reader)
    defects.InputScalars   = ['POINTS', 'order']
    defects.ThresholdRange = [0.0, DEFECT_S_MAX]
    RenameSource('defects', defects)

    for f in [deriv, slice_yz, slice_xz, glyph, defects]:
        f.UpdatePipeline()

    return {
        'deriv':        deriv,
        'slice_yz':     slice_yz,
        'slice_xz':     slice_xz,
        'glyph':        glyph,
        'glyph_v_yz':   glyph_v_yz,
        'glyph_v_xz':   glyph_v_xz,
        'defects':      defects,
    }


def setup_display(pipeline, reader, view, disc_reader=None):
    disp_reader = Show(reader, view)
    disp_reader.Representation = 'Outline'

    disp_yz = Show(pipeline['slice_yz'], view)
    disp_yz.Representation = 'Surface'
    ColorBy(disp_yz, ('CELLS', 'Vorticity', 'Magnitude'))

    disp_xz = Show(pipeline['slice_xz'], view)
    disp_xz.Representation = 'Surface'
    ColorBy(disp_xz, ('CELLS', 'Vorticity', 'Magnitude'))
    disp_xz.SetScalarBarVisibility(view, True)

    disp_glyph = Show(pipeline['glyph'], view)
    disp_glyph.Representation = 'Surface'
    ColorBy(disp_glyph, ('POINTS', 'order'))
    Hide(pipeline['glyph'], view)

    disp_glyph_v_yz = Show(pipeline['glyph_v_yz'], view)
    disp_glyph_v_yz.Representation = 'Surface'
    disp_glyph_v_yz.AmbientColor   = [0.0, 0.0, 0.0]
    disp_glyph_v_yz.DiffuseColor   = [0.0, 0.0, 0.0]

    disp_glyph_v_xz = Show(pipeline['glyph_v_xz'], view)
    disp_glyph_v_xz.Representation = 'Surface'
    disp_glyph_v_xz.AmbientColor   = [0.0, 0.0, 0.0]
    disp_glyph_v_xz.DiffuseColor   = [0.0, 0.0, 0.0]

    disp_defects = Show(pipeline['defects'], view)
    disp_defects.Representation = 'Surface'
    disp_defects.Opacity = 0.5
    ColorBy(disp_defects, ('POINTS', 'order'))
    if S_RANGE:
        lut = GetColorTransferFunction('order')
        lut.RescaleTransferFunction(*S_RANGE)

    if disc_reader is not None:
        disp_disc = Show(disc_reader, view)
        disp_disc.Representation = 'Wireframe'
        disp_disc.LineWidth       = 3
        disp_disc.AmbientColor    = [1.0, 1.0, 1.0]
        disp_disc.DiffuseColor    = [1.0, 1.0, 1.0]


def parse_args():
    script_argv = sys.argv[sys.argv.index('--') + 1:] if '--' in sys.argv else []
    parser = argparse.ArgumentParser(
        description='ParaView single-frame nematic + disclination visualization.',
        epilog='Pass script arguments after --:\n'
               '  pvpython visualize_frame_in_paraview.py -- --frame 42 --save',
    )
    parser.add_argument(
        '--frame', type=int, default=None,
        help='Frame index (zero-based). Defaults to the last available frame.',
    )
    parser.add_argument(
        '--save', action='store_true',
        help='Save the rendered frame as a PNG to OUTPUT_DIR.',
    )
    return parser.parse_args(script_argv)


def main():
    args = parse_args()

    lbm_files  = sorted(glob.glob(os.path.join(DATA_DIR, 'lbm_*.vtkhdf')))
    disc_files = sorted(glob.glob(os.path.join(DATA_DIR, 'disclinations_*.vtkhdf')))

    if not lbm_files:
        raise FileNotFoundError(f"No lbm_*.vtkhdf files in {DATA_DIR}")

    frame_idx = args.frame if args.frame is not None else len(lbm_files) - 1
    if not (0 <= frame_idx < len(lbm_files)):
        raise IndexError(
            f"Frame {frame_idx} out of range (0–{len(lbm_files) - 1})"
        )

    lbm_file = lbm_files[frame_idx]
    print(f"Loading frame {frame_idx}: {lbm_file}")

    reader = OpenDataFile(lbm_file)
    reader.UpdatePipeline()

    disc_reader = None
    if frame_idx < len(disc_files):
        disc_file = disc_files[frame_idx]
        print(f"Loading disclinations: {disc_file}")
        disc_reader = OpenDataFile(disc_file)
        disc_reader.UpdatePipeline()
    else:
        print("No matching disclination file found; skipping disclination display.")

    pipeline = build_pipeline(reader)

    view = GetActiveViewOrCreate('RenderView')
    view.ViewSize   = [IMAGE_WIDTH, IMAGE_HEIGHT]
    view.Background = [0.15, 0.15, 0.15]

    setup_display(pipeline, reader, view, disc_reader)

    ResetCamera()
    view.CameraPosition   = [14.9, 116.4, 235.0]
    view.CameraFocalPoint = [49.6, 24.6, 24.6]
    view.CameraViewUp     = [0.315, 0.888, -0.335]

    Render()

    if args.save:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        out_path = os.path.join(OUTPUT_DIR, f'frame_{frame_idx:06d}.png')
        SaveScreenshot(out_path, view,
                       ImageResolution=[IMAGE_WIDTH, IMAGE_HEIGHT],
                       TransparentBackground=0)
        print(f"Saved: {out_path}")


if __name__ == '__main__':
    main()
