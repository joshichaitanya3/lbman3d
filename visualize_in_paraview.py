#!/usr/bin/env python3
"""
ParaView offscreen rendering pipeline for 3D nematic time-series data.

Usage (script arguments go after '--'):

  # Full headless render — all timesteps, save PNGs:
  pvpython paraview_pipeline.py -- --headless

  # Interactive debug — 5 files, no PNGs:
  paraview --script=paraview_pipeline.py -- --no-render

  # Interactive debug — 5 files, save PNGs:
  paraview --script=paraview_pipeline.py

Requires ParaView >= 5.11 for VTKHDF reader support.
"""

import argparse
import glob
import os
import sys

from paraview.simple import (
    Calculator, ComputeDerivatives, GetActiveViewOrCreate, GetAnimationScene,
    GetColorTransferFunction, GetOpacityTransferFunction,
    Glyph, Hide, IsoVolume, OpenDataFile,
    RenameSource, Render, ResetCamera,
    SaveScreenshot, Show, Slice, ColorBy,
)


# ===========================================================================
#  Configuration — adjust to match your simulation
# ===========================================================================

DATA_DIR   = './data'
OUTPUT_DIR = os.path.join(DATA_DIR, 'frames')

# Physical dimensions (must match what was passed to VTKHDF3DNematicWriter)
NX, NY, NZ = 128, 32, 32
LX, LY, LZ = NX, NY, NZ
ORIGIN      = (0.0, 0.0, 0.0)

# Midplane positions in world coordinates for the two slice planes
MID_PT_X = ORIGIN[0] + LX / 2
MID_PT_Y = ORIGIN[1] + LY / 2
MID_PT_Z = ORIGIN[2] + LZ / 2

MID_PT = [MID_PT_X, MID_PT_Y, MID_PT_Z]

# Director glyph: one glyph every GLYPH_STRIDE grid points
GLYPH_STRIDE = 4

# Defect threshold: cells where S < DEFECT_S_MAX are shown as defect regions
DEFECT_S_MAX = 0.2

# Colormap ranges — set to None to auto-scale per frame (slower)
S_RANGE = [0.0, 0.7]    # scalar order parameter

# Output image resolution
IMAGE_WIDTH  = 1920
IMAGE_HEIGHT = 1080

# ===========================================================================


def build_pipeline(reader):
    """
    Construct the full filter pipeline from a loaded reader source.
    Returns a dict of named pipeline objects for display setup.
    """
    # ---- Assemble velocity vector from the three scalar components ----
    vel = Calculator(Input=reader)
    vel.AttributeType    = 'Point Data'
    vel.Function         = 'iHat*ux + jHat*uy + kHat*uz'
    vel.ResultArrayName  = 'velocity'
    RenameSource('velocity', vel)

    # ---- Vorticity: curl of velocity via ComputeDerivatives ----
    # ComputeDerivatives with ComputeVorticity=1 directly outputs a
    # 'Vorticity' vector array — no Calculator needed.
    deriv = ComputeDerivatives(Input=vel)
    deriv.Vectors         = ['POINTS', 'velocity']
    # deriv.Vorticity = 1
    # deriv.VorticityArrayName = 'Vorticity'
    # Suppress the Jacobian tensor (we only need vorticity)
    deriv.OutputVectorType = 'Vorticity'
    deriv.OutputTensorType = 'Nothing'
    RenameSource('vorticity_pipeline', deriv)

    # ---- Slice 1: YZ plane coloured by velocity magnitude ----
    slice_yz = Slice(Input=deriv)
    slice_yz.SliceType        = 'Plane'
    slice_yz.SliceType.Normal = [1, 0, 0]
    slice_yz.SliceType.Origin = MID_PT
    RenameSource('slice_YZ', slice_yz)

    # ---- Slice 2: XZ plane coloured by vorticity magnitude ----
    slice_xz = Slice(Input=deriv)
    slice_xz.SliceType        = 'Plane'
    slice_xz.SliceType.Normal = [0, 1, 0]
    slice_xz.SliceType.Origin = MID_PT
    RenameSource('slice_XZ', slice_xz)

    # ---- Velocity glyph on the slices, subsampled ----
    glyph_v_yz = Glyph(Input=slice_yz, GlyphType='Arrow')
    glyph_v_yz.OrientationArray  = ['POINTS', 'velocity']
    glyph_v_yz.ScaleFactor       = (LX / NX) * GLYPH_STRIDE * 0.9  # ~glyph spacing
    glyph_v_yz.GlyphMode         = 'Uniform Spatial Distribution (Surface Sampling)'
    glyph_v_yz.ScaleArray        = ['POINTS', 'velocity']
    RenameSource('velocity_glyph_slice_yz', glyph_v_yz)

    glyph_v_xz = Glyph(Input=slice_xz, GlyphType='Arrow')
    glyph_v_xz.OrientationArray  = ['POINTS', 'velocity']
    glyph_v_xz.ScaleFactor       = (LX / NX) * GLYPH_STRIDE * 0.9  # ~glyph spacing
    glyph_v_xz.GlyphMode         = 'Uniform Spatial Distribution (Surface Sampling)'
    glyph_v_xz.ScaleArray        = ['POINTS', 'velocity']
    RenameSource('velocity_glyph_slice_xz', glyph_v_xz)

    # ---- Director glyph: line representation, subsampled ----
    # GlyphType='Line' produces a segment centered at each point
    # (from -0.5 to +0.5 along the orientation vector) — appropriate
    # for the nematic director where n ≡ -n.
    glyph = Glyph(Input=reader, GlyphType='Line')
    glyph.OrientationArray  = ['POINTS', 'director']
    glyph.ScaleFactor       = (LX / NX) * GLYPH_STRIDE * 0.9  # ~glyph spacing
    glyph.GlyphMode         = 'Uniform Spatial Distribution (Volume Sampling)'
    RenameSource('director_glyph', glyph)

    # ---- IsoVolume: defect regions (low scalar order) ----
    defects = IsoVolume(Input=reader)
    defects.InputScalars  = ['POINTS', 'order']
    defects.ThresholdRange = [0.0, DEFECT_S_MAX]
    RenameSource('defects', defects)

    # Execute all filters so array metadata is available when setting up colormaps.
    for f in [deriv, slice_yz, slice_xz, glyph, defects]:
        f.UpdatePipeline()

    return {
        'deriv':    deriv,
        'slice_yz': slice_yz,
        'slice_xz': slice_xz,
        'glyph':    glyph,
        'defects':  defects,
        'glyph_v_yz': glyph_v_yz,
        'glyph_v_xz': glyph_v_xz
    }


def setup_display(pipeline, reader, view):
    """
    Add all pipeline objects to the render view with their colour mappings.
    Returns display objects so their properties can be adjusted later.
    """
    # Show the domain bounding box as a wireframe outline.
    disp_reader = Show(reader, view)
    disp_reader.Representation = 'Outline'

    # --- YZ slice: vorticity magnitude ---
    # ComputeDerivatives outputs cell data, so colour by CELLS not POINTS.
    # The LUT range is set later in main() after a pre-pass over all timesteps.
    disp_yz = Show(pipeline['slice_yz'], view)
    disp_yz.Representation = 'Surface'
    ColorBy(disp_yz, ('CELLS', 'Vorticity', 'Magnitude'))
    # --- XZ slice: vorticity magnitude (shares the same LUT as YZ) ---
    disp_xz = Show(pipeline['slice_xz'], view)
    disp_xz.Representation = 'Surface'
    ColorBy(disp_xz, ('CELLS', 'Vorticity', 'Magnitude'))
    disp_xz.SetScalarBarVisibility(view, True)

    # --- Director glyphs: hidden by default; coloured by scalar order S ---
    disp_glyph = Show(pipeline['glyph'], view)
    disp_glyph.Representation = 'Surface'
    ColorBy(disp_glyph, ('POINTS', 'order'))
    Hide(pipeline['glyph'], view)

    # --- Velocity glyphs on slices ---
    disp_glyph_v_yz = Show(pipeline['glyph_v_yz'], view)
    disp_glyph_v_yz.Representation = 'Surface'
    disp_glyph_v_yz.AmbientColor = [0.0, 0.0, 0.0] # Black arrows
    disp_glyph_v_yz.DiffuseColor = [0.0, 0.0, 0.0] # Black arrows
    disp_glyph_v_xz = Show(pipeline['glyph_v_xz'], view)
    disp_glyph_v_xz.Representation = 'Surface'
    disp_glyph_v_xz.AmbientColor = [0.0, 0.0, 0.0] # Black arrows
    disp_glyph_v_xz.DiffuseColor = [0.0, 0.0, 0.0] # Black arrows
    # ColorBy(disp_glyph, ('POINTS', 'S'))

    # --- Defect isosurface: semi-transparent, coloured by S ---
    disp_defects = Show(pipeline['defects'], view)
    disp_defects.Representation = 'Surface'
    disp_defects.Opacity = 0.5
    ColorBy(disp_defects, ('POINTS', 'order'))
    if S_RANGE:
        lut = GetColorTransferFunction('order')
        lut.RescaleTransferFunction(*S_RANGE)

    return {
        'slice_yz': disp_yz,
        'slice_xz': disp_xz,
        'glyph':    disp_glyph,
        'glyph_v_yz': disp_glyph_v_yz,
        'glyph_v_xz': disp_glyph_v_xz,
        'defects':  disp_defects,
    }


def compute_vorticity_range(pipeline, scene, timesteps):
    """
    Iterate over all timesteps and return the global (min, max) of the
    vorticity magnitude across the YZ slice cell data.
    """
    vort_min =  float('inf')
    vort_max = -float('inf')
    for t in timesteps:
        scene.AnimationTime = t
        pipeline['slice_yz'].UpdatePipeline(t)
        arr_info = (pipeline['slice_yz']
                    .GetDataInformation()
                    .GetCellDataInformation()
                    .GetArrayInformation('Vorticity'))
        if arr_info:
            lo, hi = arr_info.GetComponentRange(-1)  # -1 = magnitude
            vort_min = min(vort_min, lo)
            vort_max = max(vort_max, hi)
    return vort_min, vort_max


def parse_args():
    # ParaView injects its own flags into sys.argv; everything after '--'
    # belongs to this script.
    script_argv = sys.argv[sys.argv.index('--') + 1:] if '--' in sys.argv else []
    parser = argparse.ArgumentParser(
        description='ParaView nematic visualization pipeline.',
        epilog='Pass script arguments after -- when using pvpython/paraview:\n'
               '  pvpython paraview_pipeline.py -- --headless\n'
               '  paraview --script=paraview_pipeline.py -- --no-render',
    )
    parser.add_argument(
        '--headless', action='store_true',
        help='Render all timestep files.',
    )
    parser.add_argument(
        '--test', action='store_true',
        help='Load a few timestep files for testing the render.',
    )
    parser.add_argument(
        '--no-render', action='store_true',
        help='Build and display the pipeline without saving any PNG frames.',
    )
    return parser.parse_args(script_argv)


def main():
    args = parse_args()
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ---- Load time series ----
    files = sorted(glob.glob(os.path.join(DATA_DIR, 'lbm_*.vtkhdf')))
    if not files:
        raise FileNotFoundError(f"No lbm_*.vtkhdf files in {DATA_DIR}")

    if args.test:
        files = files[:min(len(files), 5)]
        print(f"Testing mode: loading {len(files)} file(s) for debugging.")
    else:
        print(f"Found {len(files)} timestep file(s).")

    reader = OpenDataFile(files)
    reader.UpdatePipeline()

    # ---- Build pipeline (no display yet) ----
    pipeline = build_pipeline(reader)

    # ---- Compute global vorticity range across all timesteps ----
    # This only calls UpdatePipeline on the slice — no rendering needed.
    scene = GetAnimationScene()
    scene.UpdateAnimationUsingDataTimeSteps()
    timesteps = reader.TimestepValues if reader.TimestepValues else [0]

    print("Computing vorticity range over all timesteps...")
    vort_min, vort_max = compute_vorticity_range(pipeline, scene, timesteps)
    print(f"  Vorticity magnitude range: [{vort_min:.4g}, {vort_max:.4g}]")

    # ---- Set up display with the correct LUT range already known ----
    view = GetActiveViewOrCreate('RenderView')
    view.ViewSize       = [IMAGE_WIDTH, IMAGE_HEIGHT]
    view.Background     = [0.15, 0.15, 0.15]
    view.Background2    = [0.0, 0.0, 0.0]
    # view.UseGradientBackground = 1

    setup_display(pipeline, reader, view)

    lut = GetColorTransferFunction('Vorticity')
    lut.RescaleTransferFunction(vort_min, vort_max)

    ResetCamera()
    view.CameraPosition    = [14.9, 116.4, 235.0]
    view.CameraFocalPoint  = [49.6, 24.6, 24.6]
    view.CameraViewUp      = [0.315, 0.888, -0.335]

    # Reset to first timestep before the render loop
    scene.AnimationTime = timesteps[0]
    reader.UpdatePipeline(timesteps[0])

    # ---- Render one PNG per timestep ----
    if args.no_render:
        Render()
        print("Pipeline ready. Skipping PNG export (--no-render).")
        return

    for i, t in enumerate(timesteps):
        scene.AnimationTime = t
        reader.UpdatePipeline(t)
        Render()

        out_path = os.path.join(OUTPUT_DIR, f'frame_{i:06d}.png')
        SaveScreenshot(out_path, view,
                       ImageResolution=[IMAGE_WIDTH, IMAGE_HEIGHT],
                       TransparentBackground=0)
        print(f"[{i+1}/{len(timesteps)}] {out_path}")

    print(f"\nDone. Frames written to {OUTPUT_DIR}")
    # print(f"Stitch with ffmpeg:")
    # print(f"  ffmpeg -r 24 -i {OUTPUT_DIR}/frame_%06d.png "
    #       f"-c:v libx264 -pix_fmt yuv420p output.mp4")


if __name__ == '__main__':
    main()
