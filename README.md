# Libero Adaptor for UniEnvPy

Under development.

## Space metadata

`LiberoHDF5Data` annotates the space it infers from the HDF5 file with unienv
space metadata (`libero_space_metadata`; requires `unienv>=0.0.1b16`),
following the layout of LIBERO's `scripts/create_dataset.py`: RGB cameras
(flagged `flipped="vertical"` for robosuite's `"opengl"` image convention),
end-effector / joint / gripper states, and `actions` as normalized OSC_POSE
deltas (position + axis-angle + gripper).
