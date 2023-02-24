using Conda
using PyCall

# swig, gcc, g++ need to be installed

bin_dir = dirname(PyCall.pyprogramname)

run(`$bin_dir/pip install "gym[box2d]" "gym[mujoco]" mujoco imageio "gym[atari]" ale-py`)
run(`$bin_dir/ale-import-roms roms/`)