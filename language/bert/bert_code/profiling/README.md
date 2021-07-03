The profiler is located here: `/usr/local/cuda/bin/nvprof`, so make sure to export PATH.
Install pyprof2: https://github.com/adityaiitb/pyprof2
Note: on newer drivers, run nvprof with root.

## Usage: 
0. `cd profiling`
1. Run profiling: `nvprof -f -o bert_noapex.sql --profile-from-start off -- python profiling.py -a mcbert data`
2. Parse nvprof data with pyprof2: `../../pyprof2/pyprof2/parse/parse.py bert_noapex.sql > bert_noapex.dict`
3. Extract kernel information and align with pytorch semantics: `../../pyprof2/pyprof2/prof/prof.py --csv bert_noapex.dict > bert_noapex.csv`

## Caveats

pyprof2 doesn't support `bool`, so modify `../../pyprof2/pyprof2/prof/utility.py`:
- L12, `if (t in ["bool", "uint8", "int8", "byte", "char"]):`
- L24, `if (t in ["bool", "uint8", "byte", "char"]):`
!!note not sure if bool is vectorized to bits in pytorch, need to confirm

The visual profiler `nvvp` can open and visualize the SQL file.
The default Ubuntu JRE doesn't run it. Use openjdk-8 instead. 
See: https://bugs.launchpad.net/ubuntu/+source/nvidia-cuda-toolkit/+bug/1766948
