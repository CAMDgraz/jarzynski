# smd_analysis
> A simple script for determining Free Energy profile from Steered Molecular
> Dynamics. User must provide a file containing the path to the smd results or
> the extension if the files are the current folder.

## Download and use
For cloning the repository:
```
git clone https://github.com/CAMDgraz/jarzynski.git
```
Then you should be able to see the help by typing:
`python smd_analysis.py -h`

```
$ python smd_analysis.py -h

usage: smd_analysis -jobs_list (or -ext) [options]

Analizys of sMD results using Jarzynski equality

optional arguments:
  -h, --help            show this help message and exit

Input Options:
  -jobs_list JOBS_LIST  File with the sMD jobs
  -ext FILE_EXTENSION   If not -jobs_list provided you must supply the extension of the jobs file (e.g. dat)
  -time TIME            Simulation time in ps (default: 2)
  -tstep TIME_STEP      Time step of the simulation in ps (default: 0.02)
  -T TEMPERATURE        Temperature of the simulation in K (default: 300)
  -check {True,False}   Identify incomplete/missing jobs (default: False)

Jarzynski options:
  -jarz_type {cumulant,exponential}
                        Type of Jarzynski to perform (default: exponential)

Output options:
  -out OUTPUT           Output directory (default: ./)
```
In the example folder you can run:
`python ../smd_analysis.py -jobs_list test.txt`
or 
`python ../smd_analysis.py -ext dat`

Please check the format of the jobs_list file and the smd result files
