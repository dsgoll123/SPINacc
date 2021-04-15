# SPINacc
A spinup acceleration procedure for land surface models which is model independent.

Documentation of aims, concepts, workflows are found in file: !MISSING!


The SPINacc package includes:
* job
* main.py
* Tools/*
* DEF_*/

 
## INFORMATION FOR USERS:
 
### HOW TO RUN THE CODE:

* First: copy the code to your own directory on obelix (this can be done from the command line with "git clone https://github.com/dsgoll123/SPINacc").
* Second: specify in the file 'job' where the code is located using dirpython (L4), and specify the folder with the configuration for your version of ORCHIDEE using dirdef. See more details below.
* Third: adjust the files in configuration folder: i.e. the type of task to be performed as well as the specifis of your simulation:
	* MLacc.def defines the tasks to do and the executation directory: tasks are 1= clustering, 2=training , 3=extrapolation.
	* varlist.json defines the specification of the input data: e.g. resolution, state variables to predict, etc.
*Forth: execute the tool by: qsub -q long job   


### HOW TO SPECIFY THE INPUT DATA / SIMULATIONS:

using the varlist.json files in fiolder DEF_:

You need to modify where the data for ML training is located using sourcepath & sourcefile. The data might be provided in different files, thus the data is separated into groups of variables:

* -climate: climate forcing data used during spinup (mind to specify the same time period as in the ORCHIDEE pixel level runs and in the full spatial domain simulation)
* -pred: other predictor variables used for the ML taken from a short (duration is model version specific; min. 1 year) transient ORCHIDEE simulation over the whole spatial domain (from scratch; not restarting). (This is done as the resolution of boundary conditions other than climate can differ from the resolution of ORCHIDEE, and need to be remapped first to ORCHIDEE resolution. This is automatically done by ORCHIDEE).
	* var1: NPP and LAI from last year of initial short simulation. (These two variables are not boundary conditions for ORCHIDEE but state variables. Information on these variables from the transient spinup phase have been found to improve ML performance for ORCHIDEE-CNP.)
	* var2 - var4: soil properties and/or nutrient-related variables. 
	* For var2 - var4, if they are missing in *_rest.nc (e.g. N and P deposition), please use the variables in *_stomate_history.nc.	
* -PFTmask: max PFT cover fractions, which are used to mask grids with extreme low PFT fractions. This information is usually found in the *_stomate_history.nc (VEGET_COV_MAX).

* -resp: response variables, i.e. the equlibrium pools from the traditional spin-up. Ultimately, it should specify the conventional spinup for part of the spatial domain. At the moment we use simulation for the whole spatial domain. They are usually in the *_stomate_history.nc

You can find the detailed information for each variable in the Trunk and CNP examples: DEF_Trunk/varlist.json, DEF_CNP/varlist.json 
You can create your varlist.json according to your case. 


## INFORMATION FOR CODE DEVELOPERS:


HOW TO UPDATE THE CODE ON GITHUB: you need to do multiple steps: 
* First, "git add" to add all the files that you changed. 
* Second, "git commit" to commit them to your local copy (a difference between svn and git and is that git has a local copy that you commit to). 
* Third, "git push" to push them to the master repository (here). 
This might help: https://git-scm.com/docs/gittutorial

USEFUL COMMANDS: "git diff" will show you all the changes you have made. "git pull" will update all your local files with what exists in the master code (here). "git status" will show you what files have been modified.





