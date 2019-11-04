from gridworld import Gridworld
import numpy as np
import itertools
#grid parameters
nrows = 3
ncols = 3
initial=[0]
targets=[3,5]
obstacles=[]
gwg = Gridworld(initial, nrows=nrows, ncols=ncols, nagents=1, targets=targets, obstacles=obstacles, moveobstacles = [])
write_slugs_file()

def write_slugs_file(gwg,infile):

	if infile == None:
		infile = 'grid_example'
		filename = infile+'.structuredslugs'
	else:
		filename = infile+'.structuredslugs'

	file = open(filename, 'w')


	file.write('[INPUT]\n')
	file.write('\n') # Add moving obstacles

	file.write('\n[OUTPUT]\n')
	file.write('s:0...{}'.format(gwg.nstates-1))

	file.write('\n[ENV_INIT]\n')
	file.write('\n') # Add moving obstacles

	file.write('\n[SYS_INIT]\n')
	for n in range(gwg.nagents):
		file.write('s{}={}\n'.format(n,gwg.current[n]))


	file.close(infile)


