import numpy as np
import os
import subprocess
import simplejson as json

def Policy(gwg,infile=None,slugs_location=None):

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
	file.write('s={}\n'.format(gwg.current[0]))

	file.write('\n[SYS_TRANS]\n')

	for s in gwg.states:
		strn = 's = {} ->'.format(s)
		repeat = set()
		for u in gwg.actlist:
			snext = np.nonzero(gwg.prob[u][s])[0][0]
			if snext not in repeat:
				repeat.add(snext)
				strn += 's\' = {} \\/ '.format(snext)
		strn = strn[:-3]
		file.write(strn+'\n')
	for s in gwg.obstacles:
		file.write('!s = {}\n'.format(s))
	file.write('\n[SYS_LIVENESS]\n')
	for s in gwg.targets:
		file.write('s = {}\n'.format(s))

	file.close()

	if slugs_location!=None:
		os.system(
		'python ' + slugs_location + 'tools/StructuredSlugsParser/compiler.py ' + infile + '.structuredslugs > ' + infile + '.slugsin')
		sp = subprocess.Popen(slugs_location + 'src/slugs --explicitStrategy --jsonOutput ' + infile + '.slugsin > ' + infile+'.json',shell=True, stdout=subprocess.PIPE)
		sp.wait()
		print('Computing controller...')
		return parseJson(infile+'.json')

def writeJson(infile,outfile,dict=None):
	if dict is None:
		dict = parseJson(infile)
	j = json.dumps(dict, indent=1)
	f = open(outfile, 'w')
	print >> f, j
	f.close()

def parseJson(filename,outfilename=None):
	automaton = dict()
	file = open(filename)
	data = json.load(file)
	file.close()
	variables = dict()
	for var in data['variables']:
		v = var.split('@')[0]
		if v not in variables.keys():
			for var2ind in range(data['variables'].index(var),len(data['variables'])):
				var2 = data['variables'][var2ind]
				if v != var2.split('@')[0]:
					variables[v] = [data['variables'].index(var), data['variables'].index(var2)]
					break
				if data['variables'].index(var2) == len(data['variables'])-1:
					variables[v] = [data['variables'].index(var), data['variables'].index(var2)+1]

	for s in data['nodes'].keys():
		automaton[int(s)] = dict.fromkeys(['State','Successors'])
		automaton[int(s)]['State'] = dict()
		automaton[int(s)]['Successors'] = []
		for v in variables.keys():
			if variables[v][0] == variables[v][1]:
				bin  = [data['nodes'][s]['state'][variables[v][0]]]
			else:
				bin = data['nodes'][s]['state'][variables[v][0]:variables[v][1]]
			automaton[int(s)]['State'][v] = int(''.join(str(e) for e in bin)[::-1], 2)
			automaton[int(s)]['Successors'] = data['nodes'][s]['trans']
	if outfilename==None:
		return automaton
	else:
		writeJson(None,outfilename,automaton)