import numpy as np
import random
import itertools
import os
import subprocess
from scipy.special import comb
import numpy as np
from copy import deepcopy
import random
import operator
from collections import OrderedDict
import pickle
import policy_reactive
import simplejson as json
import operator

class Policy():
	
	# def __init__(self,mdp,public_mdp,init,target,lookahead,public_target,bad_mdp=[],bad_target=[],slugs_location=None):
	def __init__(self,init,target,public_target,bad_target,slugs_location,gw_env,lookahead=50,id_no=0):
		# self.mdp = mdp
		# self.nfa = nfa  # Deterministic transitions -- used for nominal trace
		self.id_no = id_no
		self.gw = gw_env
		self.init = init
		self.target = target
		self.public_target = public_target
		self.lookahead = lookahead
		self.last_node = 0
		self.actualpolicy = self.ComputePolicy(slugs_location=slugs_location,targets=target)
		if target != public_target:
			self.badpolicy = self.actualpolicy
			self.publicpolicy = self.ComputePolicy(slugs_location=slugs_location,targets=public_target)
		else:
			self.badpolicy = self.ComputePolicy(slugs_location=slugs_location,targets=bad_target)
			self.publicpolicy = self.actualpolicy


	# def computePolicy(self,targ,mdp):
	# 	T = self.lookahead
	# 	R = dict([(s, a, next_s), 0.0] for s in mdp.states for a in mdp.available(s) for next_s in mdp.post(s, a))
	# 	R.update([(s, a, next_s), 1.0] for s in mdp.states for a in mdp.available(s) for next_s in mdp.post(s, a) if next_s in set(targ))
	# 	V,pol_dict = mdp.E_step_value_iteration(R,set(),set())
	# 	return pol_dict
		
	# def updatePolicy(self,targ,mdp):
	# 	T = self.lookahead
	# 	R = dict([(s, a, next_s), 0.0] for s in mdp.states for a in mdp.available(s) for next_s in mdp.post(s, a))
	# 	R.update([(s, a, next_s), 1.0] for s in mdp.states for a in mdp.available(s) for next_s in mdp.post(s, a) if next_s in set(targ))
	# 	V, pol_dict = mdp.T_step_value_iteration(R, T)
	# 	self.policy = pol_dict
	# 	self.mc = mdp.construct_MC(self.policy)
	# 	self.updateNominal(self.init)
	#
	# def nominalTrace(self,loc,mdp,policy=None):
	# 	T = self.lookahead
	# 	if policy:
	# 		return mdp.computeTrace(loc,policy,T)
	# 	else:
	# 		return mdp.computeTrace(loc,self.public_policy,T)
	# def updateNominal(self,loc,mdp=None):
	# 	if mdp:
	# 		self.nom_trace = self.nominalTrace(loc,mdp)
	# 	self.nom_trace = self.nominalTrace(loc,self.public_mdp)
	# 	if self.bad_pol:
	# 		self.bad_trace = self.nominalTrace(loc,self.bad_mdp,self.bad_pol)
	#
	# def changeTarget(self,new_target):
	# 	self.target = new_target
	# 	self.updatePolicy()
	#
	# def sample(self,state):
	# 	s_l = list(self.policy[state])
	# 	random.shuffle(s_l)
	# 	return s_l[0]
	#
	# def mcProb(self, init, T,mc=[]):
	# 	if mc ==[]:
	# 		mc = self.public_mc
	# 	s = init
	# 	MC_prob = dict([s_t, 1.0] for s_t in self.mdp.states)
	# 	t = 0
	# 	while t < T:
	# 		MC_prob_up = dict([s_t, 0.0] for s_t in self.mdp.states)
	# 		for s_i in s:
	# 			for z in self.mdp.states:
	# 				MC_prob_up[z] += MC_prob[s_i] * mc[(s_i, z)]
	# 		MC_prob = MC_prob_up  # /sum([MC_prob_up[e] for e in MC_prob_up])
	# 		# assert sum([MC_prob[e] for e in MC_prob])==1.0
	# 		s = [s_e for s_e in self.mdp.states if MC_prob[s_e] != 0.0]
	# 		t += 1
	# 	return MC_prob
	
	def observation(self, est_loc, last_sight, t):
		self.last_node = [i_e for i_e,v_e in self.actualpolicy.items() if v_e['State']['s']==last_sight[0]][0]
		for t_i in range(t):
			self.last_node = self.publicpolicy[self.last_node%len(self.publicpolicy)]['Successors'][0]
		return self.publicpolicy[self.last_node]['State']['s']
	
	def bad_observation(self,est_loc,last_sight,t):
		# last_node = [i_e for i_e,v_e in self.badpolicy.items() if v_e['State']['s']==last_sight[0]][0]
		# for t_i in range(t):
		# 	last_node = self.badpolicy[last_node]['Successors'][0]
		return self.badpolicy[self.last_node%len(self.badpolicy)]['State']['s']

	def ComputePolicy(self, infile=None, slugs_location=None, preload=False,targets=None):

		if infile == None:
			infile = 'grid_example_{}'.format(self.id_no)
			filename = infile + '.structuredslugs'
		else:
			filename = infile + '_{}'.format(self.id_no) + '.structuredslugs'

		if preload:
			return self.parseJson(infile + '.json')

		file = open(filename, 'w')

		file.write('[INPUT]\n')
		file.write('\n')  # Add moving obstacles

		file.write('\n[OUTPUT]\n')
		file.write('s:0...{}'.format(self.gw.nstates - 1))

		file.write('\n[ENV_INIT]\n')
		file.write('\n')  # Add moving obstacles

		file.write('\n[SYS_INIT]\n')
		file.write('s={}\n'.format(self.init))

		file.write('\n[ENV_TRANS]\n')


		file.write('\n[SYS_TRANS]\n')

		for s in self.gw.states:
			strn = 's = {} ->'.format(s)
			repeat = set()
			for u in self.gw.actlist:
				snext = np.nonzero(self.gw.prob[u][s])[0][0]
				if snext not in repeat:
					repeat.add(snext)
					strn += 's\' = {} \\/ '.format(snext)
			strn = strn[:-3]
			file.write(strn + '\n')
		for s in self.gw.obstacles:
			file.write('!s = {}\n'.format(s))
		file.write('\n[SYS_LIVENESS]\n')
		# t_s = self.id_idx[self.id_no] % len(targets)

		for i,s in enumerate(targets):
			file.write('s = {} \n'.format(s))


		file.close()

		if slugs_location != None:
			os.system(
				'python2 ' + slugs_location + 'tools/StructuredSlugsParser/compiler.py ' + infile + '.structuredslugs > ' + infile + '.slugsin')
			sp = subprocess.Popen(
				slugs_location + 'src/slugs --explicitStrategy --jsonOutput ' + infile + '.slugsin > ' + infile + '.json',
				shell=True, stdout=subprocess.PIPE)
			sp.wait()
			print('Computing controller...')
			return self.parseJson(infile + '.json')

	def writeJson(self, infile, outfile, dict=None):
		if dict is None:
			dict = self.parseJson(infile)
		j = json.dumps(dict, indent=1)
		f = open(outfile, 'w')
		print >> f, j
		f.close()

	def parseJson(self, filename, outfilename=None):
		automaton = dict()
		file = open(filename)
		data = json.load(file)
		file.close()
		variables = dict()
		for var in data['variables']:
			v = var.split('@')[0]
			if v not in variables.keys():
				for var2ind in range(data['variables'].index(var), len(data['variables'])):
					var2 = data['variables'][var2ind]
					if v != var2.split('@')[0]:
						variables[v] = [data['variables'].index(var), data['variables'].index(var2)]
						break
					if data['variables'].index(var2) == len(data['variables']) - 1:
						variables[v] = [data['variables'].index(var), data['variables'].index(var2) + 1]

		for s in data['nodes'].keys():
			automaton[int(s)] = dict.fromkeys(['State', 'Successors'])
			automaton[int(s)]['State'] = dict()
			automaton[int(s)]['Successors'] = []
			for v in variables.keys():
				if variables[v][0] == variables[v][1]:
					bin = [data['nodes'][s]['state'][variables[v][0]]]
				else:
					bin = data['nodes'][s]['state'][variables[v][0]:variables[v][1]]
				automaton[int(s)]['State'][v] = int(''.join(str(e) for e in bin)[::-1], 2)
				automaton[int(s)]['Successors'] = data['nodes'][s]['trans']
		if outfilename == None:
			return automaton
		else:
			self.writeJson(None, outfilename, automaton)
			return automaton

	def definePolicyDict(self, id_list, policy_array):
		self.policy_list = dict([[i, p_i] for i, p_i in zip(id_list, policy_array)])

	def savePolicy(self):
		print(self.id_no)
		with open('policies/' + str(self.id_no) + '.pkl', 'wb') as f:
			pickle.dump(self.policy, f, pickle.HIGHEST_PROTOCOL)

	def loadPolicy(self):
		print(self.id_no)
		with open('policies/' + str(self.id_no) + '.pkl', 'rb') as f:
			self.policy = pickle.load(f)