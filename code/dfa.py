from types import *


class ExceptionFSM(Exception):
	"""This is the FSM Exception class."""
	
	def __init__(self, value):
		self.value = value
	
	def __str__(self):
		return self.value


class DFA:
	"""This is a deterministic Finite State Automaton (NFA).
	"""
	
	def __init__(self, initial_state=None, alphabet=None, transitions=dict([]), final_states=None, memory=None):
		self.state_transitions = {}
		self.final_states = set([])
		self.state_transitions = transitions
		if alphabet == None:
			self.alphabet = []
		else:
			self.alphabet = alphabet
		self.initial_state = initial_state
		self.states = [initial_state]  # the list of states in the machine.
	
	def reset(self):
		
		"""This sets the current_state to the initial_state and sets
		input_symbol to None. The initial state was set by the constructor
		 __init__(). """
		
		self.current_state = self.initial_state
		self.input_symbol = None
	
	def add_transition(self, input_symbol, state, next_state=None):
		if next_state is None:
			next_state = state
		else:
			self.state_transitions[(input_symbol, state)] = next_state
		if next_state in self.states:
			pass
		else:
			self.states.append(next_state)
		if state in self.states:
			pass
		else:
			self.states.append(state)
		
		if input_symbol in self.alphabet:
			pass
		else:
			self.alphabet.append(input_symbol)
	
	def get_transition(self, input_symbol, state):
		
		"""This returns a list of next states given an input_symbol and state.
		"""
		return self.state_transitions.get((input_symbol, state))


class DRA(DFA, object):
	"""A child class of DFA --- determinisitic Rabin automaton
	"""
	
	def __init__(self, initial_state=None, alphabet=None, transitions=dict([]), rabin_acc=None, memory=None):
		# The rabin_acc is a list of rabin pairs rabin_acc=[(J_i, K_i), i =0,...,N]
		# Each K_i, J_i is a set of states.
		# J_i is visited only finitely often
		# K_i has to be visited infinitely often.
		super(DRA, self).__init__(initial_state, alphabet, transitions)
		self.acc = rabin_acc
	
	def add_rabin_acc(self, rabin_acc):
		self.acc = rabin_acc


if __name__ == '__main__':
	# construct a DRA, which is a complete automaton.
	dra = DRA(0, ['1', '2', '3', '4', 'E'])  # we use 'E' to stand for everything else other than 1,2,3,4.
	dra.add_transition('2', 0, 0)
	dra.add_transition('3', 0, 0)
	dra.add_transition('E', 0, 0)
	
	dra.add_transition('1', 0, 1)
	dra.add_transition('1', 1, 2)
	dra.add_transition('3', 1, 2)
	dra.add_transition('E', 1, 2)
	
	dra.add_transition('2', 1, 3)
	
	dra.add_transition('1', 2, 2)
	dra.add_transition('3', 2, 2)
	dra.add_transition('E', 2, 2)
	
	dra.add_transition('2', 2, 3)
	
	dra.add_transition('1', 3, 3)
	dra.add_transition('2', 3, 3)
	dra.add_transition('E', 3, 3)
	
	dra.add_transition('3', 3, 0)
	
	dra.add_transition('4', 0, 4)
	dra.add_transition('4', 1, 4)
	dra.add_transition('4', 2, 4)
	dra.add_transition('4', 3, 4)
	dra.add_transition('4', 4, 4)
	dra.add_transition('1', 4, 4)
	dra.add_transition('2', 4, 4)
	dra.add_transition('3', 4, 4)
	dra.add_transition('E', 4, 4)
	
	J0 = {4}
	K0 = {1}
	rabin_acc = [(J0, K0)]
	dra.add_rabin_acc(rabin_acc)

