import copy
from digraph import DIGRAPH


class NFA(object):
    def __init__(self, states, alphabet, transitions=[]):
        # we need at least one state and one letter
        assert states
        # assert accepting_states
        assert alphabet
        self.states = set(states)
        #self.accepting_states = set(accepting_states)
        # print "states {}".format(self.states)
        # print "accepting states {}".format(self.accepting_states)
        #assert self.accepting_states <= self.states
        self.alphabet = set(alphabet)
        self.transitions = set()
        self._post_cache = dict()
        self._pre_cache = dict()
        self._available_cache = dict()

        for s, a, t in transitions:
            # print "transition ({},{},{})".format(s, a, t)
            self.transitions.add((s, a, t))

    def get_sub_nfa(self, states, allowed=None):
        assert states
        new_accepting_states = self.accepting_states & self.states
        transitions = []
        for s in states:
            if allowed is None:
                actions = self.available(s)
            else:
                actions = allowed[s]
            for a in actions:
                post_sa = self.post(s, a)
                for t in post_sa & states:
                    transitions.append((s, a, t))
        return NFA(states, new_accepting_states, self.alphabet, transitions)

    def reachable_sub_nfa(self, initial):
        visited = set([])
        to_visit = set([initial])
        while to_visit:
            s = to_visit.pop()
            # print "visiting state {}".format(s)
            visited.add(s)
            for t in self.post_all(s):
                if t not in visited:
                    to_visit.add(t)
        return self.get_sub_nfa(visited)

    def get_graph(self):
        trans = [(s, t) for s, a, t in self.transitions]
        graph = DIGRAPH(self.states, trans)
        return graph

    def get_subgraph(self, states, allowed):
        state_set = set(states)
        assert state_set <= set([x for x in allowed])
        trans = [(s, t) for s, a, t in self.transitions
                 if s in state_set
                 and t in state_set
                 and a in allowed[s]]
        graph = DIGRAPH(state_set, trans)
        return graph

    def available(self, q):
        if q not in self._available_cache:
            self._available_cache[q] = set()
        for s, a, t in self.transitions:
            if s == q:
                self._available_cache[q].add(a)
        return set(self._available_cache[q])

    def _prepare_post_cache(self):
        for s, a, t in self.transitions:
            if (s, a) not in self._post_cache:
                self._post_cache[(s, a)] = set()
            self._post_cache[(s, a)].add(t)

    def _prepare_pre_cache(self):
        for s, a, t in self.transitions:
            if t not in self._pre_cache:
                self._pre_cache[t] = set()
            self._pre_cache[t].add((s, a))

    def post(self, q, a):
        assert a in self.available(q)
        if (q, a) not in self._post_cache:
            self._post_cache[(q, a)] = set()
        for s, b, t in self.transitions:
            if s == q and a == b:
                self._post_cache[(q, a)].add(t)
        return set(self._post_cache[(q, a)])

    def post_all(self, q):
        all_states = set()
        for a in self.available(q):
            all_states.update(self.post(q, a))
        return all_states

    def pre(self, q):
        assert q in self.states
        if q in self._pre_cache.keys():
            return (self._pre_cache[q])
        else:
            return set()

    def is_total(self):
        self._prepare_post_cache()
        for s in self.states:
            for a in self.alphabet:
                if not self.post(s, a):
                    return False
        return True

    def make_total(self, make_accepting=False):
        non_total = set()
        self._prepare_post_cache()
        for s in self.states:
            for a in self.alphabet:
                if not self.post(s, a):
                    non_total.add(s)
        if not non_total:
            return
        sink = "sink"
        self.states.add(sink)
        for a in self.alphabet:
            self.transitions.add((sink, a, sink))
        if make_accepting:
            self.accepting_states.add(sink)
        for s in non_total:
            for a in self.alphabet:
                if not self.post(s, a):
                    self.transitions.add((s, a, sink))

    def get_mecs(self):
        allowed = dict()
        for s in self.states:
            allowed[s] = self.available(s)
        mecs = []
        mecs_new = [set(self.states)]

        while True:
            mecs = mecs_new
            mecs_new = []
            for T in mecs:
                to_remove = set([])
                sccs = self.get_subgraph(T, allowed).get_sccs()
                for S in sccs:
                    for q in S:
                        allowed[q] = set([a for a in allowed[q]
                                          if self.post(q, a) <= S])
                        if not allowed[q]:
                            to_remove.add(q)
                while to_remove:
                    s = to_remove.pop()
                    T.remove(s)
                    for t in T:
                        for a in self.available(t):
                            if s in self.post(t, a):
                                allowed[t].discard(a)
                        if not allowed[t]:
                            to_remove.add(t)
                for S in sccs:
                    inter = T & S
                    if inter:
                        mecs_new.append(inter)
            if mecs == mecs_new:
                break

        def restrict(S, dictionary):
            d = dict()
            for s in S:
                d[s] = dictionary[s]
            return d

        return [(T, restrict(T, allowed)) for T in mecs]

    def prob_max_0(self, target=None):
        if target is None:
            target = self.accepting_states

        U = target.copy()
        while True:
            R = U.copy()
            for s in self.states:
                for a in self.available(s):
                    if len(U.intersection(self.post(s, a))) > 0:
                        U.add(s)
            if U == R:
                break
        prob_max_0_states = self.states - U
        return prob_max_0_states

    def prob_min_1(self, target=None):
        if target is None:
            target = self.accepting_states
        prob_min_1_states = set()
        while True:
            temp_targetset = prob_min_1_states.copy()
            for s in target:
                for t in self.states:
                    post_t = set()
                    for a in self.available(t):
                        post_t = post_t.union(self.post(t, a))
                    if post_t == {s}:
                        prob_min_1_states.add(t)
            if len(temp_targetset) == len(prob_min_1_states):
                break
        return prob_min_1_states

    def prob_max_1(self,target = None):
        if target  is None:
            target = self.accepting_states
        sub_NFA = copy.deepcopy(self)
        # U = sub_NFA.states - target
        U = self.prob_max_0()

        while True:
            sub_NFA._prepare_pre_cache()
            allowed = dict()
            for s in sub_NFA.states:
                allowed[s] = sub_NFA.available(s)

            R = U.copy()
            while len(R) > 0:
                u = R.pop()
                for (t, a) in sub_NFA.pre(u):
                    if t not in U:
                        if a in allowed[t]:
                            # print "action {} from state {} is bad".format(
                            #     a, t)
                            allowed[t].remove(a)
                        if len(allowed[t]) == 0:
                            # print "state {} is bad because it cannot "\
                            #     "avoid bad states".format(t)
                            R.add(t)
                            U.add(t)
                        sub_NFA = sub_NFA.get_sub_nfa(sub_NFA.states, allowed)
                        sub_NFA._prepare_pre_cache()

                if u not in target:
                    # print "state {} is removed because it is not a "\
                    #     "target and it is bad".format(u)
                    reduced_states = sub_NFA.states - set([u])
                sub_NFA = sub_NFA.get_sub_nfa(reduced_states)
                sub_NFA._prepare_pre_cache()
                allowed = dict()
                for s in sub_NFA.states:
                    allowed[s] = sub_NFA.available(s)

            # U = (self.states - U) - target
            U = sub_NFA.prob_max_0()
            if len(U) == 0:
                break
        return sub_NFA.states