
class DIGRAPH(object):
    def __init__(self, vertices, edges):
        # we need at least one vertex and one edge
        assert vertices
        # assert edges
        self.vertices = set(vertices)
        self.edges = set(edges)
        self._succ_cache = dict()
        self._pred_cache = dict()

    def succ(self, u):
        if u not in self._succ_cache:
            self._succ_cache[u] = set()
            for v, w in self.edges:
                # print "{} is of type {}".format(u, type(u))
                # print "{} is of type {}".format(v, type(v))
                if u == v:
                    self._succ_cache[u].add(w)
        return self._succ_cache[u]

    def pred(self, v):
        if v not in self._pred_cache:
            self._pred_cache[v] = set()
            for u, s in self.edges:
                if s == v:
                    self._pred_cache[v].add(u)
        return self._pred_cache[v]

    def get_reachable(self, u):
        visited = set()
        to_visit = set([u])
        while to_visit:
            v = to_visit.pop()
            visited |= set([v])
            for t in self.succ(v):
                if t not in visited:
                    to_visit.add(t)
        return visited

    def get_reachable_set(self, U):
        assert U
        reach_set = set()
        for u in U:
            reach_set |= self.get_reachable(u)

    def sub_graph(self, vertices, edges=None):
        assert vertices
        if self.vertices <= vertices:
            if (edges is None) or (self.edges <= edges):
                return self
        restr_edges = set()
        if edges is not None:
            return DIGRAPH(vertices, edges)
        for v, w in self.edges:
            if (v in vertices) and (w in vertices):
                restr_edges.add((v, w))
        return DIGRAPH(vertices, restr_edges)

    def get_sub_reachable(self, u):
        reachable_vertices = self.get_reachable(u)
        return self.sub_graph(reachable_vertices)

    def get_sccs(self):
        """Return the set of maximal SCCs (Tarjan's algorithm)"""
        sccs = []
        stack = []
        index = dict()
        lowlink = dict()
        onstack = dict()

        class ScopeHolder:
            next_index = 0

        def strong_connect(v):
            index[v] = ScopeHolder.next_index
            lowlink[v] = ScopeHolder.next_index
            ScopeHolder.next_index += 1
            stack.append(v)
            onstack[v] = True

            for w in self.succ(v):
                if w not in index:
                    strong_connect(w)
                    lowlink[v] = min(lowlink[v], lowlink[w])
                elif w in onstack:
                    lowlink[v] = min(lowlink[v], index[w])

            if lowlink[v] == index[v]:
                scc = set()
                while True:
                    w = stack.pop()
                    onstack.pop(w)
                    scc.add(w)
                    if w == v:
                        break
                sccs.append(scc)

        for u in self.vertices:
            if u not in index:
                strong_connect(u)
        return sccs