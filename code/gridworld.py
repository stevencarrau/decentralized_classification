__author__ = 'sudab'
""" Generate a grid world """
import os, sys, getopt, pdb, string
import random
import numpy as np
import pygame
import pygame.locals as pgl

class Gridworld():
    # a gridworld with uneven terrain
    def __init__(self, initial, nrows=8, ncols=8, nagents=1, targets=[], obstacles=[], moveobstacles = [], regions=dict(),size=30):
        # walls are the obstacles. The edges of the gridworld will be included into the walls.
        # region is a string and can be one of: ['pavement','gravel', 'grass', 'sand']
        self.initial = initial
        self.current = initial
        self.nrows = nrows  
        self.ncols = ncols
        self.nagents = nagents
        self.nstates = nrows * ncols
        self.nactions = 5
        self.regions = regions
        self.actlist = ['N', 'S', 'W', 'E', 'R']
        self.targets = targets
        self.left_edge = []
        self.right_edge = []
        self.top_edge = []
        self.bottom_edge = []
        self.obstacles = obstacles
        self.moveobstacles = moveobstacles
        self.states = range(nrows*ncols)
        self.colorstates = set()
        for x in range(self.nstates):
            # note that edges are not disjoint, so we cannot use elif
            if x % self.ncols == 0:
                self.left_edge.append(x)
            if 0 <= x < self.ncols:
                self.top_edge.append(x)
            if x % self.ncols == self.ncols - 1:
                self.right_edge.append(x)
            if (self.nrows - 1) * self.ncols <= x <= self.nstates:
                self.bottom_edge.append(x)
        self.edges = self.left_edge + self.top_edge + self.right_edge + self.bottom_edge
        self.walls = self.edges + obstacles
        self.prob = {a: np.zeros((self.nstates, self.nstates)) for a in self.actlist}

        self.probOfSuccess = dict([])
        self.getProbRegions()

        for s in self.states:
            for a in self.actlist:
                self.getProbs(s, a)

    def coords(self, s):
        return (int(s / self.ncols), int(s % self.ncols))  # the coordinate for state s.

    def isAllowed(self, co_ords):
        if co_ords[1] not in range(self.ncols) or co_ords[0] not in range(self.nrows):
            return False
        return True

    def isAllowedState(self,co_ords,returnState):
        if self.isAllowed(co_ords):
            return self.rcoords(co_ords)
        return returnState

    def getProbRegions(self):
        probOfSuccess = dict([])
        for ground in self.regions.keys():
            for direction in ['N', 'S', 'E', 'W']:
                if ground == 'pavement':
                    mass = random.choice(range(90, 95))
                    massleft = 100 - mass
                    oneleft = random.choice(range(1, massleft))
                    twoleft = massleft - oneleft
                if ground == 'gravel':
                    mass = random.choice(range(80, 85))
                    massleft = 100 - mass
                    oneleft = random.choice(range(1, massleft))
                    twoleft = massleft - oneleft
                if ground == 'grass':
                    mass = random.choice(range(85, 90))
                    massleft = 100 - mass
                    oneleft = random.choice(range(1, massleft))
                    twoleft = massleft - oneleft
                if ground == 'sand':
                    mass = random.choice(range(65, 70))
                    massleft = 100 - mass
                    oneleft = random.choice(range(1, massleft))
                    twoleft = massleft - oneleft
                if ground == 'deterministic':
                    mass = 100
                    oneleft = 0
                    twoleft = 0
                probOfSuccess[(ground, direction)] = [float(mass) / 100, float(oneleft) / 100, float(twoleft) / 100]
        self.probOfSuccess = probOfSuccess
        return

    def rcoords(self, coords):
        s = coords[0] * self.ncols + coords[1]
        return s

    def getProbs(self, state, action):
        successors = []

        if state in self.obstacles:
            successors = [(state, 1)]
            for (next_state, p) in successors:
                self.prob[action][state, next_state] = p
                return
        row,col = self.coords(state)
        northState = self.isAllowedState((row-1,col),state)
        northwestState = self.isAllowedState((row-1,col-1),state)
        northeastState = self.isAllowedState((row-1,col+1),state)
        southState = self.isAllowedState((row+1,col),state)
        southeastState = self.isAllowedState((row+1,col+1),state)
        southwestState = self.isAllowedState((row+1,col-1),state)
        westState = self.isAllowedState((row,col-1),state)
        eastState = self.isAllowedState((row,col+1),state)
        # northState = (self.isAllowed(state - self.ncols) and state - self.ncols) or state
        # northwestState = (self.isAllowed(state - 1 - self.ncols) and state - 1 - self.ncols) or state
        # northeastState = (self.isAllowed(state + 1 - self.ncols) and state - self.ncols + 1) or state
        #
        # southState = (self.isAllowed(state + self.ncols) and state + self.ncols) or state
        # southeastState = (self.isAllowed(state + 1 + self.ncols) and state + 1 + self.ncols) or state
        # southwestState = (self.isAllowed(state - 1 + self.ncols) and state - 1 + self.ncols) or state
        #
        # westState = (self.isAllowed(state - 1) and state - 1) or state
        # eastState = (self.isAllowed(state + 1) and state + 1) or state

        reg = self.getStateRegion(state)
        if action == 'N':
            [p0, p1, p2] = self.probOfSuccess[(reg, 'N')]
            successors.append((northState, p0))
            successors.append((northwestState, p1))
            successors.append((northeastState, p2))

        if action == 'S':
            [p0, p1, p2] = self.probOfSuccess[(reg, 'S')]
            successors.append((southState, p0))
            successors.append((southwestState, p1))
            successors.append((southeastState, p2))

        if action == 'W':
            [p0, p1, p2] = self.probOfSuccess[(reg, 'W')]
            successors.append((westState, p0))
            successors.append((southwestState, p1))
            successors.append((northwestState, p2))

        if action == 'E':
            [p0, p1, p2] = self.probOfSuccess[(reg, 'W')]
            successors.append((eastState, p0))
            successors.append((southeastState, p1))
            successors.append((northeastState, p2))

        if action == 'R':
            successors.append((state,1))

        for (next_state, p) in successors:
            self.prob[action][state, next_state] += p

    def getStateRegion(self, state):
        if state in self.regions['pavement']:
            return 'pavement'
        if state in self.regions['grass']:
            return 'grass'
        if state in self.regions['gravel']:
            return 'gravel'
        if state in self.regions['sand']:
            return 'sand'
        if state in self.regions['deterministic']:
            return 'deterministic'

    ## Everything from here onwards is for creating the image

    def render(self, size=100):
        self.height = self.nrows * size + self.nrows + 1
        self.width = self.ncols * size + self.ncols + 1
        self.size = size

        #       # initialize pygame ( SDL extensions )
        pygame.init()
        pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption('Gridworld')
        self.screen = pygame.display.get_surface()
        self.surface = pygame.Surface(self.screen.get_size())
        self.bg = pygame.Surface(self.screen.get_size())
        self.bg_rendered = False  # optimize background render

        self.background()
        self.screen.blit(self.surface, (0, 0))
        pygame.display.flip()

        self.build_templates()
        self.updategui = True  # switch to stop updating gui if you want to collect a trace quickly

        self.state2circle(self.current)

    def getkeyinput(self):
        events = pygame.event.get()
        for event in events:
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_LEFT:
                    return 'W'
                elif event.key == pygame.K_RIGHT:
                    return 'E'
                if event.key == pygame.K_UP:
                    return 'N'
                elif event.key == pygame.K_DOWN:
                    return 'S'
                elif event.key == pygame.K_SPACE:
                    return 'Space'

    def build_templates(self):

        # Note: template already in "graphics" coordinates
        template = np.array([(-1, 0), (0, 0), (1, 0), (0, 1), (1, 0), (0, -1)])
        template = self.size / 3 * template  # scale template

        v = 1.0 / np.sqrt(2)
        rot90 = np.array([(0, 1), (-1, 0)])
        rot45 = np.array([(v, -v), (v, v)])  # neg


        #
        # align the template with the first action.
        t0 = np.dot(template, rot90)
        t0 = np.dot(t0, rot90)
        t0 = np.dot(t0, rot90)

        t1 = np.dot(t0, rot45)
        t2 = np.dot(t1, rot45)
        t3 = np.dot(t2, rot45)
        t4 = np.dot(t3, rot45)
        t5 = np.dot(t4, rot45)
        t6 = np.dot(t5, rot45)
        t7 = np.dot(t6, rot45)

        self.t = [t0, t1, t2, t3, t4, t5, t6, t7]

    def indx2coord(self, s, center=False):
        # the +1 indexing business is to ensure that the grid cells
        # have borders of width 1px
        i, j = self.coords(s)
        if center:
            return int(i * (self.size + 1) + 1 + self.size / 2), \
                   int(j * (self.size + 1) + 1 + self.size / 2)
        else:
            return int(i * (self.size + 1) + 1), int(j * (self.size + 1) + 1)

    def accessible_blocks(self, s):
        """
        For a give state s, generate the list of walls around it.
        """
        W = []
        if s in self.walls:
            return W
        if s - self.ncols < 0 or s - self.ncols in self.walls:
            pass
        else:
            W.append(s - self.ncols)
        if s - 1 < 0 or s - 1 in self.walls:
            pass
        else:
            W.append(s - 1)
        if s + 1 in self.walls:
            pass
        else:
            W.append(s + 1)
        if s + self.ncols in self.walls:
            pass
        else:
            W.append(s + self.ncols)
        return W

    def coord2indx(self, xy):
        return self.rcoords(int((xy[0] / (self.size + 1)), int(xy[1] / (self.size + 1))))

    def draw_state_labels(self):
        font = pygame.font.SysFont("FreeSans", 10)
        for s in range(self.nstates):
            x, y = self.indx2coord(s, False)
            txt = font.render("%d" % s, True, (0, 0, 0))
            self.surface.blit(txt, (y, x))

        self.screen.blit(self.surface, (0, 0))
        pygame.display.flip()

    def coord2state(self, coord):
        s = self.coord2indx((coord[0], coord[1]))
        return s

    def state2circle(self, state, bg=True, blit=True):
        if bg:
            self.background()

        for n in range(self.nagents):
            x, y = self.indx2coord(state[n], center=True)
            pygame.draw.circle(self.surface, (0, 0, 255), (y, x), int(self.size / 2))
        if len(self.moveobstacles) > 0:
            for s in self.moveobstacles:
                x, y = self.indx2coord(s, center=True)
                pygame.draw.circle(self.surface, (205, 92, 0), (y, x), int(self.size / 2))
        if blit:
            self.screen.blit(self.surface, (0, 0))
            pygame.display.flip()

    def draw_values(self, vals):
        """
        vals: a dict with state labels as the key
        """
        font = pygame.font.SysFont("FreeSans", 10)

        for s in range(self.nstates):
            x, y = self.indx2coord(s, False)
            v = vals[s]
            txt = font.render("%.1f" % v, True, (0, 0, 0))
            self.surface.blit(txt, (y, x))

        self.screen.blit(self.surface, (0, 0))
        pygame.display.flip()

    #
    def save(self, filename):
        pygame.image.save(self.surface, filename)

    def redraw(self):
        self.screen.blit(self.surface, (0, 0))
        pygame.display.flip()

    def move_obj(self, s, bg=True, blit=True):

        """Including A moving object into the gridworld, which moves uniformly at
        random in all accessible directions (including idle), without
        hitting the wall or another other statitic obstacle.  Input: a
        gridworld gui, the current state index for the obstacle and the
        number of steps.

        """
        if bg:
            self.background()
        x, y = self.indx2coord(s, center=True)
        pygame.draw.circle(self.surface, (205, 92, 0), (y, x), int(self.size / 2))

        if blit:
            self.screen.blit(self.surface, (0, 0))
            pygame.display.flip()

        return

    def move_deter(self, next_state):
        self.current = next_state

        return

    def background(self):

        if self.bg_rendered:
            self.surface.blit(self.bg, (0, 0))
        else:
            self.bg.fill((84, 84, 84))
            font = pygame.font.SysFont("FreeSans", 10)

            for s in range(self.nstates):
                x, y = self.indx2coord(s, False)
                coords = pygame.Rect(y, x, self.size, self.size)
                pygame.draw.rect(self.bg, ((250, 250, 250)), coords)
            for n in range(self.nagents):
                for t in self.targets[n]:
                    x, y = self.indx2coord(t, center=True)
                    coords = pygame.Rect(y - self.size / 2, x - self.size / 2, self.size, self.size)
                    pygame.draw.rect(self.bg, (0, 204, 102), coords)

                # Draw Wall in black color.
            # for s in self.edges:
            #     (x, y) = self.indx2coord(s)
            #     coords = pygame.Rect(y - self.size / 2, x - self.size / 2, self.size, self.size)
            #     coords = pygame.Rect(y, x, self.size, self.size)
            #     pygame.draw.rect(self.bg, (192, 192, 192), coords)  # the obstacles are in color grey

            for s in self.obstacles:
                (x, y) = self.indx2coord(s)
                coords = pygame.Rect(y, x, self.size, self.size)
                pygame.draw.rect(self.bg, (255, 0, 0), coords)  # the obstacles are in color red

            color = {'sand': (223, 225, 179), 'gravel': (255, 255, 255), 'grass': (211, 255, 192),
                     'pavement': (192, 255, 253),'deterministic': (255,255,255)}
            for s in range(self.nstates):
                if not any(s in x for x in self.targets) and s not in self.obstacles and not any(s in x for x in self.colorstates):
                    (x, y) = self.indx2coord(s)
                    coords = pygame.Rect(y - self.size / 2, x - self.size / 2, self.size, self.size)
                    coords = pygame.Rect(y, x, self.size, self.size)
                    pygame.draw.rect(self.bg, color[self.getStateRegion(s)], coords)  # the obstacles are in color grey
            statecols = [(0,0,0),(150,150,150)]
            for i in range(len(self.colorstates)):
                for s in self.colorstates[i]:
                    if not any(s in x for x in self.targets) and s not in self.obstacles:
                        (x, y) = self.indx2coord(s)
                        coords = pygame.Rect(y, x, self.size, self.size)
                        pygame.draw.rect(self.bg, statecols[i], coords)  # the obstacles are in color grey

        self.bg_rendered = True  # don't render again unless flag is set
        self.surface.blit(self.bg, (0, 0))
    
    def play(self,policy=None):
        
        while self.current[0] not in self.targets[0]:
            for idx_j,j in enumerate(self.current):
                self.render()
                if policy is None:
                    while True:
                        arrow = self.getkeyinput()
                        if arrow != None:
                            break
                else:
                    arrow = self.actlist[next(iter(policy[j]))]
                    pygame.time.wait(500)
                self.current[idx_j] = int(np.random.choice(range(self.prob[arrow][self.current].reshape(-1,).shape[0]),None,False,self.prob[arrow][self.current].reshape(-1,)))
                # self.current = [int(np.random.choice(self.prob[arrow][self.current].reshape(-1,),None,False,self.prob[arrow][self.current].reshape(-1,)))]
                self.render()
        return print("Goal!")
    
    def observation(self,nom_pol,est_loc,agent_no,t):
        self.initial[agent_no]