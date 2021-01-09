import math
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from scipy.optimize import root_scalar


class State():
    g = 10
    # alpha = 1 # rotational inertia to mass ratio

    w = 15
    h = 3
    r = math.sqrt((w / 2) ** 2 + (h / 2) ** 2)
    # when it's laying on the long side, what's the angle between the ground and the line from the
    # lower-right corner to the center? that's phi.
    # this means theta=90-phi rotates it clockwise just enough to rest on that corner
    # this means that r*sin(theta+phi) is the height of the COM if that right corner is touching
    phi = math.atan2(h, w)
    mass = 1
    moi = mass * 1 / 12 * (3 * (w / 2) ** 2 + h ** 2)
    #moi = mass * 1/12 * (w**2 + h**2) # box

    e_flip = mass * g * r

    def __init__(self, v, theta, omega):
        self.v = v#19
        self.theta = theta#2
        self.omega = omega#3
        # rests on corner
        #self.v = 15
        #self.theta = 1.2
        #self.omega = 5
        self.start_height = self.get_start_height()
        self.energy = self.get_energy()

        # no real solutions to energy thing
        # self.v = 20
        # self.theta = .1
        # self.omega = 0

        # accidentally great for testing time of first hit
        # self.v = 5
        # self.theta = 2.25
        # self.omega = 3

    def get_energy(self):
        ke = .5 * self.mass * self.v ** 2
        rke = .5 * self.moi * self.omega ** 2
        pe = self.mass * self.g * self.get_start_height()
        return ke + rke + pe

    def get_start_height(self):
        # return self.r * math.sin(self.phi + self.theta)
        corners = self.get_corners(0, self.theta)
        ys = corners[1]
        return -min(ys)

    def get_com_height(self, t):
        return -0.5 * self.g * t ** 2 + self.v * t + self.start_height

    def get_pos_at_time(self, t):
        height = self.get_com_height(t) #-0.5 * self.g * t ** 2 + self.v * t + self.start_height
        theta = self.theta + self.omega * t
        return height, theta

    def get_height(self, t):
        # height of lowest corner at time t
        theta = self.theta + t * self.omega
        # y = -abs(math.cos(theta) * self.h / 2) - abs(math.sin(theta) * self.w / 2)
        # ret = self.get_com_height(t) + y
        cs = self.get_corners(self.get_com_height(t), theta)
        return min(cs[1])

    def collision_time(self):
        # assuming we are paused with COM at given height with given theta and omega
        # t_min is first time it could possibly hit regardless of rotation

        '''
        # we reproduce these to avoid passing self to scipy ... although it probably is ok?
        def get_com_height(t):
            return -0.5 * self.g * t ** 2 + self.height
        def get_height(t):
            theta = self.theta + t * self.omega
            y = -abs(math.cos(theta) * self.h / 2) - abs(math.sin(theta) * self.w / 2)
            return get_com_height(t) + y
        '''

        '''
        TODO this is not valid with our new notion of height, and also with the realization
        That we sometimes hit the ground before returning to max height

        # -0.5 * g * t ** 2 + self.height <= r, find minimum positive t
        # t**2 >= 0.5*g*(self.height - r)
        t_min = 0 if self.height < self.r else math.sqrt(1/(.5*self.g)*(self.height - self.r))
        t_max = 0 if self.height < -self.r else math.sqrt(1/(.5*self.g)*(self.height + self.r))
        # if we're in range, a corner will hit before finishing a 180 degree rotation
        t_max = min(t_max, t_min + math.pi/abs(self.omega))
        ts = np.linspace(t_min, t_max, 1000)
        ys = np.array([self.get_height(t) for t in ts])
        hit = ys < 0
        i = np.nonzero(hit)[0][0]
        t_min2, t_max2 = ts[i-1], ts[i]
        '''

        t_min = 1e-6
        t_max = 10

        ts = np.linspace(t_min, t_max, 1000)
        ys = np.array([self.get_height(t) for t in ts])

        #assert ys[0] >= 0, "tmin not small enough; probably coming to rest on a corner"
        if ys[0] < 0:
            print('TMIN ISSUE')
            # caller should special-case this
            return 0
            #return t_min

        first_hit = np.where(ys < 0)[0][0]
        t_min2 = ts[first_hit - 1]
        t_max2 = ts[first_hit]

        # print('Check a, b', self.get_height(t_min2), self.get_height(t_max2))
        # return 100
        #assert self.get_height(t_min2) >= 0
        res = root_scalar(self.get_height, bracket=(t_min2, t_max2), x0=t_min2)
        t = res.root

        #ts = np.linspace(t_min, t*1.1+.5, 1000)
        #ys = [self.get_height(t) for t in ts]
        ##plt.figure()
        #plt.plot(ts, ys)
        #plt.plot([t], [self.get_height(t)], '*')
        #plt.grid()
        ##plt.ion()
        #plt.show()


        return t

    # if the coin is turned theta radians from lying on the horizontal axis, what's the dot product of the floor
    # normal and PERPENDICULAR TO the direction from the contact point to the COM?
    # in other words, what percentage of impulse to the bottom corner goes to CCW rotation?
    def f(self, theta):
        # c1 = math.atan2(self.w, self.h)
        # return math.sin(c1 + theta)
        # that method only works for one corner
        cs = list(zip(*self.get_corners(0, theta)))
        c = min((c[1], c) for c in cs)[1]
        return c[0] / self.r

    def get_side(self, theta):
        cutoff = math.atan2(self.w, self.h)
        theta = theta % (math.pi*2)
        if theta < cutoff:
            return 'H'
        elif theta < math.pi - cutoff:
            return 'S'
        elif theta < math.pi + cutoff:
            return 'T'
        elif theta < 2*math.pi - cutoff:
            return 'S'
        else:
            return 'H'

    def next_state(self, t):
        # assuming we hit the ground after t seconds, what's our status just after hitting the ground?
        gamma = 0.2

        end_theta = self.theta + self.omega * t
        end_com_height = self.get_com_height(t)
        end_v = self.v + -self.g * t
        end_omega = self.omega

        f = self.f(end_theta)
        # print('\nf:', f)
        # THIS IS THE OLD BROKEN VELOCITY VERSION
        # x = -(gamma+1)*(end_v + f*self.r*self.omega) / (1/self.mass + f**2 * self.r / self.moi)

        # we want to find the quadratic equation for energy(impulse)
        # amazingly, we know denergy(impulse)/dimpulse is just v_final(impulse)
        a = 1/2*((1/self.mass) + ((f*self.r)**2/self.moi))
        b = end_v + (f*self.r)*end_omega
        c = 1/2*self.mass*end_v**2 + 1/2 * self.moi * end_omega**2

        perfectly_inelastic_impulse = -b / (2*a)

        # force should be up - note that this is equivalent to saying b (velocity) is down,
        # which makes good physical sense
        assert perfectly_inelastic_impulse > 0

        # resulting energy after a perfectly inelastic collision
        minimum_energy = a*perfectly_inelastic_impulse**2 + b*perfectly_inelastic_impulse + c

        # of the energy put into the floor (excess above minimum_energy), we keep fraction gamma
        new_energy = minimum_energy + gamma * (c - minimum_energy)
        new_c = c - new_energy
        impulse = (-b + math.sqrt(b**2 - 4*a*new_c)) / (2*a)
        #print('energy into floor', c-minimum_energy)
        #print('Kinetic energy before', c)
        #print('Kinetic energy after (desired)', new_energy)


        new_v = end_v + impulse / self.mass
        new_omega = self.omega + impulse * f * self.r / (self.moi)
        new_theta = end_theta

        if self.get_energy() < self.e_flip:
            side = self.get_side(new_theta)
            return side

        s = State(new_v, new_theta, new_omega)
        #print('Kinetic energy after (calculated)', s.energy - s.get_start_height()*s.mass*s.g)
        #print('New V', s.v + s.f(s.theta)*s.omega*self.r)
        #print(s.v, s.f(s.theta), s.omega)
        #print()

        #print(s.get_corners(*s.get_pos_at_time(0))[1])
        #print(s.get_corners(*s.get_pos_at_time(1e-12))[1])
        #print(s.get_corners(*s.get_pos_at_time(1e-9))[1])
        #print(s.get_corners(*s.get_pos_at_time(1e-6))[1])


        return s

    def plot_at_time(self, t):
        height = self.get_com_height(t)
        theta = self.theta + t * self.omega
        self.plot(height, theta)

    def get_corners(self, height=None, theta=None):

        if height is None:
            height = self.height
        if theta is None:
            theta = self.theta

        xs = []
        ys = []
        for a, b in [[-0.5, -0.5], [-0.5, 0.5], [0.5, 0.5], [0.5, -0.5]]:
            x = a * self.w
            y = b * self.h
            xx = math.cos(theta) * x - math.sin(theta) * y
            yy = height + math.cos(theta) * y + math.sin(theta) * x
            xs.append(xx)
            ys.append(yy)

        return xs, ys

    def plot(self, height=None, theta=None):
        xs, ys = self.get_corners(height, theta)
        xs.append(xs[0])
        ys.append(ys[0])
        plt.plot(xs, ys, '-')
        plt.plot([0], [height], '*')
        plt.plot([-self.w, self.w], [0, 0], '--')

        ax = plt.gca()
        ratio = 1.0
        xleft, xright = ax.get_xlim()
        ybottom, ytop = ax.get_ylim()
        size = max(xright - xleft, ytop - ybottom)
        extents = lambda left, right, size: ((left + right) / 2 - size / 2, (left + right) / 2 + size / 2)
        ax.set_xlim(extents(xleft, xright, size * ratio))
        ax.set_ylim(extents(ybottom, ytop, size))
        # the abs method is used to make sure that all numbers are positive
        # because x and y axis of an axes maybe inversed.
        ax.set_aspect(ratio)

        plt.show()

    @classmethod
    def animate(cls, state, t_start, t_stop):
        fig1 = plt.figure()

        line, = plt.plot([], [], '-')
        plt.xlim(-10, 10)
        plt.ylim(0, 20)

        cls.state = state
        cls.t_tr = cls.state.collision_time()
        cls.t_tr_total = 0

        def update_line(t):
            if t >= cls.t_tr_total + cls.t_tr:
                # TODO how to update this? probably want a static function
                cls.state = cls.state.next_state(cls.t_tr)
                cls.t_tr_total += cls.t_tr
                cls.t_tr = cls.state.collision_time()
            height, theta = cls.state.get_pos_at_time(t - cls.state.t_tr_total)
            xs, ys = cls.state.get_corners(height, theta)
            xs.append(xs[0])
            ys.append(ys[0])
            line.set_data((xs, ys))
            return line,

        fps = 30
        speed = 2
        line_ani = animation.FuncAnimation(fig1, update_line,
                                           np.linspace(t_start, t_stop, int((t_stop - t_start) * fps)) * speed,
                                           interval=(1000 / fps))
        plt.show()

    def get_data(self):
        tmax = self.collision_time() * 1.1
        ts = np.linspace(0, tmax, 1000)
        hs = [self.get_height(t) for t in ts]
        return ts, hs

    @classmethod
    def get_result(cls, state):
        while type(state) == State:
            t = state.collision_time()
            state = state.next_state(t)
        return state

d1 = State(19, 2, 3)
ts, hs = d1.get_data()
print('COLLISIONTIME', ts[-1])
print('MAXHEIGHT', max(hs))
def lerp(x):
    if x <= ts[0]:
        return 0
    elif x >= ts[-1]:
        return 0
    print(x)
    print(ts[:10])
    index = np.where(ts > x)[0][0]
    return hs[index]


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    #plt.plot(ts, hs)
    #plt.show()

    #d1 = State(15, .02 + math.pi/2, 0)
    #d1 = State(3, math.pi/4, 1.1)

    for i in range(45):
        d = State(4, 8*i*math.pi/180, 0)
        print(8*i, State.get_result(d))
    exit()

    d1 = State(3, 108*math.pi/180, 0)

    #for i in range(10):
    #    t = d1.collision_time()
    #    d1 = d1.next_state(t)
    #exit()

    State.animate(d1, 0, 1000)
    #d2 = d1.next_state(t)

    # print(t)
    # d1.plot_at_time(t)


