from pylab import *
import matplotlib.pylab as plt
import numpy
import numpy as np
from time import sleep
import matplotlib.animation as animation
from mpl_toolkits.axes_grid1 import AxesGrid


class Gridworld(object):
    """
    A class that implements a quadratic NxN gridworld. 
    
    Methods:
    
    learn(N_trials=1)  : Run 'N_trials' trials. A trial is finished, when the agent reaches the reward location.
    visualize_trial()  : Run a single trial with graphical output.
    reset()            : Make the agent forget everything he has learned.
    plot_Q()           : Plot of the Q-values .
    learning_curve()   : Plot the time it takes the agent to reach the target as a function of trial number. 
    navigation_map()     : Plot the movement direction with the highest Q-value for all positions.
    """

    def __init__(self, N, reward_position=(0, 0), obstacle=False,
                 lambda_eligibility=0.,
                 learning_rate=0.1,
                 n_max=10000,
                 discount_factor=0.99,exploration_factor=0.5, Train_length=25):
        """
        Creates a quadratic NxN gridworld. 

        Mandatory argument:
        N: size of the gridworld

        Optional arguments:
        reward_position = (x_coordinate,y_coordinate): the reward location
        reward_position = (x_coordinate,y_coordinate): the reward location
        obstacle = True:  Add a wall to the gridworld.
        """

        # gridworld size
        self.N = N

        # reward location
        self.reward_position = reward_position

        # reward administered t the target location and when
        # bumping into walls
        self.reward_at_target = 1.
        self.reward_at_wall = -0.5

        # maximal number of steps in a trial
        self.n_max_ = n_max

        # probability at which the agent chooses a random
        # action. This makes sure the agent explores the grid.
        self.epsilon = 0.5
        self.exploration_factor=exploration_factor
        # Number of Trainingsteps with decreasing epsilon
        self.Train_length=Train_length

        # learning rate
        self.eta = learning_rate

        # discount factor - quantifies how far into the future
        # a reward is still considered important for the
        # current action
        self.gamma = discount_factor

        # the decay factor for the eligibility trace the
        # default is 0., which corresponds to no eligibility
        # trace at all.
        self.lambda_eligibility = lambda_eligibility

        # is there an obstacle in the room?
        self.obstacle = obstacle
        self.obstacle_position = (0.65, 0.65)

        # radius for touching reward or obstacle
        self.reward_radius = 0.1

        # initialize the Q-values etc.
        self._init_Qvalues()


    def run(self, N_trials=10, N_runs=1):
        ''' train model for N_trial iterations and take tha average ofer N_runs'''

        self.latency_table = zeros((N_runs, N_trials))
        self.reward_table_ = zeros((N_runs, N_trials))

        for run in range(N_runs):

            if N_runs > 1:
                print 'Started Run %i' % run

            # resets Q-values and latencies, ie forget all he learnt
            self.reset()
            epsilon_range=self._get_epsilon_range(N_trials)

            # list that contains the times it took the agent to reach the target for all trials
            # serves to track the progress of learning
            latency_list = []

            for trial in range(N_trials):

                # run a trial and store the time it takes to the target
                self.epsilon = epsilon_range[trial]
                latency, total_reward = self._run_trial()
                latency_list.append(latency)
                self.reward_table_[run, trial] = total_reward
                self.latency_table[run, trial] = latency

        self.latencies = self.latency_table.mean(0)

    def visualize_trial(self, visualize=True, verbose=True, animate=False):
        """
        Run a single trial with a graphical display that shows in
                red   - the position of the agent
                blue  - walls/obstacles
                green - the reward position

        Note that for the simulation, exploration is reduced -> self.epsilon=0.1
    
        """
        # store the old exploration/exploitation parameter
        # epsilon = self.epsilon

        # favor exploitation, i.e. use the action with the
        # highest Q-value most of the time
        # self.epsilon = 0.1

        self._run_trial(visualize=(visualize or animate), verbose=verbose)

        if visualize:
            self._visualize_trial()

        if animate:
            self._animation()

        # restore the old exploration/exploitation factor
        # self.epsilon = epsilon

    def learning_curve(self, log=False, filter=1.):
        """
        Show a running average of the time it takes the agent to reach the target location.

        Options:
        filter=1. : timescale of the running average.
        log    : Logarithmic y axis.
        """
        plt.figure()  # a matplotlib figure instance
        plt.xlabel('trials')
        plt.ylabel('time to reach target')
        latencies = np.median(self.latency_table, 0)
        latency_table = self.latency_table.copy()
        x = range(1, len(latencies) + 1)

        plt.plot(x, latency_table.T, c=(0.2980392156862745, 0.4470588235294118, 0.6901960784313725), linewidth=1.5)

        # calculate a running average over the latencies with a averaging time 'filter'
        # for i in range(1, latencies.shape[0]):
        #     latencies[i] = latencies[i-1] + (latencies[i] - latencies[i-1])/float(filter)
        #     # latency_table[ :,i]=latency_table[:,i-1]+(latency_table[:,i]-latency_table[:,i-1])/float(filter)


        plt.plot(x, latencies, 'k', linewidth=3, label='median')
        plt.legend()
        plt.xlim([1, len(latencies)])
        if log:
            plt.yscale('log')

        plt.draw()

    def integrated_reward(self, log=False):
        """
        the total reward that was received on each trial.

        Options:
        log    : Logarithmic y axis.
        """
        figure()  # a matplotlib figure instance
        xlabel('trials')
        ylabel('integrated reward')

        reward_m = np.median(self.reward_table_, 0)
        x = range(1, len(reward_m) + 1)
        plt.plot(x, self.reward_table_.T, c=(0.2980392156862745, 0.4470588235294118, 0.6901960784313725), linewidth=1.5)

        plt.plot(x, reward_m, 'k', linewidth=3, label='median')
        plt.legend()
        plt.xlim([1, len(reward_m) + 1])

        if log:
            plt.yscale('log')

        plt.draw()

    def navigation_map(self):
        """
        Plot the direction with the highest Q-value for every position.
        Useful only for small gridworlds, otherwise the plot becomes messy.
        """
        self.x_direction = numpy.zeros((self.N, self.N))
        self.y_direction = numpy.zeros((self.N, self.N))

        self.actions = argmax(self.Q[:, :, :], axis=2)
        self.y_direction[self.actions == 0] = 1.
        self.y_direction[self.actions == 1] = -1.
        self.y_direction[self.actions == 2] = 0.
        self.y_direction[self.actions == 3] = 0.

        self.x_direction[self.actions == 0] = 0.
        self.x_direction[self.actions == 1] = 0.
        self.x_direction[self.actions == 2] = 1.
        self.x_direction[self.actions == 3] = -1.

        figure()
        quiver(self.x_direction, self.y_direction)
        axis([-0.5, self.N - 0.5, -0.5, self.N - 0.5])

    def reset(self):
        """
        Reset the Q-values.

        Instant amnesia -  the agent forgets everything he has learned before
        """
        self._init_Qvalues()

    def plot_Q(self):
        """
        Plot the dependence of the Q-values on position.
        The figure consists of 4 subgraphs, each of which shows the Q-values 
        colorcoded for one of the actions.
        """
        figure()
        for i in range(4):
            subplot(2, 2, i + 1)
            imshow(self.Q[:, :, i], interpolation='nearest', origin='lower', vmax=1.1)
            if i == 0:
                title('Up')
            elif i == 1:
                title('Down')
            elif i == 2:
                title('Right')
            else:
                title('Left')

            colorbar()
        draw()

    ###############################################################################################
    # The remainder of methods is for internal use and only relevant to those of you
    # that are interested in the implementation details
    ###############################################################################################


    def _init_Qvalues(self):
        """
        Initialize the Q-values.
        """

        # initialize the Q-values and the eligibility trace
        self.Q = 0.01 * numpy.random.rand(self.N, self.N, 4) + 0.1

    def _get_epsilon_range(self,N):

        # probability at which the agent chooses a random
        # action. This makes sure the agent explores the grid.
        lr = self.exploration_factor
        if isinstance(lr, float) or isinstance(lr, int):
            # constant value generator
            return np.ones(N) * lr
        else:
            assert len(lr) == 2  # 'lr has to be of length one or two'
            assert lr[1]<=lr[0]

            t = np.arange(N)
            x0 = lr[0]
            x_end = lr[1]
            #Linear

            x=np.ones(N) #*x_end
            # decreasing epsilon until train end
            Train_end= self.Train_length
            index_Train_end=min(Train_end,N)

            x[:index_Train_end] += t[:index_Train_end] * (x_end-x0) / float(Train_end)
            x[index_Train_end:]=x_end

            return x
            # eponentiel
            # return np.array(x0* (x_end/x0)**(t/float(self.n_max_)))




    def init_trial_(self, visualize=True, verbose=False):

        """
        initialize the state and action variables

        :param visualize:
        :param verbose:
        """
        # choose the initial position and make sure that its not in the wall
        while True:
            self.x_position = numpy.random.randint(self.N)
            self.y_position = numpy.random.randint(self.N)
            if not self._is_wall(self.x_position, self.y_position):
                break

        self.action = None

        self.e = numpy.zeros((self.N, self.N, 4))

    def _run_trial(self, visualize=True, verbose=False):
        """
        Run a single trial on the gridworld until the agent reaches the reward position.
        Return the time it takes to get there.

        Options:
        visual: If 'visualize' is 'True', show the time course of the trial graphically

        text output if 'verbose is 'True' , write out movements
        """

        self.init_trial_(visualize=visualize, verbose=verbose)
        # initialize constant or decreasing epsilon generators



        self.pos_history_ = []  # save positions

        if verbose:
            print "Starting trial at position ({0},{1}), reward at ({2},{3})".format(self.x_position, self.y_position,
                                                                                     self.reward_position[0],
                                                                                     self.reward_position[1])
            if self.obstacle:
                print "Obstacle is in position (?,?)"

        # initialize the latency (time to reach the target) for this trial
        latency = 0.
        total_reward = 0

        # run the trial
        self._choose_action()
        while not self._arrived() and latency < self.n_max_:
            self._update_state(verbose=verbose)
            self._choose_action()
            self._update_Q()

            # if visualize:
                # self._visualize_current_state()
            self.pos_history_.append((self.x_position, self.y_position))


            latency = latency + 1
            total_reward += self._reward()
        if verbose:
            if self._arrived():
                print "Arrived at goal after %i steps " % latency
            elif latency >= self.n_max_:
                print "Mouse took more than %i steps trial aborted! " % self.n_max_

                # if visualize:
                # self._close_visualization()

        return latency, total_reward

    def _update_Q(self):
        """
        Update the current estimate of the Q-values according to SARSA.
        """
        # update the eligibility trace
        self.e = self.lambda_eligibility * self.e
        self.e[self.x_position_old, self.y_position_old, self.action_old] += 1.

        # update the Q-values
        if self.action_old != None:
            self.Q += \
                self.eta * self.e * \
                (self._reward() \
                 - (self.Q[self.x_position_old, self.y_position_old, self.action_old] \
                    - self.gamma * self.Q[self.x_position, self.y_position, self.action]))

    def _choose_action(self):
        """
        Choose the next action based on the current estimate of the Q-values.
        The parameter epsilon determines, how often agent chooses the action 
        with the highest Q-value (probability 1-epsilon). In the rest of the cases
        a random action is chosen.
        """
        self.action_old = self.action
        if numpy.random.rand() < self.epsilon:
            self.action = numpy.random.randint(4)
        else:
            self.action = argmax(self.Q[self.x_position, self.y_position, :])

    def _arrived(self):
        """
        Check if the agent has arrived.
        """
        return (self.x_position == self.reward_position[0] and self.y_position == self.reward_position[1])

    def _reward(self):
        """
        Evaluates how much reward should be administered when performing the 
        chosen action at the current location
        """
        if self._arrived():
            return self.reward_at_target

        if self._wall_touch:
            return self.reward_at_wall
        else:
            return 0.

    def _update_state(self, verbose=False):
        """
        Update the state according to the old state and the current action.    
        """
        # remember the old position of the agent
        self.x_position_old = self.x_position
        self.y_position_old = self.y_position

        # update the agents position according to the action
        #  move right
        if self.action == 0:
            self.x_position += 1
            if verbose:
                print "({0},{1}) >>> ({2},{3})".format(self.x_position_old, self.y_position_old, self.x_position,
                                                       self.y_position)
        # move left
        elif self.action == 1:
            self.x_position -= 1
            if verbose:
                print "({0},{1}) <<< ({2},{3})".format(self.x_position_old, self.y_position_old, self.x_position,
                                                       self.y_position)
        # move up
        elif self.action == 2:
            self.y_position += 1
            if verbose:
                print "({0},{1}) ^^^ ({2},{3})".format(self.x_position_old, self.y_position_old, self.x_position,
                                                       self.y_position)
        # move down
        elif self.action == 3:
            self.y_position -= 1
            if verbose:
                print "({0},{1}) vvv ({2},{3})".format(self.x_position_old, self.y_position_old, self.x_position,
                                                       self.y_position)
        else:
            print "There must be a bug. This is not a valid action!"

        # check if the agent has bumped into a wall.
        if self._is_wall():
            self.x_position = self.x_position_old
            self.y_position = self.y_position_old
            self._wall_touch = True
            if verbose:
                print "#### wally ####"
        else:
            self._wall_touch = False

    def _is_wall(self, x_position=None, y_position=None):
        """
        This function returns, if the given position is within an obstacle
        If you want to put the obstacle somewhere else, this is what you have 
        to modify. The default is a wall that starts in the middle of the room
        and ends at the right wall.

        If no position is given, the current position of the agent is evaluated.
        """
        if x_position is None or y_position is None:
            x_position = self.x_position
            y_position = self.y_position

        # check of the agent is trying to leave the gridworld
        if x_position < 0 or x_position >= self.N or y_position < 0 or y_position >= self.N:
            return True

        # check if the agent has bumped into an obstacle in the room
        if self.obstacle:
            if y_position == self.N / 2 and x_position > self.N / 2:
                return True

        # if none of the above is the case, this position is not a wall
        return False

    def _visualize_current_state(self):
        """
        Show the gridworld. The squares are colored in 
        red - the position of the agent - turns yellow when reaching the target or running into a wall
        blue - walls
        green - reward
        """

        # set the agents color
        self._display[self.x_position_old, self.y_position_old, 0] = 0
        self._display[self.x_position_old, self.y_position_old, 1] = 0
        self._display[self.x_position, self.y_position, 0] = 1
        if self._wall_touch:
            self._display[self.x_position, self.y_position, 1] = 1

        # set the reward locations
        self._display[self.reward_position[0], self.reward_position[1], 1] = 1

        # update the figure
        self._visualizations.append((imshow(self._display, interpolation='nearest', origin='lower'),))



        # draw()


        # and wait a little while to control the speed of the presentation
        # sleep(0.2)

    def _init_visualization(self):

        # create the figure
        fig = figure()
        # initialize the content of the figure (RGB at each position)
        self._display = numpy.zeros((self.N, self.N, 3))

        # position of the agent
        self._display[self.x_position, self.y_position, 0] = 1
        self._display[self.reward_position[0], self.reward_position[1], 1] = 1

        for x in range(self.N):
            for y in range(self.N):
                if self._is_wall(x_position=x, y_position=y):
                    self._display[x, y, 2] = 1.

        self._visualizations = []
        self._visualizations.append((imshow(self._display, interpolation='nearest', origin='lower'),))

        return fig

    def _close_visualization(self):
        # print "Press <return> to proceed..."
        # raw_input()
        close()

    def _plot_dots(self):

        # plt.plot(self.reward_position[0],self.reward_position[1],'ro',markersize=15,label='reward')

        circle1 = plt.Circle((self.reward_position[0], self.reward_position[1]), self.reward_radius, color='r',
                             alpha=0.6)
        plt.gca().add_artist(circle1)

        pl = plt.Line2D([0.1], [0.1], color='g', markersize=10., marker='o', linewidth=0)
        plt.gca().add_artist(pl)

        handlers = [pl, circle1]
        labels = ['initial position', 'goal']

        if self.obstacle:
            # plt.plot(self.obstacle_position[0],self.obstacle_position[1],'bo',markersize=15,label='obstacle')
            circle2 = plt.Circle((self.obstacle_position[0], self.obstacle_position[1]), self.reward_radius, color='b',
                                 alpha=0.6)
            plt.gca().add_artist(circle2)
            handlers.append(circle2)
            labels.append('obstacle')

        plt.legend(handlers, labels, bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
                   ncol=3, mode="center", borderaxespad=0.)

    def _visualize_trial(self):
        positions = np.array(self.pos_history_)

        plt.figure(figsize=(5.5, 5.5))
        plt.subplot(aspect='equal')
        plt.plot(positions[:, 0], positions[:, 1], 'ko-', alpha=0.1)

        plt.xlim(0, 1)
        plt.ylim(0, 1)

        self._plot_dots()

        plt.draw()

    def _animation(self):
        positions = np.array(self.pos_history_)
        plt.subplot(aspect='equal')

        ## Animation

        def update_line(num, data, line):
            line.set_data(data[..., :num])
            return line,

        fig1 = plt.figure()

        plt.plot(positions[:, 0], positions[:, 1], 'k', alpha=0.1)

        self._plot_dots()

        l, = plt.plot([], [], 'ko', alpha=0.2)
        plt.xlim(0, 1)
        plt.ylim(0, 1)

        line_ani = animation.FuncAnimation(fig1, update_line, positions.shape[0], fargs=(positions.T, l),
                                           interval=30, blit=True)
        # line_ani.save('lines.mp4')


        plt.show()

    def latency_score(self, N_values=10):
        """
        :return: takes mean over last N values
        value of plateau
        """

        assert self.latency_table.shape[-1] >= N_values, 'nead more than %i values ' % N_values
        return self.latency_table[:, -N_values:].mean(1)


def kernel_function(x, y, centers, sigma2):
    """
    rbf function
    :param x: x pos
    :param y: y pos
    :param centers: [x,y] pos ov center
    :param sigma2: squared value of sigma
    :return: kernel function evaluated at position
    """
    return np.exp(-(np.square(x - centers[0]) + np.square(y - centers[1])) / (2 * sigma2))


class ContinuousWorld(Gridworld):
    def __init__(self, reward_position=(0.8, 0.8), obstacle=False, lambda_eligibility=0.95, n_max=5000,
                 learning_rate=0.005, exploration_factor=0.5, discount_factor=0.95,Train_length=25):

        ## Fixed variables
        # Size of Grid
        N = 20
        # number of directions one can choose
        self.n_actions_ = 8

        # initialize positions of neurons
        centers = np.array(np.meshgrid(np.arange(N), np.arange(N)))
        ds = 1. / (N - 1)
        centers = centers * ds
        self.centers_ = centers

        super(ContinuousWorld, self).__init__(N, reward_position=reward_position, obstacle=obstacle,
                                              lambda_eligibility=lambda_eligibility, learning_rate=learning_rate,
                                              n_max=n_max, discount_factor=discount_factor,
                                              exploration_factor=exploration_factor,
                                              Train_length=Train_length)

        self.step_size_ = 0.03
        self.sigma_kernel2_ = 0.05 ** 2  # squared sigma value

        # reward administered t the target location and when
        # bumping into walls
        self.reward_at_target = 10.
        self.reward_at_wall = -2.




    def init_trial_(self, visualize=False, verbose=False):

        """
        initialize the state and action variables

        :param visualize:
        :param verbose:
        """
        self.x_position = 0.1
        self.y_position = 0.1
        self.action = None
        self.e = numpy.zeros((self.N, self.N, self.n_actions_))

    def _init_Qvalues(self):
        """
        Initialize the Q-values.
        weights of the neuronal network
        """
        self.weights_ = self.eta / 20. * (-1 + 2 * numpy.random.rand(self.N, self.N, self.n_actions_))  # + 0.1
        # self.weights_ = np.zeros((self.N, self.N, self.n_actions_))

    def get_Qvalue_(self, x_position, y_position, action=None):
        discrete_input = kernel_function(x_position, y_position, self.centers_, self.sigma_kernel2_)

        if action is None:
            input3d = np.repeat(discrete_input[:, :, np.newaxis], self.n_actions_, 2)
            return (self.weights_ * input3d).sum(0).sum(0)
        else:
            return np.sum(discrete_input * self.weights_[:, :, action])

    def _update_Q(self):
        """
        Update the current estimate of the Q-values according to SARSA.
        """
        # update the eligibility trace
        self.e = self.lambda_eligibility * self.e  # * self.gamma ## acording to SARSA

        self.e[:, :, self.action_old] += kernel_function(self.x_position_old, self.y_position_old, self.centers_,
                                                         self.sigma_kernel2_)

        # update the Q-values
        if self.action_old != None:
            self.weights_ += \
                self.eta * self.e * \
                (self._reward() \
                 - (self.get_Qvalue_(self.x_position_old, self.y_position_old, self.action_old) \
                    - self.gamma * self.get_Qvalue_(self.x_position, self.y_position, self.action)))

    def _choose_action(self):
        """
        Choose the next action based on the current estimate of the Q-values.
        The parameter epsilon determines, how often agent chooses the action
        with the highest Q-value (probability 1-epsilon). In the rest of the cases
        a random action is chosen.
        """
        self.action_old = self.action
        if numpy.random.rand() < self.epsilon:
            self.action = numpy.random.randint(self.n_actions_)
        else:
            self.action = argmax(self.get_Qvalue_(self.x_position, self.y_position))

    def _arrived(self):
        """
        Check if the agent has arrived.
        """
        r = self.reward_radius
        return abs(self.x_position - self.reward_position[0]) < r and abs(self.y_position - self.reward_position[1]) < r

    def _is_wall(self, x_position=None, y_position=None):
        """
        This function returns, if the given position is within an obstacle
        If you want to put the obstacle somewhere else, this is what you have
        to modify. The default is a wall that starts in the middle of the room
        and ends at the right wall.

        If no position is given, the current position of the agent is evaluated.
        """
        if x_position is None or y_position is None:
            x_position = self.x_position
            y_position = self.y_position

        # check of the agent is trying to leave the gridworld
        if x_position <= 0 or x_position >= 1 or y_position <= 0 or y_position >= 1:
            return True

        # check if the agent has bumped into an obstacle in the room
        if self.obstacle:
            r = self.reward_radius
            if abs(self.x_position - self.obstacle_position[0]) < r \
                    and abs(self.y_position - self.obstacle_position[1]) < r:
                return True

        # if none of the above is the case, this position is not a wall
        return False

    def _update_state(self, verbose=False):
        """
        Update the state according to the old state and the current action.
        """
        # remember the old position of the agent
        self.x_position_old = self.x_position
        self.y_position_old = self.y_position

        # update the agents position according to the action

        assert self.action < self.n_actions_, "There must be a bug. This is not a valid action!"

        direction = 2 * np.pi * self.action / self.n_actions_
        self.x_position += self.step_size_ * np.cos(direction)
        self.y_position += self.step_size_ * np.sin(direction)

        if verbose:
            print "({0:4.2f},{1:4.2f}) a={4} ({2:4.2f},{3:4.2f})".format(self.x_position_old, self.y_position_old,
                                                                         self.x_position,
                                                                         self.y_position, self.action)

        # check if the agent has bumped into a wall.
        if self._is_wall():
            self.x_position = self.x_position_old
            self.y_position = self.y_position_old
            self._wall_touch = True
            if verbose:
                print "#### wally ####"
        else:
            self._wall_touch = False

    def navigation_map(self, K=25):

        plt.figure(figsize=(5.5, 5.5))
        plt.subplot(aspect='equal')
        X, Y = np.meshgrid(np.linspace(0, 1, K + 1), np.linspace(0, 1, K + 1))

        best_actions = np.zeros((K + 1, K + 1))

        for i in range(K + 1):
            for j in range(K + 1):
                best_actions[i, j] = np.argmax(self.get_Qvalue_(X[i, j], Y[i, j]))

        dX = np.cos(2 * np.pi / self.n_actions_ * best_actions)
        dY = np.sin(2 * np.pi / self.n_actions_ * best_actions)
        plt.quiver(X, Y, dX, dY, scale=20)

        self._plot_dots()

        plt.draw()

    def plot_Q(self, v_max=1.):

        fig = plt.figure(figsize=(7, 10))
        ax = AxesGrid(fig, 111, nrows_ncols=(2, self.n_actions_ / 2), axes_pad=0.2,
                      label_mode="1", share_all=True,
                      cbar_location="top", cbar_mode="single",
                      cbar_size="7%", cbar_pad="2%")

        for i in range(self.n_actions_):
            x1, y1 = np.cos(2 * np.pi / self.n_actions_ * i), np.sin(2 * np.pi / self.n_actions_ * i)

            img = ax[i].pcolor(self.weights_[:, :, i], cmap='RdBu_r', vmax=v_max, vmin=-v_max)
            ax[i].arrow(self.N / 2., self.N / 2., x1, y1, head_width=1, head_length=1, fc='k', ec='k')
        ax.cbar_axes[0].colorbar(img)

        plt.tight_layout()

        plt.draw()

    def plot_neuron_arrows(self, colorrange=[0, 2],visualize=False):
        """ Plot arrows from each neuron center and color with strength """

        best_actions = np.argmax(self.weights_, 2)
        action_strength = np.max(self.weights_, 2)

        dX = np.cos(2 * np.pi / 8 * best_actions)
        dY = np.sin(2 * np.pi / 8 * best_actions)

        plt.figure(figsize=(5.5, 5.5))
        plt.subplot(aspect='equal')
        arrows = plt.quiver(self.centers_[0], self.centers_[1], dX, dY, action_strength, scale=10, headwidth=5,
                            cmap='autumn_r', clim=colorrange)

        if visualize:
            positions = np.array(self.pos_history_)
            plt.plot(positions[:, 0], positions[:, 1], 'ko-', alpha=0.2,markersize=5)


        plt.xlim(-0.05, 1.05)
        plt.ylim(-0.05, 1.05)
        self._plot_dots()

        from mpl_toolkits.axes_grid1 import make_axes_locatable
        divider = make_axes_locatable(plt.gca())
        cax = divider.append_axes("right", "5%", pad="3%")
        cbar = plt.colorbar(arrows, cax=cax)
        cbar.ax.set_ylabel('highest weight value')
        plt.tight_layout()

        plt.draw()


if __name__ == "__main__":
    grid =ContinuousWorld(exploration_factor=[1.,0.1],obstacle=True)
    grid.visualize_trial(verbose=True,animate=True,visualize=False)
    grid.run(N_trials=50,N_runs=1)
    print grid.latency_score()
    grid.visualize_trial(verbose=True,animate=True,visualize=False)

    
    plt.draw()
