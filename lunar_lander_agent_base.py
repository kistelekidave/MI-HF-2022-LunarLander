import numpy as np

# np.random.seed(0)


# The resolution of the observation space = Felbontas
# The four variables of the observation space, from left to right:
#   0: X component of the vector pointing to the middle of the platform from the lander
#   1: Y component of the vector pointing to the middle of the platform from the lander
#   2: X component of the velocity vector of the lander
#   3: Y component of the velocity vector of the lander
OBSERVATION_SPACE_RESOLUTION = [31, 11, 15, 15]  # TargetX, TargetY, VelocityX, VelocityY


class LunarLanderAgentBase:
    def __init__(self, observation_space, action_space, n_iterations):
        # self.temp_q_table = []
        self.observation_space = observation_space  # [0][0] : TargetX min , [0][1] : TargetX max , [1][0] : TargetY min
        self.q_table = np.zeros(
            [*OBSERVATION_SPACE_RESOLUTION, len(action_space)])  # [TargetX][TargetY][VelocityX][VelocityY][Action]
        self.env_action_space = action_space  # elvegezheto cselekvesek
        self.n_iterations = n_iterations

        self.epsilon = 1.  # felfedezes eselye
        self.iteration = 0
        self.test = False  # teszteles elesben, itt mar nem szabad felfedezni

        # self.alpha = 0.7  # ilyen aranyban irja be a visszajelzest a q-tablaba


    @staticmethod
    def quantize_state(observation_space, state):  # allapot egesz reszre kerekitese
        targetX = round((state[0] + 300) / 20)
        # targetY = round((state[1]) / 20)
        targetY = round(np.log2(state[1] + 1) * 10 / 7.64)
        velocityX = round((state[2] + 7))
        velocityY = round((state[3] + 7))
        return targetX, targetY, velocityX, velocityY

    def epoch_end(self, epoch_reward_sum):  # minden menet utan meghivott tanulo fv
        pass
        # self.epsilon = self.epsilon / 1.001
        # for element in self.temp_q_table:
        #     element[2] += epoch_reward_sum / 10
        #
        #     quantized_state = self.quantize_state(self.observation_space, element[0])
        #
        #     self.q_table[quantized_state][element[1]] = \
        #         self.q_table[quantized_state][element[1]] * (1 - self.alpha) + \
        #         element[2] * self.alpha

    def learn(self, old_state, action, new_state, reward):  # minden lepes utan meghivott tanulo fv
        quantized_state = self.quantize_state(self.observation_space, old_state)

        self.q_table[quantized_state[0]][quantized_state[1]][quantized_state[2]][quantized_state[3]][action] = reward

        # self.temp_q_table.append([old_state, action, reward])

    def train_end(self):  # teljes tanulas vege

        # self.q_table = None  # TODO
        self.test = True
