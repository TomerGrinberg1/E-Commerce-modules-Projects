import numpy as np
import random



class Planner:
    def __init__(self, num_rounds, phase_len, num_arms, num_users, arms_thresh, users_distribution):
        """
        :input: the instance parameters (see explanation in MABSimulation constructor)
        """
        self.num_rounds = num_rounds
        self.ucb_values = np.zeros((num_arms, num_users))
        self.current_user = -1
        self.chosen_arm = -1
        self.t = 0
        self.phase_len = phase_len
        self.count_end_phase = phase_len
        self.count_test_ucb = 0
        self.flag = True
        self.fix = 0
        self.keepArms = True
        self.arms_thresh = arms_thresh
        self.phase_count = [0] * num_arms
        self.thresh_diff = [0] * num_arms
        self.num_deactivated = 0;
        self.inactive = set()
        self.active = []
        self.thresh_array = [False] * num_arms
        self.deactived = []
        self.users_distribution = users_distribution
        self.average_rewards = np.zeros((num_arms, num_users))
        self.count_arm = np.zeros((num_arms, num_users))
        self.radius = np.zeros((num_arms, num_users))
        self.arms_left = [i for i in range(num_arms)]
        self.critical_phase = 100 - sum(arms_thresh)
        self.num_arms = num_arms
        self.num_users = num_users
        self.init = self.num_users * self.num_arms
        self.thresh_distance()
        self.threshold_flag = self.count_end_phase == sum(self.phase_count)


    def thresh_distance(self):
        for i in range(self.num_arms):
            distance = self.arms_thresh[i] - self.phase_count[i]
            if distance > 0:
                self.thresh_diff[i] = distance
            else:
                self.thresh_diff[i] = 0

    def save_arm(self):
        self.threshold_flag = self.count_end_phase == sum(self.thresh_diff) + 1
        if False not in self.thresh_array:
            self.threshold_flag = False

    def init_ucb(self, user_context):
        for i in range(self.num_arms):
            if self.count_arm[i][user_context] == 0:
                return i
        return -1

    def check_deactive_arms(self):

        self.deactived = [i for i, v in enumerate(self.thresh_array) if v == False]
        self.arms_left = [i for i, v in enumerate(self.thresh_array) if v == True]

    def check_thresh(self):
        for arm in self.arms_left:
            if self.arms_thresh[arm] <= (self.phase_count[arm]):
                self.thresh_array[arm] = True

        self.thresh_distance()
        self.save_arm()

    def ucb_radius(self):
        return (np.sqrt(2 * np.log(self.num_rounds) / self.count_arm[self.chosen_arm][self.current_user]))

    def update_ucb(self, reward):

        temp_mean = self.average_rewards[self.chosen_arm][self.current_user]
        temp_count = self.count_arm[self.chosen_arm][self.current_user]
        self.count_arm[self.chosen_arm][self.current_user] += 1
        self.phase_count[self.chosen_arm] += 1
        self.check_thresh()
        self.average_rewards[self.chosen_arm][self.current_user] = (temp_mean * temp_count + reward * self.users_distribution[
            self.current_user]) / self.count_arm[self.chosen_arm][self.current_user]
        self.radius[self.chosen_arm][self.current_user] = self.ucb_radius()
        self.ucb_values[self.chosen_arm][self.current_user] = self.average_rewards[self.chosen_arm][
                                                                   self.current_user] + \
                                                              self.radius[self.chosen_arm][self.current_user]

    def choose_arm(self, user_context):
        """
        :input: the sampled user (integer in the range [0,num_users-1])
        :output: the chosen arm_to_init, content to show to the user (integer in the range [0,num_arms-1])
        """
        for i in self.deactived:
            self.inactive.add(i)
        self.active = [arm for arm in range(self.num_arms) if arm not in self.inactive]
        self.current_user = user_context
        if (self.init_ucb(user_context) != -1):
            arm_to_init = self.init_ucb(user_context)
            if arm_to_init != -1:
                self.chosen_arm = arm_to_init
                return arm_to_init

        # if self.time_to_be_greedy and (len(self.inactive) == 1):
        #     # print("why2?")
        #     self.keepArms = True

        if self.threshold_flag and self.keepArms:
            if self.flag:
                self.fix += self.count_end_phase
                self.flag = False
            if self.t > 10000:
                p = self.fix / self.t
                if p > 0.5 or (len(self.inactive) == 1):
                    self.keepArms = False
            num_false = 0
            false_indx = -1
            for i, j in enumerate(self.thresh_array):
                if j == False:
                    num_false += 1
                    false_indx = i
            if num_false == 1:
                self.chosen_arm = false_indx
                return self.chosen_arm
            elif num_false > 1:
                arms_array = []
                for i, k in enumerate(self.thresh_array):
                    if i not in self.inactive:
                        user_diff = self.arms_thresh[i] - self.phase_count[i]
                        arms_array.append((i, self.ucb_values[i][user_context], user_diff))
                    else:
                        continue
                arms_array.sort(key=lambda x: (x[2], x[1]), reverse=True)
                self.chosen_arm = arms_array[0][0]
                if arms_array[0][0] in self.inactive:
                    print("something wrong in algorithm")
                return arms_array[0][0]
        else:

            self.chosen_arm = np.argmax(list(self.ucb_values[:, user_context]))
            if self.chosen_arm in self.inactive:

                if len(self.active) > 0:
                    self.chosen_arm = random.choice(self.active)
                    return self.chosen_arm

            self.count_test_ucb += 1

            return self.chosen_arm

    def notify_outcome(self, reward):
        """
        :input: the sampled reward of the current round.
        """

        self.update_ucb(reward)

        if ((self.t + 1) % self.phase_len) == 0:
            self.count_test_ucb = 0
            self.check_deactive_arms()
            self.phase_count = [0] * self.num_arms
            self.thresh_array = [False] * self.num_arms
            self.count_end_phase = self.phase_len
            self.flag = True
        self.t = self.t + 1
        self.count_end_phase -= 1

    def get_id(self):
        # TODO: Make sure this function returns your ID, which is the name of this file!
        return "id_318155843_302342498"
