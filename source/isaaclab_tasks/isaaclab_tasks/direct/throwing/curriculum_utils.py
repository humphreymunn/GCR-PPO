import torch
import os

class Schedule():
    
    def __init__(self, param_names, param_ranges, converg_crit, cl_levels, num_envs, device, random_sample=True):
        assert len(param_ranges), len(param_ranges[0]) == (cl_levels, len(param_names))
        self.param_names = param_names # List[String] of variable names to modify
        self.param_ranges = param_ranges # np.arr[np.arr]
        self.converg_crit = converg_crit # function taking in an array of all previous reward values and outputting True if move to next CL level else False
        self.cl_levels = cl_levels # number of cl levels or stages
        self.current_level = 0
        self.last_update_iter = 0
        self.num_envs = num_envs
        self.device = device
        self.random_sample = random_sample # this means if a range is specified, it will resampled at every call to update rather than only when the curriculum level is incremented
    
    def set_parameters(self, parent_obj, use_print=True):
        self.values = torch.zeros(len(self.param_ranges[0])) #torch.rand(self.num_envs,len(self.param_names)-1,device=self.device)
        for i,n in enumerate(self.param_names):
            # if range is a single value, set it instead of uniformly sample from the range
            if len(self.param_ranges[self.current_level][i]) == 1:
                setattr(parent_obj, n, self.param_ranges[self.current_level][i][0])
                if use_print: print(f"Set {n} in the curriculum to {self.param_ranges[self.current_level][i][0]}.")
            else:
                value = torch.rand(self.num_envs, 1, device=self.device) * (self.param_ranges[self.current_level][i][1]-self.param_ranges[self.current_level][i][0]) + self.param_ranges[self.current_level][i][0]
                setattr(parent_obj, n, value)
                if use_print: print(f"Set {n} in the curriculum to [{self.param_ranges[self.current_level][i][0]}, {self.param_ranges[self.current_level][i][1]}].")


    def update(self, parent_obj, past_rewards, past_avg_rewards):
        '''# update stability coefficient
        if len(past_avg_rewards[0]) > 200:# and past_avg_rewards[1][-1] : # only start it after 200 iters
            original_scale = parent_obj.new_reward_scales[1]
            difference = parent_obj.desired_stability - past_avg_rewards[0][-1]
            #if difference > 0:
            #    difference /= 2 # slow down the increase
            stability_coef = min(0.1, max(parent_obj.reward_scales["stability"] + parent_obj.stability_adjustment_rate * difference, -1e-8))
            parent_obj.new_reward_scales[1] = float(stability_coef)
            rew_diff = past_avg_rewards[0][-1]*original_scale - stability_coef*past_avg_rewards[0][-1]
            parent_obj.new_reward_scales[0] += rew_diff / (-past_avg_rewards[1][-1]/parent_obj.MAX_DIST_PARAM + 1)
            #parent_obj.new_reward_scales[0] *= 1 + ((1-float(stability_coef))*past_avg_rewards[0][-1])/(past_avg_rewards[1][-1])    #(1-(original_scale-float(stability_coef)))*past_avg_rewards[0][-1] '''
        
        
            
        if len(past_avg_rewards[0]) >= 4000 and parent_obj.cfg.env.task_throw == "general":
            if past_avg_rewards[0][-1] <= 1. and parent_obj.noise_obs < 1 and self.last_update_iter < len(past_avg_rewards[0]):
                new_noise_obs = min(1., parent_obj.noise_obs + 1/100)
                setattr(parent_obj, "noise_obs", new_noise_obs)
                print(f"Set noise_obs in the curriculum to {new_noise_obs}.")
                self.last_update_iter = len(past_avg_rewards[0])

        # check convergence
        if len(past_avg_rewards[0])>0 and self.converg_crit([x[self.last_update_iter:] for x in past_avg_rewards]) and self.current_level < self.cl_levels - 1:
            self.current_level += 1
            self.set_parameters(parent_obj)
            self.last_update_iter = len(past_avg_rewards[0])
            
        elif self.random_sample:
            self.set_parameters(parent_obj, use_print=False) # resample parameters that are within a range.

class NoCurriculum():
    def __init__(self, param_names, param_ranges, num_envs, device):
        if os.path.exists("random_log.txt"):
            os.remove("random_log.txt")
        self.param_names = param_names
        self.param_ranges = param_ranges # assuming max is max difficulty
        self.device = device
        self.num_envs = num_envs
        self.throwing_values = torch.rand(self.num_envs,len(param_names),device=self.device)
        for i in range(len(param_names)):
            self.throwing_values[:,i] = self.throwing_values[:,i] * (param_ranges[i][1]-param_ranges[i][0]) + param_ranges[i][0]
        self.last_update_iter = 0

    def set_parameters(self, parent_obj):
        for i,n in enumerate(self.param_names):
            setattr(parent_obj, n, self.throwing_values[:,i].unsqueeze(-1))
            #print(f"Set {n} in the curriculum.")

    def update(self, parent_obj, past_rewards, past_avg_rewards):
        #if len(past_avg_rewards[self.last_update_iter:]) > 0:
        self.throwing_values = torch.rand(self.num_envs,len(self.param_names),device=self.device)
        for i in range(len(self.param_names)):
            self.throwing_values[:,i] = self.throwing_values[:,i] * (self.param_ranges[i][1]-self.param_ranges[i][0]) + self.param_ranges[i][0]
        self.set_parameters(parent_obj)
        self.last_update_iter = len(past_avg_rewards)