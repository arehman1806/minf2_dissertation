
# Abstract class for sensors
class Sensor():

    def __init__(self, robot, sensor_name, sensor_params, robot_params):
        self.robot = robot
        self._sensor_name = sensor_name
        self._sensor_params = sensor_params
        self._robot_params = robot_params 

    def get_observation_space_size(self):
        return self.dim_obs_space
    
    def get_observation(self):
        raise NotImplementedError
    
    def _normalize_observation(self, observation, lower_bound, upper_bound):
        return (observation - lower_bound) / (upper_bound - lower_bound)
    