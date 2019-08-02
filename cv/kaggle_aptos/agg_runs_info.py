class Run_Agg:
    stored_points_log = []
    point_metrics = {}

    def __init__(self, goal, env_params={}):
        '''
        env_params contains all hyperparams that might be tuned or should be memorized as a part of experiment
        '''
        self.goal = goal
        self.env_params = env_params

    def save_point(self, learn, point_name, point_metrics):
        learn.save(point_name)
        if point_name not in self.stored_points_log:
            self.stored_points_log.append(point_name)
        self.point_metrics[point_name] = point_metrics

    def get_summary(self):
        par = []
        for k, v in self.env_params.items():
            par.append(f' - {k}: {v}')
        par = '\n'.join(par)

        return f'[Experiment goal]: {self.goal}\n' + \
               '[Metrics by runs] \n ' + "\n".join(self.point_metrics.values()) + '\n' + \
               f'[Essential hyperparams and conf] \n{par}' + \
               '\n[Stored poits] : ' + ", ".join(self.stored_points_log)
    
