from datetime import datetime


# Use te = TimeEstimator(n_total)
# te.start()
# te.estimate(n_finished) - at each estimated step
class TimeEstimator:
    def __init__(self, size, units='minutes', precision=2):
        self.size = size
        self.t_start = None
        self.t_now = None
        self.units = units
        self.precision = precision
        if units != 'minutes':
            raise NotImplementedError

    def start(self):
        self.t_start = datetime.now()

    # TODO
    def get_process_pow(self):
        return 1

    def estimate(self, n_processed_records):
        self.t_now = datetime.now()
        spent = round((self.t_now - self.t_start).seconds / 60, self.precision)
        app_left = round(
            ((int(self.size) - n_processed_records)
             * (self.t_now - self.t_start).seconds) / (n_processed_records * 60),
            self.precision)
        print('Spent %.2f Minutes; Left approximately %.2f Minutes'
              % (spent, app_left))

