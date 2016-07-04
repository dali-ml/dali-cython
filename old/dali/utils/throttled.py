import time




class Throttled(object):
    def __init__(self, min_time_since_last_run_s=5):
        """Used for simple throttled execution.

        Here's a simple example:

            @Throttled(1)
            def lol(i):
                print('epoch %d' % (i,), flush=True)

            for i in range(100000000):
                lol(i)

        Above code will report the epoch every second.

        Here's another way:

            throttled = Throttled(1)


            for i in range(100000000000):
                if throttled.should_i_run():
                    print('epoch %d' % (i,), flush=True)
        """
        self.last_time = None
        self.min_time_since_last_run_s = min_time_since_last_run_s


    def should_i_run(self, min_time_since_last_run_s=None):
        min_time_since_last_run_s = min_time_since_last_run_s or self.min_time_since_last_run_s
        now = time.time()
        if self.last_time is None or (now - self.last_time) > min_time_since_last_run_s:
            self.last_time = now
            return True
        else:
            return False

    def maybe_run(self, f, min_time_since_last_run_s=None):
        if self.should_i_run(min_time_since_last_run_s):
            return f()
        else:
            return None

    def __call__(self, f):
        def wrapper(*args, **kwargs):
            return self.maybe_run(lambda: f(*args, **kwargs))
        return wrapper
