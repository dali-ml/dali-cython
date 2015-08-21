import time

class Throttled(object):
    decorated_to_throttled = {}

    def __init__(self):
        self.last_time = None

    def maybe_run(self, min_time_since_last_run_s, f):
        now = time.time()
        if self.last_time is None or (now - self.last_time) > min_time_since_last_run_s:
            self.last_time = now
            return f()
        else:
            return None

def throttled(min_time_between_run_s):
    def decorator(f):
        def wrapper(*args, **kwargs):
            if f not in Throttled.decorated_to_throttled:
                Throttled.decorated_to_throttled[f] = Throttled()
            t = Throttled.decorated_to_throttled[f]
            def ok_this_is_getting_ridiculous():
                return f(*args, **kwargs)
            return t.maybe_run(min_time_between_run_s, ok_this_is_getting_ridiculous)
        return wrapper
    return decorator

__all__ = [
    "Throttled","throttled"
]
