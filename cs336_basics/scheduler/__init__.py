import math
def cosine_schedule(it, max_learning_rate, min_learning_rate, warmup_iters, final_cosine_iter):
    Tw = warmup_iters
    Tc = final_cosine_iter
    if it < Tw:
        return (it / Tw) * max_learning_rate
    elif it >= Tw and it <= Tc:
        delta = max_learning_rate - min_learning_rate
        frac = (it - Tw) / (Tc - Tw) #distance traversed, total distance -> asymptotes at 1 starting at 0 cosine up
        cosine_frac = math.cos(frac * math.pi)
        return (min_learning_rate + (1/2) * (1 + cosine_frac)*delta) # as we approach t -> Tw. frac -> 1. cos pi = -1. then we go to a_min
    else:
        return min_learning_rate
        