import math


def lr_cosine_schedule_with_warmup(
        t:int,
        init_alpha:float,
        final_alpha:float,
        T_w:int,
        T_c:int,
)-> float:
    if t < T_w:
        return (t/T_w)*init_alpha
    if  t>=T_c:
        return final_alpha
    progress = (t - T_w) / (T_c - T_w)  # 0 to 1
    cosine = 0.5 * (1 + math.cos(math.pi * progress))  # 1 to 0
    return final_alpha + cosine * (init_alpha - final_alpha)