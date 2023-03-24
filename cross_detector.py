

def detect_red_line(obs):
    part = obs[479:480,:,0]
    for x in part:
        if x >= 180:
            return True
    return False