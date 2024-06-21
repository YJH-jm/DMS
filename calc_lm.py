
from scipy.spatial import distance

def ear(pts):
    A = distance.euclidean(pts[1], pts[7])
    B = distance.euclidean(pts[3], pts[5])
    C = distance.euclidean(pts[0], pts[4])

    return (A+B)/ (2.0*C)

def head_rate(pts):
    A = distance.euclidean(pts[51], pts[54])
    B = distance.euclidean(pts[54], pts[16])
    
    return A/B

def stir_rate(pts):
    A = distance.euclidean(pts[4],pts[54])
    B = distance.euclidean(pts[28],pts[54])

    return A/B
    
def face_metrics(pts):
    left_EAR = ear(pts[60:68])
    right_EAR = ear(pts[68:76])

    avg_EAR = (left_EAR + right_EAR) / 2
    HR = head_rate(pts)
    SR = stir_rate(pts)

    
    return avg_EAR, HR, SR