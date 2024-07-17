'''This module calculates the fraud score based on the probability'''
def calc_fraud_score(prob:float) -> int:
    '''This function calculates the fraud score based on the probability'''
    prob = prob*100
    prob = round(prob, ndigits=2)
    return prob