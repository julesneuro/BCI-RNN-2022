import numpy as np
import numpy.matlib as ml
from scipy.optimize import minimize
import pickle as pkl
import rsatoolbox

# Model Inference

def calculate_likelihood_c1(Xv, Xa, M, pCommon, sigV, varV, sigA, varA, sigP, varP):
    #likelihood P(Xv, Xa|C =1)
    
    firstDenom = 2*np.pi*np.sqrt(varV*varA + varV*varP +varA*varP)
    firstTerm = 1/firstDenom 
    secondNum = (Xv - Xa)**2 * varP + (Xv -0)**2 * varA + (Xa - 0)**2* varV 
    secondDenom = (varV * varA) + (varV * varP) + (varA * varP)
    secondTerm = np.exp((-0.5*(secondNum/secondDenom)))
    likelihood_com = firstTerm*secondTerm
    return likelihood_com

def calculate_likelihood_c2(Xv,Xa, N, pCommon, sigV, varV, sigA, varA, sigP, varP):
    # likelihood P(Xv, Xa|C =2)
    
    firstTerm = 2*np.pi*np.sqrt((varV + varP)*(varA+varP))
    secondTerm1 = (Xv - 0)**2/(varV + varP)
    secondTerm2 = (Xa - 0)**2 / (varA + varP)
    secondTermFull = np.exp((-0.5*(secondTerm1+secondTerm2)) )
    likelihood_ind = secondTermFull/firstTerm
    return likelihood_ind

def calculate_posterior(Xv, Xa, N, pCommon, sigV, varV, sigA, varA, sigP, varP):
    # p(C = 1|Xv,Xa) posterior
    
    likelihood_common = calculate_likelihood_c1(Xv, Xa, N, pCommon, sigV, varV, sigA, varA, sigP, varP)
    likelihood_ind = calculate_likelihood_c2(Xv, Xa, N, pCommon, sigV, varV, sigA, varA, sigP, varP)
    post_common = likelihood_common * pCommon 
    post_indep = likelihood_ind * (1-pCommon)
    posterior = post_common/(post_common +post_indep)
    
    return posterior

def opt_position_conditionalised_C1(Xv, Xa, N, pCommon, sigV, varV, sigA, varA, sigP, varP):
    # Get optimal location given C = 1
    
    cues = Xv/varV + Xa/varA + ml.repmat(pCommon,N,1)/varP
    inverseVar = 1/varV + 1/varA + 1/varP
    sHatC1 = cues/inverseVar
    return sHatC1

def opt_position_conditionalised_C2(Xv, Xa, N, pCommon, sigV, varV, sigA, varA, sigP, varP):
    # Get optimal locationS given C = 2
    
    visualCue = Xv/varV +ml.repmat(pCommon,N,1)/varP
    visualInvVar = 1/varV + 1/ varP
    sHatVC2 = visualCue/visualInvVar
    audCue = Xa/varA + ml.repmat(pCommon,N,1)/varP
    audInvVar = 1/varA + 1/ varP
    sHatAC2 = audCue/audInvVar
    return sHatVC2, sHatAC2

def optimal_visual_location(Xv, Xa, N, pCommon, sigV, varV, sigA, varA, sigP, varP):
    # Use Model Averaging to compute final visual est
    
    posterior_1C = calculate_posterior(Xv, Xa, N, pCommon, sigV, varV, sigA, varA, sigP, varP)
    sHatVC1 = opt_position_conditionalised_C1(Xv, Xa, N, pCommon, sigV, varV, sigA, varA, sigP, varP)
    sHatVC2 = opt_position_conditionalised_C2(Xv, Xa, N, pCommon, sigV, varV, sigA, varA, sigP, varP)[0]
    sHatV = posterior_1C*sHatVC1 + (1-posterior_1C)*sHatVC2 #model averaging
    return sHatV

def optimal_aud_location(Xv, Xa, N, pCommon, sigV, varV, sigA, varA, sigP, varP):
    # Use Model Averaging to compute final auditory est
    
    posterior_1C = calculate_posterior(Xv, Xa, N, pCommon, sigV, varV, sigA, varA, sigP, varP)
    sHatAC1 = opt_position_conditionalised_C1(Xv, Xa, N, pCommon, sigV, varV, sigA, varA, sigP, varP)
    sHatAC2 = opt_position_conditionalised_C2(Xv, Xa, N, pCommon, sigV, varV, sigA, varA, sigP, varP)[1]
    sHatA = posterior_1C*sHatAC1 + (1-posterior_1C)*sHatAC2 #model averaging
    return sHatA

# Model Fitting Functions

def clip(i):
    
    """
        Avoid division by zero!
    """
    if i == 0:
        i += 0.0001
        
    return i

def get_classprobs(preds, locs = [20, 40, 60, 80, 100]):
    
    d_list = []
    discrete_list = []
    
    # Create Bins
    for i in preds:
        d = min(locs, key=lambda x:abs(x-i))
        d_list.append(d)
    for i in locs:
        count = d_list.count(i)
        discrete_list.append(count)
       
    # Return as list of probabilities
    probability_list = [i / sum(discrete_list) for i in discrete_list]
    probability_list = [clip(i) for i in probability_list]
    
    return probability_list

def loglik(n, p):
    
    """
        Returns MLE estimate for multinomial distribution.
        n = list of counts
        p = list of estimated class probabilities from 
            model sampling.
    """
    
    ll_list = []
    
    for i, j in zip(n, p):
        term = i * np.log(j)
        ll_list.append(term)
    
    return sum(ll_list)

def fit_model(N, data, params):
    
    """
        Args:
        N = number of times the inference model is sampled to
            parameterize the multinomial dist.
        data = np.array dim n_conditions * n_modalities * n_buttons
        params = parsed param guesses from the scipy.optimize function.
    """
    
    iteration_ll = []
    
    for cond in data:
        
        v_counts = data[cond, 0, :]
        a_counts = data[cond, 1, :]
        
        # Get response estimates
        Xv, Xa = (sigV * np.random.randn(N,1) + Sv), (sigA * np.random.randn(N,1) + Sa)
        
        # Get response dists
        sHatV = optimal_visual_location(Xv, Xa, N, pCommon, sigV, varV, sigA, varA, sigP, varP)
        sHatA = optimal_aud_location(Xv, Xa, N, pCommon, sigV, varV, sigA, varA, sigP, varP)
        
        # Get Predicted Bins
        v_probs, a_probs = get_classprobs(sHatV), get_classprobs(sHatA)
        
        # Calculate loglikelihood
        v_ll, a_ll = loglik(v_counts, v_probs), loglik(a_counts, a_probs)
        ll = v_all + a_ll
        
        iteration_ll.append(ll)
        
    ll = sum(iteration_ll)
    
    return ll, params

# OOP Implementation

class BCIModel:
    
    """
        Object to fit BCI model to behavioural data from either human observers
        or artificial neural networks.
        
        Args:
        
        data = n_conditions * 5d vector of counts
        
        Methods:
    """
    
    def __init__(self, data):
        
        self.data = data
        
    def get_simulated_rdm():
        
       pass

    def fit():
        
        ll, params, minimize(fit_model, )
        
        # should n_condition * counts array
        # fitted parameters
        
