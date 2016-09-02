import numpy as np
from sklearn.neighbors import KDTree
import json
import multiHypoTracking_with_gurobi as mht


"""
This class contains two dictionary one according to a graphical model with detection variables and transition variables and factors and the other just with weights for setting the importance of different energy terms, the "model" dictionary is for creating the graph with detection and transition variables and factors with features, and the "weight" dictionary is for weighting the features 
"""
class GraphDict:
    #initialize object with skeleton dictionary 
    def __init__(self, author="Mathis", date="today", statesShareWeights=True, optimizerVerbose=True, optimizerEpGap=0.05 ): 
        self.model = {"author":author, "date": date }
        settings = {"statesShareWeights":statesShareWeights, "optimizerVerbose": optimizerVerbose, "optimizerEpGap":optimizerEpGap}
        self.model["settings"] = settings
        self.model["segmentationHypotheses"] = []
        self.model["linkingHypotheses"] = []
        self.model["exclusions"] = []
        self.weights = {"weights": []}        
   
    #add a detection variable to the graph with a unary factor graph
    #identity: add unique identity to detection variable
    #x, y, t, alpha: further properties used for reconstruction of solution
    #features: add list with energies
    #appearanceFeatures: list with energies
    #disappearanceFeatures: list with energies
    def add_seg_hypo(self, identity, features, x=0., y=0., alpha=0., t=0, appearanceFeatures=[], disappearanceFeatures=[]):
        new_seg_hypo = {"id": identity, "x":x, "y":y, "alpha":alpha, "t":t, "features": features}
        if not len(appearanceFeatures)==0:
            new_seg_hypo["appearanceFeatures"] = appearanceFeatures
        if not len(disappearanceFeatures)==0:
            new_seg_hypo["disappearanceFeatures"] = disappearanceFeatures
        self.model["segmentationHypotheses"].append(new_seg_hypo)

    #add a transition variable to the graph with a unary factor to the graph
    #src: identity of source identity as integer
    #dest: identity of destination identity as integer
    #features: list of energies
    def add_link_hypo(self, src, dest, features):
        new_link_hypo = {"src": src, "dest": dest, "features": features}
        self.model["linkingHypotheses"].append(new_link_hypo)

    #add exclusion set to graph
    def add_exclusions(self, list_excl):
        self.model["exclusions"].append(list_excl)

    #add weights to the weight dictionary, upto now no functionality,
    def add_weights(self, list_weights):
        self.weights["weights"] = list_weights


    #produce .json file for tracking
    def print_model_json(self, filename):
        f = open(filename +'.json', 'w') 
        json.dump(self.model, f)
        f.close()

    def print_weights_json(self, filename):
        f = open(filename +'.json', 'w') 
        json.dump(self.weights, f)
        f.close()

def print_json(filename, dictionary):
        f = open(filename +'.json', 'w') 
        json.dump(dictionary, f)
        f.close()


#initialization: set initial and endposition via detection reward 
def true_detection():   
    detection_reward = -99999.
    return [[-detection_reward],[detection_reward]]

def false_detection():
    no_detection_reward = -99999.
    return [[no_detection_reward],[-no_detection_reward]]
    


#here you can see how to use the class
if __name__ == '__main__':

    #mini examples with 3 timesteps and to many hypothesis
    mini_graph = GraphDict()

    #model need appearance features for nodes in first frame, otherwise no animals would be tracked
    neutral_appearance_feature = [[0.],[0.]]
    #model need disappearance features for nodes in last frame, otherwise no animals would be tracked
    neutral_disappearance_feature = [[0.],[0.]]

    #detection variables and factors t=0
    mini_graph.add_seg_hypo(1, true_detection(), appearanceFeatures=neutral_appearance_feature)
    mini_graph.add_seg_hypo(2, true_detection(), appearanceFeatures=neutral_appearance_feature)
    mini_graph.add_seg_hypo(3, true_detection(), appearanceFeatures=neutral_appearance_feature)
    mini_graph.add_seg_hypo(4, false_detection())#, appearanceFeatures=neutral_appearance_feature)
    #detection variables and factors t=1
    mini_graph.add_seg_hypo(5, [[12.],[0.2]])
    mini_graph.add_seg_hypo(6, [[13.],[0.3]])
    mini_graph.add_seg_hypo(7, [[12.],[0.3]])
    mini_graph.add_seg_hypo(8, [[1.],[12.]])    #wrong detection
    mini_graph.add_seg_hypo(9, [[2.],[8.]])     #wrong detection
    #detection variables and factors t=2
    mini_graph.add_seg_hypo(10, true_detection(), disappearanceFeatures=neutral_disappearance_feature)
    mini_graph.add_seg_hypo(11, true_detection(), disappearanceFeatures=neutral_disappearance_feature)
    mini_graph.add_seg_hypo(12, false_detection()) #, disappearanceFeatures=neutral_disappearance_feature)
    mini_graph.add_seg_hypo(13, true_detection(), disappearanceFeatures=neutral_disappearance_feature)
        
    #transition variables and factors t=0 -> t=1
    mini_graph.add_link_hypo(1,5,[[2.],[23.]]) #wrong transition
    mini_graph.add_link_hypo(1,6,[[23.],[2.]])
    mini_graph.add_link_hypo(2,5,[[23.],[2.]])
    mini_graph.add_link_hypo(2,7,[[2.],[11.]]) #wrong transition
    mini_graph.add_link_hypo(3,7,[[23.],[2.]])
    mini_graph.add_link_hypo(3,8,[[3.],[12.]]) #wrong transition
    mini_graph.add_link_hypo(4,5,[[5.],[5.]])
    mini_graph.add_link_hypo(4,9,[[5.],[5.]])
    
    #transition variables and factors t=1 -> t=2
    mini_graph.add_link_hypo(5,11,[[23.],[2.]])
    mini_graph.add_link_hypo(5,10,[[4.],[4.]])
    mini_graph.add_link_hypo(6,10,[[23.],[2.]])
    mini_graph.add_link_hypo(6,11,[[6.],[6.]])
    mini_graph.add_link_hypo(7,13,[[23.],[2.]])
    mini_graph.add_link_hypo(7,12,[[2.],[2.]])
    mini_graph.add_link_hypo(8,12,[[2.],[2.]])
    mini_graph.add_link_hypo(8,10,[[2.],[2.]])
    mini_graph.add_link_hypo(9,11,[[2.],[2.]])
    mini_graph.add_link_hypo(9,12,[[2.],[2.]])

    #add weights
    mini_graph.add_weights([1., 1., 1., 1.])

    #solve with multiHypothesesTracking
    mymodel = mini_graph.model
    myweights = mini_graph.weights
    result = mht.track(mymodel, myweights)
    print_json('result', result)

    """
    model = GraphDict()
    for identity in np.arange(10):
        features = []
        energy = np.random.random((1,))
        features.append([energy[0]])
        features.append([1-energy[0]])
        if energy[0]<1./3.:
            model.add_seg_hypo(identity,features, appearanceFeatures=[[0.], [1.]])     
        else:
            model.add_seg_hypo(identity,features)     
    for identity in np.arange(9):
        features = []
        energy = np.random.random((1,))
        features.append([energy[0]])
        features.append([1-energy[0]])
        model.add_link_hypo(identity, identity+1, features)
    model.add_exclusions([1,2,3])
    model.add_exclusions([4,5,9])
    #one transition weight, one detection weight, no division weight, one appearance weight
    model.add_weights([0.006, -0.004, 0.02])
    model_dict = model.model
    weight_dict = model.weights
    result = mht.track(model_dict, weight_dict)
    print_json('result', result)
    #model.print_model_json('model')
    #model.print_weights_json('weights')
    """
