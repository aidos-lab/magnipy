import torch
import numpy as np
from scipy.spatial import distance_matrix
import networkx as nx

def dominatingSet(X,epsilon=0.1):
    "Dominating dataset of X with a given labels y and representativeness factor epsilon."
    ady = distance_matrix(X,X)
    g = nx.from_numpy_array(ady<epsilon)
    dom = nx.dominating_set(g)
    return np.array(list(dom))

#---------------- ITERATIVE ALGORITHMS ----------------------------------


def add_and_normalize_asvec(S,h):
    W = torch.eye(S.shape[0]).to(S.device)
    for iterations in range(h): 
        W = S @ W
        b = torch.sum(W, dim=1)
        W = W.diag()/b
        W = W.diag()
    return W

def add_and_normalize(S,h):
    W = torch.eye(S.shape[0]).to(S.device)
    for iterations in range(h): 
        V=torch.diagonal(W).diag()
        W = S @ V
        b = torch.sum(W, dim=1)
        c = (1/b).diag()
        W = c @ W
    return W

def add_and_normalize_points(points,h):
    S = similarity_matrix(points)
    return add_and_normalize(S, h)


def add_and_normalize_points_asvec(points,h):
    S = similarity_matrix(points)
    return add_and_normalize_asvec(S, h)

class Model(torch.nn.Module):
    def __init__(self, S, device):
        super(Model, self).__init__()
        self.device = device
        self.S = S.to(device)
        self.weights = torch.nn.Parameter(torch.ones(S.shape[0]).to(device))

    def forward(self): 
        V=self.weights.diag()
        W = self.S @ V
        return torch.sum(W, dim=1)

def magnitude_by_SGD(S, h, lr=0.01, device='cpu'): 
    model = Model(S, device)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    target = torch.ones(S.shape[0]).to(device)
    loss_fn = torch.nn.MSELoss()
    for i in range(h): 
        optimizer.zero_grad()
        output = model.forward()
        loss_val = loss_fn(output, target)
        loss_val.backward()
        optimizer.step()
    return model.weights

def magnitude_by_SGD_points(points, h, lr=0.01, device='cpu'):
    S = similarity_matrix(points)
    return magnitude_by_SGD(S, h, lr, device)

def magnitude_by_batch_SGD(S, num_epochs=100, batch_size=1, lr=0.01, device='cpu'): 
    model = Model(S, device)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    target = torch.ones(S.shape[0]).to(device)
    loss_fn = torch.nn.MSELoss()

    data_iter = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(model.weights, target), 
                                            batch_size, shuffle=True)
    
    for epoch in range(num_epochs):
        for batch_id, (X, y) in enumerate(data_iter):

            optimizer.zero_grad()
            output = model.forward()
            loss_val = loss_fn(output, target)
            loss_val.backward()
            optimizer.step()
    return model.weights 


def magnitude_by_batch_SGD_points(points, num_epochs=100, batch_size=1, lr=0.01, device='cpu'):
    S = similarity_matrix(points)
    return magnitude_by_batch_SGD(S, num_epochs, batch_size, lr, device)

def similarity_matrix(points):
    # Compute the pairwise distance matrix
    dist = torch.cdist(points, points, p=2)
    # Compute the similarity matrix
    sim = torch.exp(-dist)
    return sim


# -------------- MATRIX INVERSION -------------------------


def magnitude(S, device):
    S = S.to(device)
    inverse = torch.inverse(S)
    return torch.sum(inverse)

def magnitudeof_points(points, device):
    S = similarity_matrix(points).to(device)
    return magnitude(S, device)

# Numpy inversion (where GPU is not available)

def compute_magnitude_no_gpu(W, t):
    dist_mtx = distance_matrix(W, W)
    inv_fn = np.linalg.pinv
    Z = inv_fn(np.exp(-t * dist_mtx))
    magnitude = Z.sum()
    return magnitude

# -------------SUBSET SELECTION ----------------------------
# 1. Discrete Centers

def discrete_center_hierarchy(S):
    # first pass: create the centre hierarchy
    n = len(S)
    initial_centres = S
    current_centres = initial_centres
    # current_centres = []
    centres_indices = list(range(len(S)))

    levels_indices = []

    # in level 0, we have all the points, hence:
    levels_indices.append(centres_indices)

    for i in range(0, n):
        # original radius: 2**i; new reduced radius: (2**i)/10
        radius = (2) ** i / 10
        # radius = 2.718**i
        # this computes the indices of the centres in current_centres; the maximum index is len(current_centres)
        centres = dominatingSet(current_centres, radius)
        # dominating sets is indices but do not correspond to the original indices
        actual_indices = []
        for centre in centres:
            index_of_centre_in_previous_level = centres_indices[centre]
            actual_indices.append(index_of_centre_in_previous_level)
        levels_indices.append(actual_indices)
        dominating_sets = []
        for point in centres:
            dominating_sets.append(current_centres[point])

        device = 'cuda'
        # speed up the magnitude computation using tensors
        magnitude_of_dominating_set_tensor = magnitudeof_points(torch.from_numpy(np.array(dominating_sets)), device)
        magnitude_of_dominating_set = magnitude_of_dominating_set_tensor.item()
        current_centres = dominating_sets
        # centres has the indexing information
        centres_indices = actual_indices
        if len(centres) == 1:
            break

    hierarchy_ordered = []
    for i in reversed(range(len(levels_indices))):
        for point in levels_indices[i]:
            if point not in hierarchy_ordered:
                hierarchy_ordered.append(point)

    return hierarchy_ordered


# 2. Greedy Mazimization

def greedy_maximization(S, tolerance_parameter=0.01, no_gpu=True):
    best_magnitude_array = []
    initial_value_magnitude = 0
    best_magnitude_array.append(initial_value_magnitude)

    # take X to be a copy of S
    X = S

    set_of_points = []
    set_of_points.append(X[0])

    # first run of the algorithm to start with 2 values

    best_magnitude = 0
    X_copy = X.copy()

    best_i = 0

    for i in range(1, len(X_copy)):
        # repeat the re-assignment at every iteration
        new_set = []
        new_set.append(X[0])
        new_set.append(X_copy[i])
        # print('the new set is,', new_set)
        # incremental_magnitude = compute_magnitude(np.array(new_set), 1)
        # print('the incremental magnitude is,', incremental_magnitude)

        # speed up the magnitude computation using tensors
        device = 'cuda'
        incremental_magnitude_tensor = magnitudeof_points(torch.from_numpy(np.array(new_set)), device)
        incremental_magnitude = incremental_magnitude_tensor.item()

        if best_magnitude == 0:
            best_magnitude = incremental_magnitude
        if incremental_magnitude > best_magnitude:
            best_magnitude = incremental_magnitude
            best_i = i
    # select the point which increases magnitude the most
    set_of_points.append(X_copy[best_i])

    X_copy = list(X_copy)
    X_copy.pop(best_i)
    X_copy = np.array(X_copy)

    best_magnitude_array.append(best_magnitude)

    tolerance = tolerance_parameter
    j = 1

    # the next line assumes that the series of magnitude is increasing
    while True:
        if abs(best_magnitude_array[j] - best_magnitude_array[j - 1]) <= tolerance:
            break
        best_magnitude = 0
        best_i = 0
        i = 0
        for i in range(len(X_copy)):
            # repeat the re-assignment at every iteration
            new_set = set_of_points.copy()
            new_set.append(X_copy[i])

            if no_gpu == True:
                # version without tensors:
                incremental_magnitude = compute_magnitude_no_gpu(np.array(new_set), 1)
            else:
                # speed up the magnitude computation using tensors
                device = 'cuda'
                incremental_magnitude_tensor = magnitudeof_points(torch.from_numpy(np.array(new_set)), device)
                incremental_magnitude = incremental_magnitude_tensor.item()

            if best_magnitude == 0:
                best_magnitude = incremental_magnitude
            if incremental_magnitude > best_magnitude:
                best_magnitude = incremental_magnitude
                best_i = i

        maximising_element = X_copy[best_i]

        X_copy = list(X_copy)
        X_copy.pop(best_i)
        X_copy = np.array(X_copy)
        # select the point which increases magnitude the most
        # do not add the element if it is already in the list

        is_in = False

        for point in set_of_points:
            if (point == maximising_element).all():
                is_in = True

        if is_in == False:
            set_of_points.append(maximising_element)
            best_magnitude_array.append(best_magnitude)
        if len(set_of_points) >= len(X):
            print('We should break the while loop, we have reached the cardinality of the set')
            break

        j += 1
        converging_number_of_points = len(set_of_points)
    return best_magnitude_array







