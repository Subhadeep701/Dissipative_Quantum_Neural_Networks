# math related packages
import scipy as sc
import qutip as qt
import numpy as np
from qutip import Qobj
import pandas as pd
import copy
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, KFold
from scipy import stats
# further packages
from time import time
from sklearn.model_selection import train_test_split

from random import sample
# ket states
qubit0 = qt.basis(2,0)
qubit1 = qt.basis(2,1)
# density matrices
qubit0mat = qubit0 * qubit0.dag() # it is also possible to use ket2dm(qubit0)
qubit1mat = qubit1 * qubit1.dag() # it is also possible to use ket2dm(qubit1)

def partialTraceKeep(obj, keep): # generalisation of ptrace(), partial trace via "to-keep" list
    # return partial trace:
    res = obj;
    if len(keep) != len(obj.dims[0]):
        res = obj.ptrace(keep);
    return res;

def partialTraceRem(obj, rem): # partial trace via "to-remove" list
    # prepare keep list
    rem.sort(reverse=True)
    keep = list(range(len(obj.dims[0])))
    for x in rem:
        keep.pop(x)
    res = obj;
    # return partial trace:
    if len(keep) != len(obj.dims[0]):
        res = obj.ptrace(keep);
    return res;
def swappedOp(obj, i, j):
    if i==j: return obj
    numberOfQubits = len(obj.dims[0])
    permute = list(range(numberOfQubits))
    permute[i], permute[j] = permute[j], permute[i]
    return obj.permute(permute)
def tensoredId(N):
    #Make Identity matrix
    res = qt.qeye(2**N)
    #Make dims list
    dims = [2 for i in range(N)]
    dims = [dims.copy(), dims.copy()]
    res.dims = dims
    #Return
    return res

def tensoredQubit0(N):
    #Make Qubit matrix
    res = qt.fock(2**N).proj() #for some reason ran faster than fock_dm(2**N) in tests
    #Make dims list
    dims = [2 for i in range(N)]
    dims = [dims.copy(), dims.copy()]
    res.dims = dims
    #Return
    return res

def unitariesCopy(unitaries): # deep copyof a list of unitaries
    newUnitaries = []
    for layer in unitaries:
        newLayer = []
        for unitary in layer:
            newLayer.append(unitary.copy())
        newUnitaries.append(newLayer)
    return newUnitaries


########Change this function for chaning gaussian######



def randomTrainingData(unitary, N): # generating training data based on a unitary
    numQubits = len(unitary.dims[0])
    trainingData=[]
    #Create training data pairs
    for i in range(N):
        t =randomMixedQubitDensityMatrix(numQubits)
        ut = unitary*t*unitary.dag()
        trainingData.append([t,ut])
    #Return
    return trainingData #####change should be here


def randomNetwork(qnnArch, numTrainingPairs):
    assert qnnArch[0]==qnnArch[-1], "Not a valid QNN-Architecture."

    #Create the targeted network unitary and corresponding training data
    networkUnitary = randomQubitUnitary(qnnArch[-1])
    networkTrainingData = randomTrainingData(networkUnitary, numTrainingPairs)

    #Create the initial random perceptron unitaries for the network
    networkUnitaries = [[]]
    for l in range(1, len(qnnArch)):
        numInputQubits = qnnArch[l-1]
        numOutputQubits = qnnArch[l]

        networkUnitaries.append([])
        for j in range(numOutputQubits):
            unitary = randomQubitUnitary(numInputQubits+1)
            if numOutputQubits-1 != 0:
                unitary = qt.tensor(randomQubitUnitary(numInputQubits+1), tensoredId(numOutputQubits-1))
                unitary = swappedOp(unitary, numInputQubits, numInputQubits + j)
            networkUnitaries[l].append(unitary)

    #Return
    return (qnnArch, networkUnitaries, networkTrainingData, networkUnitary)

######Fidelity cost function######
def costFunction(trainingData, outputStates):
    costSum = 0
    for i in range(len(trainingData)):
        costSum += qt.fidelity(trainingData[i][1],outputStates[i])
    #print(np.real(costSum)/len(trainingData))
    return np.real(costSum)/len(trainingData)


##entropy cost function ##
def entropy_costfunction(trainingData, outputStates):
    costSum = 0
    for i in range(len(trainingData)):
        # Target state (pure state vector |psi>)
        target_state = trainingData[i][1]

        # Predicted state (density matrix)
        predicted_output = outputStates[i]

        # # Eigen decomposition of predicted_output to compute log(sigma)
        # eig_vals, eig_vecs = predicted_output.eigenstates()

        # # Construct log(sigma)
        # log_rho = sum(val * vec * vec.dag() * np.log(val) if val > 0 else 0
        #                 for val, vec in zip(eig_vals, eig_vecs))

        # # Relative entropy: - <psi| log(sigma) |psi>
        # #relative_entropy = - (target_state.dag() * log_rho * target_state)
        # relative_entropy=-((target_state*log_rho)+log_rho*target_state).tr()

        # Add only the real part to costSum
        costSum += qt.entropy_relative(target_state, predicted_output)
    #print(f"costSum:{costSum}")
    costSum=costSum/len(trainingData)
    return costSum



    ###feedforward####
def makeLayerChannel(qnnArch, unitaries, l, inputState):
    numInputQubits = qnnArch[l-1]
    numOutputQubits = qnnArch[l]

    #Tensor input state
    state = qt.tensor(inputState, tensoredQubit0(numOutputQubits))

    #Calculate layer unitary
    layerUni = unitaries[l][0].copy()
    for i in range(1, numOutputQubits):
        layerUni = unitaries[l][i] * layerUni

    #Multiply and tensor out input state
    return partialTraceRem(layerUni * state * layerUni.dag(), list(range(numInputQubits)))
def makeAdjointLayerChannel(qnnArch, unitaries, l, outputState):
    numInputQubits = qnnArch[l-1]
    numOutputQubits = qnnArch[l]

    #Prepare needed states
    inputId = tensoredId(numInputQubits)
    state1 = qt.tensor(inputId, tensoredQubit0(numOutputQubits))
    state2 = qt.tensor(inputId, outputState)

    #Calculate layer unitary
    layerUni = unitaries[l][0].copy()
    for i in range(1, numOutputQubits):
        layerUni = unitaries[l][i] * layerUni

    #Multiply and tensor out output state
    return partialTraceKeep(state1 * layerUni.dag() * state2 * layerUni, list(range(numInputQubits)) )
def feedforward(qnnArch, unitaries, trainingData):
    storedStates = []
    for x in range(len(trainingData)):
        currentState = trainingData[x][0]
        layerwiseList = [currentState]
        for l in range(1, len(qnnArch)):
            currentState = makeLayerChannel(qnnArch, unitaries, l, currentState)
            layerwiseList.append(currentState)
        storedStates.append(layerwiseList)
    return storedStates


    ##### update matrix according to paper ####
def makeUpdateMatrix(qnnArch, unitaries, trainingData, storedStates, lda, ep, l, j):
    numInputQubits = qnnArch[l-1]

    #Calculate the sum:
    summ = 0
    for x in range(len(trainingData)):
        #Calculate the commutator
        firstPart = updateMatrixFirstPart(qnnArch, unitaries, storedStates, l, j, x)
        secondPart = updateMatrixSecondPart(qnnArch, unitaries, trainingData, l, j, x)
        mat = qt.commutator(firstPart, secondPart)

        #Trace out the rest
        keep = list(range(numInputQubits))
        keep.append(numInputQubits + j)
        mat = partialTraceKeep(mat, keep)

        #Add to sum
        summ = summ + mat

    #Calculate the update matrix from the sum
    summ = (-ep * (2**numInputQubits)/(lda*len(trainingData))) * summ
    return summ.expm()


def updateMatrixFirstPart(qnnArch, unitaries, storedStates, l, j, x):
    numInputQubits = qnnArch[l-1]
    numOutputQubits = qnnArch[l]

    #Tensor input state
    state = qt.tensor(storedStates[x][l-1], tensoredQubit0(numOutputQubits))

    #Calculate needed product unitary
    productUni = unitaries[l][0]
    for i in range(1, j+1):
        productUni = unitaries[l][i] * productUni

    #Multiply
    return productUni * state * productUni.dag()


def updateMatrixSecondPart(qnnArch, unitaries, trainingData, l, j, x):
    numInputQubits = qnnArch[l-1]
    numOutputQubits = qnnArch[l]

    #Calculate sigma state
    state = trainingData[x][1]  ####### density matrix directly
    for i in range(len(qnnArch)-1,l,-1):
        state = makeAdjointLayerChannel(qnnArch, unitaries, i, state)
    #Tensor sigma state
    state = qt.tensor(tensoredId(numInputQubits), state)

    #Calculate needed product unitary
    productUni = tensoredId(numInputQubits + numOutputQubits)
    for i in range(j+1, numOutputQubits):
        productUni = unitaries[l][i] * productUni

    #Multiply
    return productUni.dag() * state * productUni


def makeUpdateMatrixTensored(qnnArch, unitaries, lda, ep, trainingData, storedStates, l, j):
    numInputQubits = qnnArch[l-1]
    numOutputQubits = qnnArch[l]

    res = makeUpdateMatrix(qnnArch, unitaries, lda, ep, trainingData, storedStates, l, j)
    if numOutputQubits-1 != 0:
        res = qt.tensor(res, tensoredId(numOutputQubits-1))
    return swappedOp(res, numInputQubits, numInputQubits + j)

def randomQubitUnitary(numQubits): # alternatively, use functions rand_unitary and rand_unitary_haar
    dim = 2**numQubits
    #Make unitary matrix
    res = np.random.normal(size=(dim,dim)) + 1j * np.random.normal(size=(dim,dim))
    res = sc.linalg.orth(res)
    res = qt.Qobj(res)
    #Make dims list
    dims = [2 for i in range(numQubits)]
    dims = [dims.copy(), dims.copy()]
    res.dims = dims
    #Return
    return res
def randomQubitState(numQubits): # alternatively, use functions rand_ket and rand_ket_haar
    dim = 2**numQubits
    #Make normalized state
    res = np.random.uniform(low=0, high=1,size=(dim,1)) + 1j * np.random.uniform(low=0, high=1,size=(dim,1))
    res = (1/sc.linalg.norm(res)) * res
    res = qt.Qobj(res)
    #Make dims list
    dims1 = [2 for i in range(numQubits)]
    dims2 = [1 for i in range(numQubits)]
    dims = [dims1, dims2]
    res.dims = dims
    #Return
    return res
def randomMixedQubitDensityMatrix(numQubits, numStates=3):
    """
    Generate a random mixed density matrix for a given number of qubits.

    Parameters:
    - numQubits: int, the number of qubits.
    - numStates: int, the number of pure states to mix (default: 3).

    Returns:
    - density_matrix: qutip.Qobj, a mixed density matrix.
    """
    dim = 2**numQubits
    density_matrix = qt.Qobj(np.zeros((dim, dim), dtype=complex))

    # Generate a set of random pure states
    for _ in range(numStates):
        state = np.random.uniform(low=0, high=1, size=(dim, 1)) + 1j * np.random.uniform(low=0, high=1, size=(dim, 1))
        state = (1 / np.linalg.norm(state)) * state  # Normalize the state
        state = qt.Qobj(state)

        # Compute the density matrix for this pure state
        pure_density = state * state.dag()

        # Add the weighted pure state to the mixed density matrix
        weight = np.random.uniform(0, 1)  # Random weight
        density_matrix += weight * pure_density

    # Normalize the mixed density matrix
    density_matrix = density_matrix / density_matrix.tr()

    # Set dimensions
    dims1 = [2 for _ in range(numQubits)]
    density_matrix.dims = [dims1, dims1]
    return density_matrix
def qnnTraining(qnnArch, initialUnitaries, trainingData, lda, ep, trainingRounds, alert=0):

    ### FEEDFORWARD
    #Feedforward for given unitaries
    s = 0
    currentUnitaries = initialUnitaries
    storedStates = feedforward(qnnArch, currentUnitaries, trainingData)

    #Cost calculation for given unitaries
    outputStates = []
    for k in range(len(storedStates)):
        outputStates.append(storedStates[k][-1])
    plotlist = [[s], [costFunction(trainingData, outputStates)],[entropy_costfunction(trainingData, outputStates)]]

    #Optional
    runtime = time()

    #Training of the Quantum Neural Network
    for k in range(trainingRounds):
        if alert>0 and k%alert==0: print("In training round "+str(k))

        ### UPDATING
        newUnitaries = unitariesCopy(currentUnitaries)

        #Loop over layers:
        for l in range(1, len(qnnArch)):
            numInputQubits = qnnArch[l-1]
            numOutputQubits = qnnArch[l]

            #Loop over perceptrons
            for j in range(numOutputQubits):
                newUnitaries[l][j] = (makeUpdateMatrixTensored(qnnArch,currentUnitaries,trainingData,storedStates,lda,ep,l,j)* currentUnitaries[l][j])

        ### FEEDFORWARD
        #Feedforward for given unitaries
        s = s + ep
        currentUnitaries = newUnitaries
        storedStates = feedforward(qnnArch, currentUnitaries, trainingData)

        #Cost calculation for given unitaries
        outputStates = []
        for m in range(len(storedStates)):
            outputStates.append(storedStates[m][-1])
        plotlist[0].append(s)
        plotlist[1].append(costFunction(trainingData, outputStates))
        plotlist[2].append(entropy_costfunction(trainingData, outputStates))

    #Optional
    runtime = time() - runtime
    #print("Trained "+str(trainingRounds)+" rounds for a "+str(qnnArch)+" network and "+str(len(trainingData))+" training pairs in "+str(round(runtime, 2))+" seconds")

    #Return1
    return [plotlist, currentUnitaries]
### our model ###
def makeUpdateMatrix_en(qnnArch, unitaries, trainingData, storedStates,outputstates, lda, ep, l, j):
    numInputQubits = qnnArch[l-1]

    #Calculate the sum:
    summ = 0
    for x in range(len(trainingData)):
        #Calculate the commutator
        firstPart = updateMatrixFirstPart_en(qnnArch, unitaries, storedStates, l, j, x)
        secondPart = updateMatrixSecondPart_en(qnnArch, unitaries, trainingData, outputstates, l, j, x)
        mat = qt.commutator(firstPart, secondPart)

        #Trace out the rest
        keep = list(range(numInputQubits))
        keep.append(numInputQubits + j)
        mat = partialTraceKeep(mat, keep)

        #Add to sum
        summ = summ + mat

    #Calculate the update matrix from the sum
    summ = (-ep * (2**numInputQubits)/(lda*len(trainingData))) * summ
    return summ.expm()


def updateMatrixFirstPart_en(qnnArch, unitaries, storedStates, l, j, x):
    numInputQubits = qnnArch[l-1]
    numOutputQubits = qnnArch[l]

    #Tensor input state
    state = qt.tensor(storedStates[x][l-1], tensoredQubit0(numOutputQubits))

    #Calculate needed product unitary
    productUni = unitaries[l][0]
    for i in range(1, j+1):
        productUni = unitaries[l][i] * productUni

    #Multiply
    return productUni * state * productUni.dag()



def updateMatrixSecondPart_en(qnnArch, unitaries, trainingData,outputstates, l, j, x):
    numInputQubits = qnnArch[l-1]
    numOutputQubits = qnnArch[l]

    #Calculate sigma state
    state = trainingData[x][1]*outputstates[x].inv()
    #print(f"trainingData:{trainingData[x][1]}")


    state=(state+state.dag())/2
    #state=state/state.tr()

    #print(f"State_before{state}")
    #breakpoint()

    state=wEVD(state)


    #print(f"stateEVD:{state} done")
    for i in range(len(qnnArch)-1,l,-1):
        state = makeAdjointLayerChannel(qnnArch, unitaries, i, state)
    #Tensor sigma state
    state = qt.tensor(tensoredId(numInputQubits), state)

    #Calculate needed product unitary
    productUni = tensoredId(numInputQubits + numOutputQubits)
    for i in range(j+1, numOutputQubits):
        productUni = unitaries[l][i] * productUni

    #Multiply
    return productUni.dag() * state * productUni


def makeUpdateMatrixTensored_en(qnnArch, unitaries, lda, ep, trainingData, storedStates,outputstates, l, j):
    numInputQubits = qnnArch[l-1]
    numOutputQubits = qnnArch[l]

    res = makeUpdateMatrix_en(qnnArch, unitaries, lda, ep, trainingData, storedStates,outputstates, l, j)
    if numOutputQubits-1 != 0:
        res = qt.tensor(res, tensoredId(numOutputQubits-1))
    return swappedOp(res, numInputQubits, numInputQubits + j)


def wEVD(sta):
    # Extract the 2D matrix (data) from the Qobj
    matrix = sta.full()

    # Perform eigenvalue decomposition on the matrix (numpy-based)
    ev, evc = np.linalg.eigh(matrix)  # eigh is used for Hermitian matrices

    # Separate negative and positive eigenvalues
    negative_sum = sum(ev[ev < 0])

    # Set negative eigenvalues to zero
    ev[ev < 0] = 0

    # Sort positive eigenvalues in ascending order
    ev_pos_sorted_indices = sorted(range(len(ev)), key=lambda i: ev[i])

    # Subtract the negative sum from the smallest positive eigenvalues
    for idx in ev_pos_sorted_indices:
        if ev[idx] > 0 and negative_sum < 0:
            subtract_amount = min(ev[idx], -negative_sum)
            ev[idx] -= subtract_amount
            negative_sum += subtract_amount  # Decrease the negative sum
            if negative_sum == 0:
                break
    ev/=np.sum(ev)
    # Reconstruct the new matrix
    evc_matrix = np.array(evc)  # Convert eigenvectors into a numpy array
    diag_matrix = np.diag(ev)   # Create diagonal matrix of modified eigenvalues
    new_matrix = evc_matrix @ diag_matrix @ evc_matrix.T.conj()  # Reconstruct new matrix

    # Create a new Qobj with the modified data but same properties as the original
    new_sta = Qobj(new_matrix, dims=sta.dims)

    # Ensure the new state has the same Hermiticity property
    new_sta.isherm = sta.isherm

    # Print the properties to verify

    return new_sta
def qnnTraining_en(qnnArch, initialUnitaries, trainingData, lda, ep, trainingRounds, alert=0):

    ### FEEDFORWARD
    #Feedforward for given unitaries
    s = 0
    currentUnitaries = initialUnitaries
    storedStates = feedforward(qnnArch, currentUnitaries, trainingData)

    #Cost calculation for given unitaries
    outputStates = []
    for k in range(len(storedStates)):
        outputStates.append(storedStates[k][-1])
    plotlist = [[s], [costFunction(trainingData, outputStates)],[entropy_costfunction(trainingData, outputStates)]]
    #Optional
    runtime = time()

    #Training of the Quantum Neural Network
    for k in range(trainingRounds):
        if alert>0 and k%alert==0: print("In training round "+str(k))

        ### UPDATING
        newUnitaries = unitariesCopy(currentUnitaries)

        #Loop over layers:
        for l in range(1, len(qnnArch)):
            numInputQubits = qnnArch[l-1]
            numOutputQubits = qnnArch[l]

            #Loop over perceptrons
            for j in range(numOutputQubits):
                newUnitaries[l][j] = (makeUpdateMatrixTensored_en(qnnArch,currentUnitaries,trainingData,storedStates,outputStates,lda,ep,l,j)* currentUnitaries[l][j])
                #print(f"stored{storedStates}")

        ### FEEDFORWARD
        #Feedforward for given unitaries
        s = s + ep
        currentUnitaries = newUnitaries
        storedStates = feedforward(qnnArch, currentUnitaries, trainingData)

        #Cost calculation for given unitaries
        outputStates = []
        for m in range(len(storedStates)):
            outputStates.append(storedStates[m][-1])
        plotlist[0].append(s)
        plotlist[1].append(costFunction(trainingData, outputStates))
        ######Entropy_costfuntion####
        plotlist[2].append(entropy_costfunction(trainingData, outputStates))
        #print(f"outputlo:{outputStates} done")

    #Optional
    runtime = time() - runtime
    #print("Trained "+str(trainingRounds)+" rounds for relative entropy based method and  a "+str(qnnArch)+" network and "+str(len(trainingData))+" training pairs in "+str(round(runtime, 2))+" seconds")

    #Return
    return [plotlist, currentUnitaries]






# -----------------------------
# Config
# -----------------------------
qnnArch = [3, 4, 3]
num_samples = 100
training_rounds = 350
training_intervals = 10
k_folds = 3
confidence_level = 0.95
num_experiments = 2
train_ratio = 0.5  # 30% for training, 70% unseen test



# -----------------------------
# Helper functions
# -----------------------------
def evaluate_qnn(qnnArch, trainedUnitaries, data):
    storedStates = feedforward(qnnArch, trainedUnitaries, data)
    outputStates = [states[-1] for states in storedStates]
    return entropy_costfunction(data, outputStates)

def evaluate_fidelity(qnnArch, trainedUnitaries, data):
    storedStates = feedforward(qnnArch, trainedUnitaries, data)
    outputStates = [states[-1] for states in storedStates]
    fidelities = [qt.fidelity(out, target) for (_, target), out in zip(data, outputStates)]
    return np.mean(fidelities)

def safe_qnnTraining_methodA(qnnArch, unitaries, trainingData, lda, ep, rounds):
    return qnnTraining_en(qnnArch, unitariesCopy(unitaries), copy.deepcopy(trainingData), lda, ep, rounds, alert=0)

def safe_qnnTraining_methodB(qnnArch, unitaries, trainingData, lda, ep, rounds):
    return qnnTraining(qnnArch, unitariesCopy(unitaries), copy.deepcopy(trainingData), lda, ep, rounds, alert=0)

def mean_ci(values_list, confidence=0.95):
    arr = np.array(values_list)
    mean = arr.mean(axis=0)
    std = arr.std(axis=0)
    ci = stats.t.ppf(1 - (1 - confidence) / 2, df=arr.shape[0]-1) * std / np.sqrt(arr.shape[0])
    return mean, ci

# -----------------------------
# Storage
# -----------------------------
all_results = {
    "A": {"train_entropy": [], "train_fidelity": [], "cv_entropy": [], "cv_fidelity": [], "test_entropy": [], "test_fidelity": []},
    "B": {"train_entropy": [], "train_fidelity": [], "cv_entropy": [], "cv_fidelity": [], "test_entropy": [], "test_fidelity": []}
}
steps = training_rounds // training_intervals
x_axis_values = np.array([x * training_intervals for x in range(1, steps+1)])

import numpy as np
import copy

def qobj_to_numpy(u):
    """Recursively convert Qobj or nested lists of Qobj to NumPy arrays."""
    if hasattr(u, "full"):   # Qobj
        return u.full()
    elif isinstance(u, list) or isinstance(u, tuple):
        return [qobj_to_numpy(x) for x in u]
    else:
        return np.array(u)


def total_parameter_change(prev_unitaries, current_unitaries):
    """Compute total Frobenius norm change for lists/nested lists of Qobj/unitaries."""
    change = 0.0
    per_unitary_changes = []
    for prev, curr in zip(prev_unitaries, current_unitaries):
        prev_array = np.array(qobj_to_numpy(prev)).flatten()
        curr_array = np.array(qobj_to_numpy(curr)).flatten()
        unitary_change = np.linalg.norm(curr_array - prev_array)
        per_unitary_changes.append(unitary_change)
        change += unitary_change
    return change, per_unitary_changes


for exp_idx in range(num_experiments):
    print(f"\n===== Experiment {exp_idx+1}/{num_experiments} =====")
    np.random.seed(42 + exp_idx)

    # Generate network and data
    data = randomNetwork(qnnArch, num_samples)
    all_data = data[2]
    initial_unitaries = data[1]
    print(f"[DEBUG] Experiment {exp_idx+1}: data length={len(all_data)}, initial_unitaries type={type(initial_unitaries)}")

    # Manual train/test split
    train_idx, test_idx = train_test_split(
        np.arange(len(all_data)), train_size=train_ratio, shuffle=True, random_state=42+exp_idx
    )
    train_data = [copy.deepcopy(all_data[i]) for i in train_idx]
    test_data = [copy.deepcopy(all_data[i]) for i in test_idx]
    print(f"[DEBUG] Train/Test sizes: {len(train_data)}/{len(test_data)}")

    # Cross-validation
    cv = KFold(n_splits=k_folds, shuffle=True, random_state=42)
    for fold_idx, (cv_train_idx, cv_val_idx) in enumerate(cv.split(train_data)):
        print(f"\n=== CV Fold {fold_idx+1}/{k_folds} ===")
        cv_train = [copy.deepcopy(train_data[i]) for i in cv_train_idx]
        cv_val = [copy.deepcopy(train_data[i]) for i in cv_val_idx]
        print(f"[DEBUG] CV train/val sizes: {len(cv_train)}/{len(cv_val)}")

        unitaries_A = unitariesCopy(initial_unitaries)
        unitaries_B = unitariesCopy(initial_unitaries)

        prev_unitaries_A = copy.deepcopy(unitaries_A)
        prev_unitaries_B = copy.deepcopy(unitaries_B)

        # Stepwise metrics
        train_entropy_A, cv_entropy_A, train_fid_A, cv_fid_A = [], [], [], []
        train_entropy_B, cv_entropy_B, train_fid_B, cv_fid_B = [], [], [], []
        test_entropy_A, test_fid_A = [], []
        test_entropy_B, test_fid_B = [], []

        for step in range(steps):
            print(f"\n[DEBUG] Experiment {exp_idx+1}|CV Fold {fold_idx+1}|Step {step+1}/{steps}")

            # --- Method B ---
            entropy_plot_B, unitaries_B = safe_qnnTraining_methodB(
                qnnArch, unitaries_B, cv_train, lda=3, ep=0.06, rounds=training_intervals
            )

            # Compute total and per-unitary parameter change
            param_change_B, per_unitary_B = total_parameter_change(prev_unitaries_B, unitaries_B)
            print(f"[DEBUG] Method B total parameter change: {param_change_B:.4e}")
            for i, change_i in enumerate(per_unitary_B):
                if change_i > 0.5:
                    print(f"[DEBUG][Method B] Large jump in unitary {i}: {change_i:.4f}")
            prev_unitaries_B = copy.deepcopy(unitaries_B)

            # Evaluate metrics
            teB = evaluate_qnn(qnnArch, unitaries_B, cv_train)
            ceB = evaluate_qnn(qnnArch, unitaries_B, cv_val)
            tfB = evaluate_fidelity(qnnArch, unitaries_B, cv_train)
            cfB = evaluate_fidelity(qnnArch, unitaries_B, cv_val)
            test_teB = evaluate_qnn(qnnArch, unitaries_B, test_data)
            test_tfB = evaluate_fidelity(qnnArch, unitaries_B, test_data)

            # Detect sudden jumps/drops
            if train_entropy_B and abs(teB - train_entropy_B[-1]) > 0.5:
                print(f"[WARN] Sudden entropy jump (B): {train_entropy_B[-1]:.3f} -> {teB:.3f}")
            if train_fid_B and abs(tfB - train_fid_B[-1]) > 0.3:
                print(f"[WARN] Sudden fidelity drop (B): {train_fid_B[-1]:.3f} -> {tfB:.3f}")

            train_entropy_B.append(teB)
            cv_entropy_B.append(ceB)
            train_fid_B.append(tfB)
            cv_fid_B.append(cfB)
            test_entropy_B.append(test_teB)
            test_fid_B.append(test_tfB)

            # --- Method A ---
            entropy_plot_A, unitaries_A = safe_qnnTraining_methodA(
                qnnArch, unitaries_A, cv_train, lda=3, ep=0.06, rounds=training_intervals
            )

            param_change_A, per_unitary_A = total_parameter_change(prev_unitaries_A, unitaries_A)
            print(f"[DEBUG] Method A total parameter change: {param_change_A:.4e}")
            for i, change_i in enumerate(per_unitary_A):
                if change_i > 0.5:
                    print(f"[DEBUG][Method A] Large jump in unitary {i}: {change_i:.4f}")
            prev_unitaries_A = copy.deepcopy(unitaries_A)

            teA = evaluate_qnn(qnnArch, unitaries_A, cv_train)
            ceA = evaluate_qnn(qnnArch, unitaries_A, cv_val)
            tfA = evaluate_fidelity(qnnArch, unitaries_A, cv_train)
            cfA = evaluate_fidelity(qnnArch, unitaries_A, cv_val)
            test_teA = evaluate_qnn(qnnArch, unitaries_A, test_data)
            test_tfA = evaluate_fidelity(qnnArch, unitaries_A, test_data)

            if train_entropy_A and abs(teA - train_entropy_A[-1]) > 0.5:
                print(f"[WARN] Sudden entropy jump (A): {train_entropy_A[-1]:.3f} -> {teA:.3f}")
            if train_fid_A and abs(tfA - train_fid_A[-1]) > 0.3:
                print(f"[WARN] Sudden fidelity drop (A): {train_fid_A[-1]:.3f} -> {tfA:.3f}")

            train_entropy_A.append(teA)
            cv_entropy_A.append(ceA)
            train_fid_A.append(tfA)
            cv_fid_A.append(cfA)
            test_entropy_A.append(test_teA)
            test_fid_A.append(test_tfA)

        # Store results for this fold
        all_results["A"]["train_entropy"].append(train_entropy_A)
        all_results["A"]["cv_entropy"].append(cv_entropy_A)
        all_results["A"]["train_fidelity"].append(train_fid_A)
        all_results["A"]["cv_fidelity"].append(cv_fid_A)
        all_results["A"]["test_entropy"].append(test_entropy_A)
        all_results["A"]["test_fidelity"].append(test_fid_A)

        all_results["B"]["train_entropy"].append(train_entropy_B)
        all_results["B"]["cv_entropy"].append(cv_entropy_B)
        all_results["B"]["train_fidelity"].append(train_fid_B)
        all_results["B"]["cv_fidelity"].append(cv_fid_B)
        all_results["B"]["test_entropy"].append(test_entropy_B)
        all_results["B"]["test_fidelity"].append(test_fid_B)


# -----------------------------
# Compute mean + CI
# -----------------------------
summary = {}
for method in ["A","B"]:
    summary[method] = {}
    for metric in ["train_entropy","cv_entropy","test_entropy","train_fidelity","cv_fidelity","test_fidelity"]:
        mean_vals, ci_vals = mean_ci(all_results[method][metric], confidence_level)
        summary[method][metric] = (mean_vals, ci_vals)

# -----------------------------
# Save CSV
# -----------------------------
arch_str = "_".join(map(str, qnnArch))
filename = f"{arch_str}_results.csv"
rows = []
for method in ["A","B"]:
    for metric in ["train_entropy","cv_entropy","test_entropy","train_fidelity","cv_fidelity","test_fidelity"]:
        mean_vals, ci_vals = summary[method][metric]
        for step, (mean, ci) in enumerate(zip(mean_vals, ci_vals)):
            rows.append({
                "Method": "Entropy" if method=="A" else "Fidelity",
                "Metric": metric,
                "TrainingRound": x_axis_values[step],
                "Mean": mean,
                "CI": ci
            })
df = pd.DataFrame(rows)
df.to_csv(filename, index=False)
print(f" Results saved to {filename}")

# -----------------------------
# Plotting
# -----------------------------
# -----------------------------
# Plotting with updated legends
# -----------------------------
def plot_metric(metric_key, ylabel, title, filename):
    plt.figure(figsize=(6,5))
    for method, style, color in [("A","o-","blue"), ("B","s--","red")]:
        mean_vals, ci_vals = summary[method][metric_key]
        label = "Proposed (Relative Entropy based)" if method=="A" else "Existing (Fidelity based)"
        plt.errorbar(x_axis_values, mean_vals, yerr=ci_vals, fmt=style, color=color, capsize=4, label=label)
    plt.xlabel("Training Rounds")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.show()

# Generate plots for entropy and fidelity
for metric, ylabel in zip(
    ["train_entropy","cv_entropy","test_entropy","train_fidelity","cv_fidelity","test_fidelity"],
    ["Entropy","Entropy","Entropy","Fidelity","Fidelity","Fidelity"]
):
    plot_metric(metric, ylabel, f"{metric.replace('_',' ').upper()} Comparison", f"{arch_str}{metric}_comparison.png")


