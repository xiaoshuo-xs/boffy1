import pennylane as qml
import torch.nn as nn
import torch
import numpy as np
import os
os.environ['OMP_NUM_THREADS'] = "256"

n_layers = 1  # Number of circuit layers
n_qubits_3 = 8 # Total number of qubits

# Specify backend simulator and number of qubits (default.qubit for pure states, default.mixed or lightning.qubit/gpu for others)
dev_3 = qml.device('lightning.qubit', wires=n_qubits_3, batch_obs = True) # Device for unsplitted model
theta1 = np.pi/3
theta2 = -np.pi/3
ttn = False

# Construct the encoding circuit
def embedding_circuit(inputs, n_qubits):
    for qub in range(n_qubits):
        qml.Hadamard(wires=qub)
        qml.RY(inputs[qub], wires=qub)

        
class LearnableScaledLayer(nn.Module):
    def __init__(self, initial_scale=5.0):
        super(LearnableScaledLayer, self).__init__()
        self.scale = nn.Parameter(torch.tensor(initial_scale))

    def forward(self, x):
        return x.to(torch.device("cuda" if torch.cuda.is_available() else "cpu")) * self.scale

# Unsplitted network
def tensor_network(layer, n_qubits, weights):
    param_index = 0
    if ttn :
        param_index = 0
        # Build pairs in a binary tree fashion
        for level in range(int(np.ceil(np.log2(n_qubits)))):
            step = 2**level
            for i in range(0, n_qubits - step, 2*step):
                if i + step >= n_qubits:
                    continue  # Skip if beyond qubit count
                    
                qml.RX(-np.pi / 2, wires=i+step)
                qml.CNOT(wires=[i+step, i])
                qml.RX(weights[layer, param_index] * theta1, wires=i)
                qml.RY(weights[layer, param_index + 1] * theta1, wires=i+step)
                qml.CNOT(wires=[i, i+step])
                qml.RY(weights[layer, param_index + 2] * theta1, wires=i+step)
                qml.CNOT(wires=[i+step, i])
                qml.RX(np.pi / 2, wires=i)
                
                qml.Barrier()
                param_index += 3

    else:
        for i in range(n_qubits-1):
            qml.RX(-np.pi / 2, wires=(i + 1) % n_qubits)
            qml.CNOT(wires=[(i + 1) % n_qubits, i])
            qml.RX(weights[layer, param_index] * theta1, wires=i)
            qml.RY(weights[layer, param_index + 1] * theta1, wires=(i + 1) % n_qubits)
            qml.CNOT(wires=[i, (i + 1) % n_qubits])
            qml.RY(weights[layer, param_index + 2] * theta1, wires=(i + 1) % n_qubits)
            qml.CNOT(wires=[(i + 1) % n_qubits, i])
            qml.RX(np.pi / 2, wires=i)
            
            qml.Barrier()
            param_index += 3 

def build_circuits_unsplitted(inputs, weights, qubits):
    embedding_circuit(inputs, qubits)
    for layer in range(n_layers):
        tensor_network(layer, qubits, weights)
    

@qml.qnode(dev_3, interface="torch")
def circuit_unsplitted(inputs, weights):
    build_circuits_unsplitted(inputs, weights, n_qubits_3)
    return qml.probs(wires=[i for i in range(n_qubits_3)])

class QNNUnsplitted(nn.Module):
    def __init__(self, input_shape, output_shape, hidden_dim=512, with_shortcut=False):
        super(QNNUnsplitted, self).__init__()

        self.mlp_input = nn.Sequential(nn.Linear(input_shape,hidden_dim), nn.Linear(hidden_dim, n_qubits_3))
        self.mlp_output = nn.Linear(2**n_qubits_3, output_shape)
        self.mlp_withshortcut = nn.Linear(2**n_qubits_3 + input_shape, output_shape)
        # Define shared weight tensor
        self.shared_weights = nn.Parameter(torch.rand((n_layers, (n_qubits_3 - 1) * 6)), requires_grad=True)
        weight_shapes_1 = {"weights": (n_layers, 2 * (n_qubits_3 - 1) * 3)} # Specify weights shape
        self.QLayer = qml.qnn.TorchLayer(circuit_unsplitted, weight_shapes_1)  # QNode to TorchLayer
        self.QLayer.weights.data = self.shared_weights

        self.tanh = nn.Tanh()
        self.activate = nn.Sigmoid()
        self.scaler = LearnableScaledLayer()
        self.scaler_input = LearnableScaledLayer()
        self.with_shortcut = with_shortcut

    def forward(self, inputs): # inputs = stack(batch_size, x], dim=0)
        
        if self.with_shortcut:
            shortcut_x = inputs
        
        inputs = self.mlp_input(inputs) #torch.Size([batch_size, Qbits])
        inputs = self.scaler_input(inputs)
        output_pack = []
        for i in inputs:
            output_pack.append(self.QLayer(i))
        combined_outputs = torch.stack(output_pack, dim=0).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

        if self.with_shortcut:
            outputs = torch.cat((combined_outputs, shortcut_x), dim=1)
            output_tensor = self.mlp_withshortcut(outputs)
        else:
            output_tensor = self.mlp_output(combined_outputs)
        
        return output_tensor
