import requests
import torch
from sklearn.preprocessing import MinMaxScaler  
import pandas as pd  
import numpy as np  
from kubernetes import client, config  
import time
import AttnConvLSTM

# Load kube config  
config.load_kube_config()  
  
# Define the parameters for the query  
params = {  
    'query': 'sum(rate(container_cpu_usage_seconds_total{container_label_io_kubernetes_pod_namespace="demo"}[30s]))',  
    'start': time.time() - 3600 * 4,  
    'end': time.time(),  
    'step': 60,  # define the interval of time (in seconds) between each data point
}  
  
# Send the GET request to the Prometheus API  
response = requests.get('http://localhost:9090/api/v1/query_range', params=params)  
data = response.json()  

# Extract the values from the response  
# {
# 	"status": "success",
# 	"data": {
# 		"resultType": "matrix",
# 		"result": [{
# 			"metric": {},
# 			"values": [
# 				[1715947513.249, "0.447546652114158"], // 2024-05-17 20:05:13
#                 [1715947573.249, "0.42269904586241663"], // 2024-05-17 20:06:13
#                 [1715947633.249, "0.46809806103791185"], // 2024-05-17 20:07:13
#                 [1715947693.249, "0.41559276797697664"],
#                 [1715947753.249, "0.48254878836485465"],
#                 [1715947813.249, "0.4374023430589249"],
#                 ......
# 			]
# 		}]
# 	}
# }
seq_length_in = 144
seq_length_out = 6
input_size = 1
num_epochs = 100  
learning_rate = 0.01  
hidden_size = 10  
num_layers = 1  
kernel_size = 20
output_size = seq_length_out  
device = torch.device("cpu")  
values = data['data']['result'][0]['values']
df = pd.DataFrame(values, columns=['timestamp', 'cpu_usage'])
df = df.iloc[-seq_length_in:]
print(df)

scaler = MinMaxScaler(feature_range=(0, 1))  
df['cpu_usage'] = scaler.fit_transform(df['cpu_usage'].values.reshape(-1,1))  
cpu_usage = torch.FloatTensor(df['cpu_usage'].values).to(device)  
cpu_usage = torch.FloatTensor(cpu_usage).view(-1, seq_length_in, input_size).to(device)  

model = AttnConvLSTM.AttnConvLSTM(input_size, hidden_size, num_layers, output_size, kernel_size).to(device)
state_dict = torch.load('./models/8_LSTM_seq2seq_multiple_machine_miniSGD.pth', map_location=device)
model.load_state_dict(state_dict)

# Use your model to predict the future CPU usage  
model.eval()  # Set the model to evaluation mode  
with torch.no_grad():  
    prediction = model(cpu_usage).to(device)

# Convert the prediction to a numpy array  
prediction = prediction.cpu().numpy()  

print(prediction)