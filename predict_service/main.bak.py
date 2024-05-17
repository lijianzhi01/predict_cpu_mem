import requests
import torch
from sklearn.preprocessing import MinMaxScaler  
from statsmodels.tsa.seasonal import STL  
import pandas as pd  
import numpy as np  
from kubernetes import client, config  
import time

# Load kube config  
config.load_kube_config()  
  
# Define the parameters for the query  
params = {  
    'query': 'sum(rate(container_cpu_usage_seconds_total{container_label_io_kubernetes_pod_namespace="demo"}[30s]))',  
    'start': time.time() - 3600,  
    'end': time.time(),  
    'step': 60,  
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
#                 [1715947813.249, "0.4374023430589249"]
# 			]
# 		}]
# 	}
# }
values = data['data']['result'][0]['values']
df = pd.DataFrame(values, columns=['time', 'cpu_usage'])
df.reset_index(drop=True, inplace=True)  
df['cpu_usage'] = df['cpu_usage'].astype(float)
df = df.iloc[:48]
print(df)

scaler = MinMaxScaler(feature_range=(0, 1))  
df['cpu_usage'] = scaler.fit_transform(df['cpu_usage'].values.reshape(-1, 1))  
stl = STL(df['cpu_usage'], seasonal=12)  
result = stl.fit()  
df['trend'] = result.trend  
df['detrended'] = df['cpu_usage'] - df['trend']  

seq_length_in = 48  
input_size = 1
trend_data = df['trend'].values[-seq_length_in:]  
detrended_data = df['detrended'].values[-seq_length_in:]  

model = torch.load('./models/10.4_LSTM_STL_Attenton_FFT_seq2seq_one_machine_miniSGD.pth')  
model.load_state_dict(torch.load('./models/10.4_LSTM_STL_Attenton_FFT_seq2seq_one_machine_miniSGD.pth'))  

# Convert them to tensor and reshape them to match the input format of your model  
trend_tensor = torch.FloatTensor(trend_data).view(-1, seq_length_in, input_size)  
detrended_tensor = torch.FloatTensor(detrended_data).view(-1, seq_length_in, input_size)  
  
# Move your tensors to the correct device  
device = torch.device("cpu")  
trend_tensor = trend_tensor.to(device)  
detrended_tensor = detrended_tensor.to(device)  
  
# Use your model to predict the future CPU usage  
model.eval()  # Set the model to evaluation mode  
with torch.no_grad():  
    prediction = model(trend_tensor, detrended_tensor)  
  
# Convert the prediction to a numpy array  
prediction = prediction.cpu().numpy()  

print(prediction)
  
# Here you can add your logic  
# For example, you might want to calculate the average  
# average = sum(my_metric_data) / len(my_metric_data)  
  
# # Create a new HPA object  
# v2beta2 = client.AutoscalingV2beta2Api()  
  
# # Fetch the current HPA  
# hpa = v2beta2.read_namespaced_horizontal_pod_autoscaler(name='my-hpa', namespace='default')  
  
# # Update the HPA  
# hpa.spec.min_replicas = average  # Just an example, you'll need to adjust this based on your logic  
# v2beta2.patch_namespaced_horizontal_pod_autoscaler(name='my-hpa', namespace='default', body=hpa)  
