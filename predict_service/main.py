from prometheus_api_client import PrometheusConnect  
from kubernetes import client, config  
  
# Load kube config  
config.load_kube_config()  
  
# Create a new PrometheusConnect object  
prom = PrometheusConnect(url="http://localhost:9090", disable_ssl=True)  
  
# Fetch Prometheus data  
my_metric_data = prom.get_current_metric_value(metric_name='my_metric')  
  
# Here you can add your logic  
# For example, you might want to calculate the average  
average = sum(my_metric_data) / len(my_metric_data)  
  
# Create a new HPA object  
v2beta2 = client.AutoscalingV2beta2Api()  
  
# Fetch the current HPA  
hpa = v2beta2.read_namespaced_horizontal_pod_autoscaler(name='my-hpa', namespace='default')  
  
# Update the HPA  
hpa.spec.min_replicas = average  # Just an example, you'll need to adjust this based on your logic  
v2beta2.patch_namespaced_horizontal_pod_autoscaler(name='my-hpa', namespace='default', body=hpa)  
