import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Settings
num_packets = 1000
start_time = datetime.now()

# Lists to hold data
timestamps = []
sizes = []
protocols = []
directions = []
labels = []

current_time = start_time

for i in range(num_packets):
    # Determine if this section is "Malware" or "Safe"
    # Switch every 100 packets
    is_malware = 1 if (i // 100) % 2 == 1 else 0
    
    if is_malware:
        # MALWARE BEHAVIOR:
        # - Small packets (beaconing)
        # - Exact 5.0 second intervals (The "Liquid" Signal)
        current_time += timedelta(seconds=5.0)
        size = 40 # Tiny keep-alive packet
        proto = 17 # UDP
        direction = 1 # Outbound
    else:
        # SAFE BEHAVIOR:
        # - Random large packets (YouTube, Browsing)
        # - Random human intervals (0.1s to 2.0s)
        current_time += timedelta(seconds=np.random.uniform(0.1, 2.0))
        size = np.random.randint(500, 1500)
        proto = 6 # TCP
        direction = np.random.choice([0, 1]) # In/Out

    timestamps.append(current_time)
    sizes.append(size)
    protocols.append(proto)
    directions.append(direction)
    labels.append(is_malware)

# Create DataFrame
df = pd.DataFrame({
    'timestamp': timestamps,
    'packet_size': sizes,
    'protocol': protocols,
    'direction': directions,
    'is_malware': labels
})

# Save to the correct folder
df.to_csv('data/traffic_log.csv', index=False)
print("Success! Generated data/traffic_log.csv with 1000 packets.")