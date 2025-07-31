import matplotlib.pyplot as plt
import json

with open('../results/result.json', 'r') as f:
    data = json.load(f)

for key, values in data.items():
    plt.figure()
    plt.plot(values, marker='o')
    plt.title(key)
    plt.xlabel('epoch')
    plt.ylabel(key)
    plt.grid(True)
    plt.tight_layout()
    plt.show()
