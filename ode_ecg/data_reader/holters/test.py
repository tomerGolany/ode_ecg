from ishneholterlib import Holter
from matplotlib import pyplot as plt

# Load a file from disk:
x = Holter('alexander.ecg')
x.load_data()

# print(x)

lead_0 = x.lead[1]
ecg_data = lead_0.data
print("Number of values in ecg signal : ", len(ecg_data))
plt.plot(ecg_data[950000:950000 + 100])
plt.show()