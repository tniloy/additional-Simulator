import numpy as np
import matplotlib.pyplot as plt


# creating the dataset
data = {
        "2000": 1.18443762809318,
        "3000": 0.019499911991802222,
        "4000": 9.47391869512967,
        "5000": 0.008864100969138413,
        "6000": 0.0009648452070193107
    }

cum_inter = list(data.values())
for i, key in enumerate(data.keys()):
        if i<1:
                continue
        print(i)
        cum_inter[i] += cum_inter[i-1]




courses = list(data.keys())
# values = list([10 * np.log10(inr) for inr in cum_inter])
values = list([10 * np.log10(inr) for inr in data.values()])

fig = plt.figure(figsize = (10, 5))

# creating the bar plot
plt.bar(courses, values, color ='maroon',
		width = 0.4)


plt.xlabel('Distance of Each BS From FSS (meters)')
plt.ylabel('Aggregate I/N (dB)')
plt.title('Sunny Weather')
plt.show()