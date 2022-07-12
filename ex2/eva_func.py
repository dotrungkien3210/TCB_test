import matplotlib.pyplot as plt
import numpy as np
import math

data1 = np.array([1.28921989, -1.78732757, -1.78732757, 0.20510316, 0.20510316, -0.29300452])
data2 = np.array([-1.28921989, -0.29300452, -1.78732757, 1.69942621, 0.20510316, 1.20131853])
data3 = np.array([0.20510316, 0.70321085, 1.20131853, 0.20510316, 1.20131853, 1.69942621])
data4 = np.array([-0.29300452, -0.29300452, -0.7911122, -0.7911122, 0.70321085, 1.69942621])
data5 = np.array([-0.7911122, -0.7911122, -0.7911122, 1.69942621, 1.20131853, 1.69942621])

use_data1 = data5
use_data2 = np.add(data1, data2)
use_data2 = np.add(use_data2, data3)
use_data2 = np.add(use_data2, data4)
z_score_1 = []
z_score_2 = []

for x in np.nditer(use_data1):
    z_score_1.append((x+0.37113 )/26.5203 )
print("\n")
for x in np.nditer(use_data2):
    z_score_2.append((x+0.0183)/31.47)
sum1 = sum(z_score_1)
sum2 = sum(z_score_2)

lambda_array = np.geomspace(0.01, 0.99, 100)

func = []
for p in lambda_array:
    loglik = -np.log(1.535*math.sqrt(1-math.pow(p,2)))-(1/(2*(1-math.pow(p,2))))*(math.pow(sum1,2)-2*p*sum1*sum2+math.pow(sum2,2))
    func.append(loglik.sum())

plt.plot(lambda_array, func)
plt.xlabel('p')
plt.ylabel('Log-likelihood')
plt.title('Log likelihood over a range of p values')
plt.show()
