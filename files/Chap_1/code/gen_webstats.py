# This script generates web traffic data for our hypothetical
# web startup "MLASS" in chapter 01

import os
import scipy as sp
import numpy as np
from scipy.stats import gamma
import matplotlib.pyplot as plt

sp.random.seed(3)  # to reproduce the data later on

x = sp.arange(1, 31 * 24)
y = sp.array(200 * (sp.sin(2 * sp.pi * x / (7 * 24))), dtype=int)
np.add(y, gamma.rvs(15, loc=0, scale=100, size=len(x)), out = y, casting = "unsafe")
# np.add(a, b, out=a, casting="unsafe")
# y += gamma.rvs(15, loc=0, scale=100, size=len(x))
np.add(y, 2 * sp.exp(x / 100.0), out = y, casting = "unsafe")
# y += 2 * sp.exp(x / 100.0)
y = sp.ma.array(y, mask=[y < 0])
print(sum(y), sum(y < 0))

plt.scatter(x, y)
plt.title("Web traffic over the last month")
plt.xlabel("Time")
plt.ylabel("Hits/hour")
plt.xticks([w * 7 * 24 for w in [0, 1, 2, 3, 4]], ['week %i' % (w + 1) for w in [
           0, 1, 2, 3, 4]])

plt.autoscale(tight=True)

# plt.savefig(os.path.join("..", "1400_01_01.png"))
#
# data_dir = os.path.join(
#     os.path.dirname(os.path.realpath(__file__)), "..", "data")
#
# # sp.savetxt(os.path.join("..", "web_traffic.tsv"),
# # zip(x[~y.mask],y[~y.mask]), delimiter="\t", fmt="%i")
# sp.savetxt(os.path.join(
#     data_dir, "web_traffic.tsv"), list(zip(x, y)), delimiter="\t", fmt="%s")

fp1, residuals, rank, sv, rcond = sp.polyfit(x, y, 1, full=True)

print("Model parameters: %s" % fp1)
print(residuals)
def error(f, x, y):
    return sp.sum((f(x) - y) ** 2)
# f(x) = 2.57152281 * x + 1002.10684085



f2p = sp.polyfit(x, y, 2)
f3p = sp.polyfit(x, y, 3)
f10p = sp.polyfit(x, y, 10)
f100p = sp.polyfit(x, y, 100)


f1 = sp.poly1d(fp1)
f2 = sp.poly1d(f2p)
f3 = sp.poly1d(f3p)
f10 = sp.poly1d(f10p)
f100 = sp.poly1d(f100p)

inflection = int(3.5*7*24)
xa = x[:inflection]
ya = y[:inflection]
xb = x[inflection:]
yb = y[inflection:]

my_list = [f1, f2, f3, f10, f100]

fx = sp.linspace(0,x[-1], 1000)

fa = sp.poly1d(sp.polyfit(xa, ya, 1))
fb = sp.poly1d(sp.polyfit(xb, yb, 1))

plt.plot(fa, linewidth=4)
plt.plot(fb, linewidth=4)
for plot_line in my_list:
    plt.plot(fx, plot_line(fx), linewidth=4)


for fsake in my_list:
    print ("Error of %i " % fsake.order , error(fsake, x, y))




plt.legend(["d=%i" % f.order for f in my_list], loc="upper left")

plt.grid()
plt.show()
