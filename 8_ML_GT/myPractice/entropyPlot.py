
import math
import matplotlib.pyplot as plt

pos = [i*.02 for i in range(1,50)]

print(pos)



ent = [-(p1*math.log2(p1) + (1-p1)*math.log2((1-p1)) )  for p1 in pos ]

pos.append(1)
ent.append(0)

pos.insert(0,0)
ent.insert(0,0)

print()
print(ent)

fig, ax = plt.subplots()
ax.plot(pos, ent, marker = 'o')

ax.set(xlabel='probability', ylabel='entropy', title='entropy for binary classification')
ax.grid()

#fig.savefig("test.png")
plt.show()

