from SumTree import Memory
import numpy as np
m = Memory(12,0)
act = np.zeros([10, 1])
for i in range(1,7):
	m.store(np.array([i]))
	m.batch_update(np.array([i + 14]), np.array([float(i**1)]))
for j in range(50000):
	t, b, _, _ = m.sample(3, 6)
	act[int(b[0]), 0] += 1
	act[int(b[1]), 0] += 1
	act[int(b[2]), 0] += 1
print(act)
t, b, IS, _ = m.sample(6, 6)
print(IS)

act *= 0
for j in range(50000):
	t, b, _, _ = m.sample(3, 6)
	act[int(b[0]), 0] += 1
	act[int(b[1]), 0] += 1
	act[int(b[2]), 0] += 1
print(act)
t, b, IS, _ = m.sample(6, 6)
print(IS)
#
# m.batch_update(t, np.array([0., 0.]))
# print('emmmmmmmmmmmmmmm')
# for i in range(6,11):
# 	m.store(np.array([i]))
# 	m.batch_update(np.array([i+8]), np.array([30.]))
# for i in range(12):
# 	_, b, _, _ = m.sample(2, 5)
# 	print(b)