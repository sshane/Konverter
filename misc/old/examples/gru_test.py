import numpy as np
import tensorflow as tf

def tanh(x):
  return np.tanh(x)

def sigmoid(x):
  return 1 / (1 + np.exp(-x))


dense_w = np.array([[2.2967770099639893], [-2.803175449371338], [-3.1811068058013916], [-2.7010140419006348]])
dense_b = np.array([1.7280919551849365])

input_kernel = np.array([[-0.17922091484069824, -0.7751619219779968, -0.5937628746032715, 0.17969973385334015, -0.3332160413265228, 1.2726248502731323, 0.12554332613945007, -0.03203260526061058, 0.30094385147094727, 0.0037008628714829683, 0.011297334916889668, -0.36017563939094543], [-0.015779945999383926, -0.20816829800605774, 0.0006857096450403333, 0.03646666929125786, 0.12818121910095215, 1.220643401145935, 0.0504753440618515, 0.21695120632648468, 0.07797575742006302, -0.001003757817670703, 0.00500526325777173, -0.01619562692940235]])
recurrent_kernel = np.array([[-0.7426908016204834, -0.9402485489845276, -1.0539801120758057, -2.4390223026275635, -0.6458730101585388, 1.1413075923919678, 0.7746387124061584, 1.1663264036178589, -0.8537243604660034, -1.1318180561065674, -0.5453411340713501, -0.9517378807067871], [-3.132922649383545, -0.44755667448043823, 1.197575569152832, -1.479921579360962, 1.7497535943984985, 0.11262882500886917, -1.2965713739395142, -0.7393601536750793, -1.3990355730056763, -0.06538951396942139, 1.357919692993164, 1.5310022830963135], [-0.6719138622283936, -0.5418422818183899, -0.18892934918403625, 0.24632994830608368, 0.8286199569702148, 0.6987065076828003, -0.993728518486023, -0.18839183449745178, -0.451945424079895, -0.7261284589767456, 0.7406968474388123, 0.4892735779285431], [-0.9068083763122559, 0.6613847017288208, 0.6276863813400269, -0.14749160408973694, 0.7953370213508606, -1.7424216270446777, -1.0275965929031372, -1.0693542957305908, 1.0246773958206177, 2.027177095413208, 1.244929313659668, 0.5734887719154358]])
bias = np.array([[0.9735522866249084, -1.3079789876937866, -0.9024412035942078, -0.2866840064525604, 0.11746568977832794, 1.997322916984558, 0.3525863587856293, 0.3685975670814514, -0.0795750543475151, 0.35735249519348145, 0.840775191783905, -0.593677818775177], [0.9735522866249084, -1.3079789876937866, -0.9024412035942078, -0.2866840064525604, 0.11746568977832794, 1.997322916984558, 0.3525863587856293, 0.3685975670814514, -0.5861431956291199, 0.32561612129211426, 0.4003056585788727, -0.5730466842651367]])

sample = np.array([[4, 4], [2.5, 1], [2, 2], [4, 4]])
x_e = sample
units = 4

Hts = [np.zeros(12)]
outputs = []
for t in range(len(sample)):
  print(t)
  Z = np.dot(sample[t], input_kernel)
  Z += np.dot([Hts[-1]], recurrent_kernel[0])
  Z += bias[0]
  Z = sigmoid(Z)

  R = np.dot(sample[t], input_kernel)
  R += np.dot([Hts[-1]], recurrent_kernel[1])
  R += bias[0]
  R = sigmoid(R)
  # H_tilda = np.dot(sample[t], input_kernel)
  # print(H_tilda.shape)
  H_tilda = np.tanh(np.dot(sample[t], input_kernel) + np.dot(R[t] * Hts[-1], recurrent_kernel[t]) + bias[0])
  Hts.append(Z * Hts[-1] + (1 - Z) * H_tilda)

  Y = np.dot(Hts[-1], recurrent_kernel[t])
  outputs.append(Y)

l0 = np.dot(outputs, dense_w) + dense_b
print(l0.tolist())

# for t in range(len(sample)):
#   print(t)
#   Rt = np.dot(sample[t], input_kernel)
#   Rt += np.dot(Hts[-1], recurrent_kernel[0])
#   Rt += bias[0]
#   print(Rt.shape)
#
#   Zt = np.dot(sample[t], input_kernel)
#   Zt += np.dot(Hts[-1], recurrent_kernel[1])
#   Zt += bias[1]
#
#   Ht = np.dot(sample[t], input_kernel)
#   Ht += np.dot(Ht * Rt, recurrent_kernel[t]) + bias[0]
#   Ht = tanh(Ht)
#   print(Ht.shape)
#   Ht = Zt * Ht + (1 - Zt) * Ht
#   Y = np.dot(Ht, recurrent_kernel[t]) + bias
#   Hts.append(Ht)
#
#   # add = mulw + mulu + bias
#   # z = np.dot(sample[0], input_kernel)  #  + np.dot(Uz, h) + bz)
#
# l0 = np.dot(Ht, dense_w) + dense_b
# print(l0.tolist())



# def forward_prop_step(x_t, s_t1_prev):
#   z_t1 = sigmoid(recurrent_kernel[0].dot(x_e) + input_kernel[0].dot(s_t1_prev) + bias[0])
#   r_t1 = sigmoid(recurrent_kernel[1].dot(x_e) + input_kernel[1].dot(s_t1_prev) + bias[1])
#   c_t1 = tanh(recurrent_kernel[2].dot(x_e) + input_kernel[2].dot(s_t1_prev * r_t1) + bias[2])
#   s_t1 = (np.ones_like(z_t1) - z_t1) * c_t1 + z_t1 * s_t1_prev
#
#   # o_t = T.nnet.softmax(V.dot(s_t1) + c)[0]
#
# forward_prop_step(1, np.zeros(4))


# timesteps = sample.shape[0]
# prev_s = np.zeros(4)
# for step in range(timesteps):
#   mulu = np.dot(sample[step], kernel)
#   mulw = np.dot(prev_s, recurrent_kernel)
#   add = mulw + mulu + bias
#   s = np.tanh(add)
#   mulv = np.dot(recurrent_kernel, s)
#   prev_s = np.array(s)
#
# l0 = np.dot(s, dense_w) + dense_b
# print(l0.tolist())


#
# weights = np.transpose(np.concatenate([np.transpose(input_matrix), recurrent_matrix], 1))
#
# gate_inputs = np.concatenate([sample, np.zeros(16)], 1)
# gate_inputs = np.matmul(gate_inputs, weights)
#
#
# gate_inputs = np.bias_add(gate_inputs, bias)
#
# output = tanh(gate_inputs)
# print(output)