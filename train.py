from Neural_Network import text_recog as tr
import numpy as np
X_train, y_train = tr.load_data("data/traindata.mat.tar.gz")
X_test, y_test = tr.load_data("data/testdata.mat.tar.gz")
#tr.display_data(X_train[:10], 2, 5)

nn = tr.NN_hwr([len(X_train[0]), 15, 10])
nn.train_nn(X_train, y_train, 10, 20, 0.06)

accuracy = 0
for i in range(len(X_test[:100])):
    out = nn.forward_prop(X_test[i])[0][-1]
    if np.argmax(out) == np.where(y_test[i])[0][0]:
        accuracy += 1
        print(True, np.argmax(out))
    else:
        print(False, np.argmax(out))
np.savetxt("weights0.csv",nn.weights[0],delimiter=",")
np.savetxt("weights1.csv",nn.weights[1],delimiter=",")
np.savetxt("biases0.csv",nn.biases[0],delimiter=",")
np.savetxt("biases1.csv",nn.biases[1],delimiter=",")
print("accuracy: ", accuracy)