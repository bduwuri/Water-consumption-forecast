import sys
import numpy as np
import math

def sigmoid(a):                            #https://github.com/Kulbear/deep-learning-nano-foundation/wiki/ReLU-and-Softmax-Activation-Functions
    return 1 / (1 + np.exp(-a))

def sigmoid_deriv(a):
	return sigmoid(a) * (1 - sigmoid(a))

def softmax(b):
    return np.exp(b)/sum(np.exp(b))

def get_random_weight(r0, r1):
    c = np.random.uniform(low = -0.1, high = 0.1, size = (r0, r1))
    c[:,0] = 0
    return c

def get_zero_weight(r0, r1):
    w = np.zeros([r0, r1])
    w[:, 0] = 0
    return w

# def get_bias(r):
# 	return np.zeros(r)

class NeuralNetwork:
    def __init__(self, x, z, y, learning_rate, init_flag):
        self.number_of_input_nodes = len(x[0])
        self.number_of_hidden_nodes = int(z)
        self.number_of_output_nodes = len(y)
        self.learning_rate = float(learning_rate)
        self.alpha = None
        self.deriv_alpha = []
        self.beta = None
        self.deriv_beta = []
        self.input = None
        self.output = None
        self.y_hat = None
        self.y = None
        self.a = None
        self.z = None 
        self.b = None
        # print("type: ", type(self.alpha))
        # self.activation_function = sigmoid

        # x = 128
        # y = 10
        # self.alpha = []
        # self.deriv_alpha = []
        # self.beta = []
        # self.deriv_beta = []
        # self.bias = []
        # self.deriv_bias = []

        if init_flag == 1:
            self.weight_init = get_random_weight
        else:
            self.weight_init = get_zero_weight
        
        self.alpha = self.weight_init(int(z), len(x[0]))
        self.beta = self.weight_init(10, int(z)+1)
        # print(len(self.alpha[0]))
        # print(len(self.beta[0]))
        # print(len(x[0]))
    def feedforward_propagation(self, x, y):

        def cross_ent_loss(y_hat, y):
            return (-1)*(np.sum(np.dot(y, np.log(y_hat))))
        
        self.input = x
        self.y = y
        self.a = np.dot(self.alpha, x)
        #print(self.a[0])
        self.z = sigmoid(self.a) 
        self.z = np.insert(self.z, 0, 1, axis = 0)
        #print(self.z)
        #print(self.beta)
        self.b = np.dot(self.beta, self.z)
        #print(self.b)
        self.y_hat = softmax(self.b)
        #print(self.y_hat)
        # print(len(self.alpha))
        # print(len(x))
        # print(len(self.beta))
        # print(len(self.a))
        # print(len(self.z))  
        # print(self.alpha)
        # # print(self.z)
        # convert probabilistic representation to one-hot vector
        # self.y_hat = np.zeros_like(y_hat)
        # self.y_hat_idx=self.y_hat.argmax() 
        # print(self.y_hat)

        loss = cross_ent_loss(self.y_hat, y) 

        return loss
    
    def backpropagation(self, loss): 
        
        # dl/dbeta = dcrossentloss/dsoftmaxactivation * dsoftmaxactivation/dy * dy/dz
        # dcrossentloss/dsoftmaxactivation * dsoftmaxactivation/dy = y_hat (1 - y)
        # dl_dy = self.y_hat * (1 - self.y)
        dl_db = self.y_hat - self.y
        #print(dl_db)
        # separate weight and bias
        # dl_dbetaw = np.dot(dl_dy,np.transpose(self.z))
        dl_dbetaw = np.outer(dl_db, np.transpose(self.z))
        # print(dl_dbetaw)
        # dl_dbetab = np.dot(dl_dy,np.ones((self.alpha.shape[0],1)))
        # dl_dbetab = np.dot(dl_db,np.ones((self.alpha.shape[0],1)))
        
        # dl/dz = loss passed on to previous layer
        beta_star = np.array(self.beta[:, 1:])
        #print(beta_star)
        # dl_dz = np.dot(beta_star, dl_dy)
        dl_dz = np.dot(np.transpose(beta_star), dl_db)

        z_star = np.delete(self.z, 0)
        #print(z_star)

        # derivative of sigmoid
        dl_dsig = np.multiply(z_star, (1-z_star))

        dlda = np.multiply(dl_dsig, dl_dz)

        # separate weight and bias  
        dl_dalphaw = np.outer(dlda, np.transpose(self.input))
        # dl_dalphab = np.dot(dlda,np.ones((self.input.shape[0], 1)))
        #print(dl_dalphaw)

        # combine weights and bias
        # dl_dalpha = np.hstack((dl_dalphab, dl_dalphaw))
        # dl_dbeta = np.hstack((dl_dbetab, dl_dbetaw))

        return dl_dalphaw, dl_dbetaw
    
    def update(self, dlderiv_alpha, dlderiv_beta):
        self.alpha -= self.learning_rate * dlderiv_alpha
        self.beta -= self.learning_rate * dlderiv_beta
        
def get_input_data(input_file):
    label = []
    true_label = []
    data = []
    with open(input_file, "r") as f:
        read_out = f.readlines()
    for content in read_out:
        label_vec = np.zeros(10)
        label_vec[int(content.split(',')[0])] = 1
        label.extend(label_vec.reshape(1, 10))
        true_label.append(int(content.split(',')[0]))
        data_vec = [int(item) for item in content.split(',')[1:]]
        # print(data_vec)
        # print(label_vec)
        data.extend(np.array(data_vec).reshape(1, 128))
        # print(data)
    return data, label, true_label

if __name__ == "__main__":
    train_input = sys.argv[1]
    test_input = sys.argv[2]
    train_out = sys.argv[3]
    test_out = sys.argv[4]
    metrics_out = sys.argv[5]
    num_epoch = int(sys.argv[6])
    hidden_units = sys.argv[7]
    init_flag = sys.argv[8]
    learning_rate = sys.argv[9]

    # train_X, train_Y = get_input_data(train_input)
    train_X, train_Y,train_true = get_input_data(train_input)
    test_X, test_Y, test_true = get_input_data(test_input)
    metrics_out_file = open(metrics_out, "w")
    train_out_file = open(train_out, "w")
    test_out_file = open(test_out, "w")
    #print(len(train_X))
    #print(len(train_Y))
    nn = NeuralNetwork(train_X, hidden_units, train_Y, learning_rate, init_flag)
    #nn = NeuralNetwork()
    #l = dir(nn)
    # print(l)
    # print(type(nn))
    # print(type(nn.learning_rate))
    # nn.number_of_input_nodes=len(train_X)
    # nn.number_of_hidden_nodes = hidden_units
    # nn.number_of_output_nodes = len(train_Y)
    # nn.learning_rate = learning_rate
    #print(nn.alpha)
    #nn.init_flag
    #nn = NeuralNetwork(x = 128, hidden_units, y = 10, learning_rate, init_flag)
    train_error = 0
    test_error = 0
    J= np.zeros(len(train_X))
    counter = 0

    for e in range(num_epoch):
        cross_ent_train = []
        cross_ent_test = []
    
        for data, label in zip(train_X, train_Y):
            loss = nn.feedforward_propagation(data, label)
            grad_alpha, grad_beta = nn.backpropagation(loss)
            if counter < 3:
                # print(grad_beta)
                counter += 1
            nn.update(grad_alpha,grad_beta)
            # cross_ent_train.append(loss)

        count = 0

        for data, label in zip(train_X, train_Y):
            loss = nn.feedforward_propagation(data, label)
            # J[count]= -np.dot(nn.y,np.log(nn.y_hat))
            J[count] = loss
            count += 1

        mean_cross_ent_train = np.mean(J)

        # print(np.mean(cross_ent_train))
        
        # print("epoch=%d crossentropy(train): %f" % (e+1,mean_cross_ent_train))
        metrics_out_file.write("epoch={} crossentropy(train): {}\n".format(e+1, mean_cross_ent_train))

        # train_error  = mean_cross_ent_train
        count=0
        for data, label in zip(test_X, test_Y):
            loss = nn.feedforward_propagation(data, label)
            # J[count]= -np.dot(nn.y,np.log(nn.y_hat))
            J[count] = loss
            count += 1
        
        
        
        #for data, label in zip(test_X, test_Y):
        #for data in range(len(test_X)):
            # print(type(data))
            # loss = nn.feedforward_propagation(data, label)
        #    loss = nn.feedforward_propagation(test_X[data], label)
        #    cross_ent_test += loss
        mean_cross_ent_test = np.mean(J)
        # mean_cross_ent_test = cross_ent_test/float(len(test_X))
        # print("epoch=%d crossentropy(test): %f" % (e+1, mean_cross_ent_test))
        metrics_out_file.write("epoch={} crossentropy(test): {}\n".format(e+1, mean_cross_ent_test))

        # test_error = mean_cross_ent_test
    pred_train = []
    for train_data, train_label in zip(train_X, train_Y):
        loss = nn.feedforward_propagation(train_data, train_label)
        pred_train.append(nn.y_hat.argmax())
        # pred_train.append(loss)
        # print(nn.y_hat.argmax())
    pred_test = []
    for test_data, test_label in zip(test_X, test_Y):
        loss = nn.feedforward_propagation(test_data, test_label)
        pred_test.append(nn.y_hat.argmax())  
        # pred_test.append(loss)

    for context in pred_train:
        train_out_file.write("%d\n" %context)  
    for context in pred_test:
        test_out_file.write("%d\n" %context)
    
    error_count = 0
    for index in range(len(pred_train)):
        if pred_train[index] != train_true[index]:
            # print(train_true[index])
            error_count +=1
    train_error = float(error_count)/len(pred_train)
    
    total_count = 0
    error_count = 0
    for index in range(len(pred_test)):
        if pred_test[index] != test_true[index]:
           error_count +=1
        total_count += 1
    test_error = float(error_count)/total_count
    
    # print("error(train): {}".format(np.round(train_error,2)))
    metrics_out_file.write("error(train): {}\n".format(np.round(train_error,2)))
    # print("error(test): {}".format(np.round(test_error,2)))
    metrics_out_file.write("error(test): {}\n".format(np.round(test_error,2)))
        





