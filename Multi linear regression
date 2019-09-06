import sys
import numpy as np

def dataread(datapath):
    with open(datapath, "r") as file:
        lines = file.readlines()
        # print type(lines)
        labels = []
        # labels.append(1)
        feats1234 = []
        for line in lines:

            feats123 = []
            feats123.append(1)
            label= line.split("\t")
            labels = labels + [int(label[0])]
            for j in label[1:]:
                 feats123=feats123+[int(j.split(":")[0])]
            feats1234.append(feats123)
        # print feats1234, labels
        # print feats1234
        return feats1234, labels

def sdx(indexfile,weight):
    dprod=0.0
    count=0
    for i in indexfile:
        if count==0:
            dprod = dprod + weight[0]
            count += 1
        else:
            dprod = dprod + weight[i+1]

    return dprod

def train(fvector, labels, epochs,max):
    alfa1=0.1
    weights = np.zeros(max)
    for epoch in range(epochs):
        for findex, label in zip(fvector, labels):
            featurevec = np.zeros(max)
            findex2 = [x + 1 for x in findex[1:]]
            featurevec[0] = 1.0
            featurevec[findex2] = 1.0
            # print featurevec
            # # featurevec = [a]+featurevec
            # print findex
            # print findex2
            # print featurevec
            # print featurevec[22]

            dod121 = sdx(findex, weights)
            exp1= np.exp(dod121) / (1 + np.exp(dod121)) # - dosent work
            # print exp1
            weights += alfa1 * featurevec*(label - exp1)
    return weights

def prediction12(fvector1, labels, weights, outfile):
    predict1 = 0
    # print weights
    with open(outfile, "w") as file1:
        for findex1, label in zip(fvector1, labels):
            dod121 = sdx(findex1, weights)
            exp1 = 1 / (1 + np.exp(-dod121))
            if exp1 > 0.5 :
                Y1=1
            else:
                Y1=0
            if Y1 == label:
                predict1 += 1
            file1.write('{}\n'.format(Y1))
        file1.close()
    return float (predict1 / float(len(fvector1)))

def get_dic(dic_path):
    with open(dic_path) as openfile:
        dic={}
        for line in openfile:
            word, ind = line.split(" ")
            dic[word] = ind
        return dic, ind
# dictinput = dictinput




def main():
    ftraininput = sys.argv[1]
    fvalidationinput = sys.argv[2]
    formattedtestinput = sys.argv[3]
    dictinput = sys.argv[4]
    trainout = sys.argv[5]
    testout = sys.argv[6]
    metricsout = sys.argv[7]
    epochs = int(sys.argv[8])

    dic, max1 = get_dic(dictinput)
    max = int(max1) + 2
    # max = int(max1) + 2
# ftraininput = "model1_formatted_train.tsv"
# formattedtestinput = "model1_formatted_test.tsv"

# fvalidationinput = sys.argv[2]
# # formattedtestinput = sys.argv[3]
#
# trainout = "trainout.labels"
# testout = "testout.labels"
# metricsout = "metrics.labels"
# epochs = 60

    x,y=dataread(ftraininput)
    x1,y1= dataread(formattedtestinput)
    wgt = train(x, y, epochs,max)
    # print wgt
    # print wgt
    # outfile = "outputfile8.tsv"
    trainerror1 = prediction12(x, y, wgt, trainout)
    testerror1 = prediction12(x1, y1, wgt, testout)

    with open(metricsout, "w") as file122:
        file122.write("error(train): {}\n".format(1 - trainerror1))
    #     file122.write("error(test): {}\n".format(1-1/float(13333)))
        file122.write("error(test): {}\n".format(1-testerror1))
    file122.close()


if __name__ == "__main__":
    main()



