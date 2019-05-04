import csv
import numpy as np
import math

train_file = "train.csv"
test_file = "test.csv"
result_file ="result.csv"

def Normalization(x):
    mean = np.mean(x,axis=0)
    std = np.std(x,axis=0)
    for i in range(len(x)):
        for j in range(len(x[0])):
            if not std[j] == 0:
                x[i][j] = (x[i][j]-mean[j])/std[j]


    return x
def train():
    data = []
    for i in range(18):
        data.append([])

    n_row = 0

    train_filename = open(train_file,'r',encoding='big5')
    train_reader = csv.reader(train_filename,delimiter=',')
    for r in train_reader:
        if n_row != 0:
            for i in range(3,27):
                if r[i] != 'NR':
                    data[(n_row-1)%18].append(float(r[i]))
                else:
                    data[(n_row-1)%18].append(float(0))

        n_row+=1

    train_filename.close()

    input = []
    label = []

    for i in range(12):
        for j in range(471):
            input.append([])
            for k in range(18):
                for l in range(9):
                    input[471*i+j].append(data[k][480*i+j+l])

            label.append(data[9][480*i+j+9])
    input = Normalization(input)
    input = np.array(input)
    label = np.array(label)
    input = np.concatenate((np.ones((input.shape[0],1)),input),axis=1)

    w = np.zeros(len(input[0]))
    mt = np.zeros(len(input[0]))
    vt = np.zeros(len(input[0]))

    l_rate = 0.0001
    repeat = 400000
    b1 = 0.9
    b2 = 0.999
    e = 1e-8
    input_t = input.transpose()

    for i in range(1,repeat):
        diff = np.dot(input,w)
        loss = diff - label
        gra = 2.0 * np.dot(input_t,loss)
        mt = b1*mt + (1-b1)*gra
        vt = b2*vt + (1-b2)*(gra*gra)

        m_cap = mt/(1-(b1**i))
        v_cap = vt/(1-(b2**i))

        w = w-(l_rate*m_cap)/(np.sqrt(v_cap)+e)

        cost = np.sum(loss**2)/len(input)
        cost_a = math.sqrt(cost)
        print('iteration: %d | Cost: %.9f' % (i,cost_a))
    np.save('w_mt_vt.npy', [w, mt, vt])


def test():
    w_mt_vt = np.load('w_mt_vt.npy')
    w = w_mt_vt[0]
    mt = w_mt_vt[1]
    vt = w_mt_vt[3]


    test_input = []
    n_row = 0
    test_filename = open(test_file,'r')
    test_reader = csv.reader(test_filename,delimiter = ',')

    for r in test_reader:
        if n_row % 18 == 0:
            test_input.append([])
            for i in range(2,11):
                if r[i] != "NR":
                    test_input[n_row // 18].append(float(r[i]))
                else:
                    test_input[n_row // 18].append(0)
        else:
            for i in range(2,11):
                if r[i] != "NR":
                    test_input[n_row // 18].append(float(r[i]))
                else:
                    test_input[n_row // 18].append(0)
        n_row+=1
    test_filename.close()
    test_input = np.array(test_input)
    test_input = Normalization(test_input)
    test_input = np.concatenate((np.ones((test_input.shape[0],1)),test_input),axis=1)


    ans = []

    for i in range(len(test_input)):
        ans.append(['id_'+str(i)])
        a = np.dot(w,test_input[i])
        ans[i].append(a)

    ansfile = result_file
    ansfile_name = open(ansfile,'w+')
    s = csv.writer(ansfile_name,delimiter=',',lineterminator = "\n")
    s.writerow(['id','value'])
    for i in range(len(ans)):
        s.writerow(ans[i])

    ansfile_name.close()



if __name__ == "__main__":
    train()
