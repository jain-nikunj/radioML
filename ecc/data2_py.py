from generate_data import *
import numpy as np
import pickle
import os


def make(k,p,e,l,r,n):
    try:
        message_seqs, encoded_seqs = generate_encode_random_sequences(l, k, n, r, p)
        noisy_seqs = simulate_bsc(encoded_seqs, p=e)
        viterbi_decoded_seqs = viterbi_decode_sequences(noisy_seqs, l, rate=r)

        dataset = {
            "message_seqs": message_seqs,
            "encoded_seqs": encoded_seqs,
            "noisy_seqs": noisy_seqs,
            "viterbi_decoded_seqs": viterbi_decoded_seqs
        }
        return dataset
    except Exception as e:
        print("\nfailed on (k,p,e,l,r,n) = ", str((k,p,e,l,r,n)))
        print("error: ", e)

def save(dataset, filename):
    if "data" not in os.listdir("."):
        os.mkdir("data")
    path = "data/{0}".format(filename)
    with open(path, 'wb') as output:
        pickle.dump(dataset, output)

def exist(filename):
    if "data" not in os.listdir("."):
        return False
    components = filename.split("/")
    directory, file = components[:-1], components[-1]
    directory = "/".join(directory)
    directory = "data/{0}".format(directory)
    return file in os.listdir(directory)


if __name__ == '__main__':

    all_datasets = dict()
    get_rate = lambda x: 0.5 if x < 0.1 else 1/3
    n = 25000
    l = 3
    p = 0.5

    k_range = np.hstack([np.arange(5,51,5),np.arange(55,201,10)])
    e_range = np.arange(0.05, 0.21, 0.1)

    i = 0
    total = len(k_range) * len(e_range)

    for k in k_range:
        for e in e_range:
            i += 1

            print("generating (k, e) = {0}, {1:.3f}".format(k, e))

            if i % 5 == 0:
                print("progress: {0:.2f}%".format(i / total * 100))
                if i % 10 != 0:
                    print()

            r = get_rate(e)
            all_datasets[(k,e)] = make(k,p,e,l,r,n)

            if i % 10 == 0:
                filename = "dataset_n{0}.pkl".format(n)
                print("updating mega data pkl {0}\n".format(filename))
                save(all_datasets, filename)



    print("progress: 100%")

    print("saving mega data pkl", filename)
    save(all_datasets, filename)


