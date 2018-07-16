import numpy as np
import csv

data = []

def test():
    global data
    results = []
    with open("prima-indians-diabetes.csv") as csvfile:
        reader = csv.reader(csvfile, quoting=csv.QUOTE_NONNUMERIC) # change contents to floats
        for row in reader: # each row is a list
            results.append(row)

        for row in results:
            label = row.pop()
            row = _find_structure(row)
            print(row)
            row.append(label)
            data.append(row)

        data = np.array(data)
        np.save('test_structure.npy', data)


def _find_structure(data):
    s_plus = data.copy()
    for n in data:
        for i, ni in enumerate(data):
            if n is not data[i]:
                s_plus.append(int(n) - int(ni))

    return s_plus

if __name__ == '__main__':
    test()
