from matplotlib import pyplot as plt


if __name__ == '__main__':
    B = .25
    B2 = .98

    # Example 3: Read all lines into a list
    with open('temps.txt', 'r') as file:
        y = [float(line.strip()) for line in file]
        x = [i + 1 for i in range(len(y))]

    guesses = [0]
    vt2 = [0]
    for i in range(len(y)):
        guesses.append(B * guesses[i] + (1-B) * y[i])
        vt2.append(B2 * vt2[i] + (1-B2) * y[i])

    v2_corr = [vt2[t]/(1-B2**t) for t in range(1, len(vt2))]

    # plt.ion()
    plt.plot(x, y, marker='o')
    plt.plot(x, guesses, color='red', marker='o')
    plt.plot(x, v2_corr, color='green', marker='o')
    plt.show()