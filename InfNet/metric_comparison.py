filenames = ['metric_prior_baseline-multi-inf-net.txt', 'metric_prior_multi-inf-net04.txt',
             'metric_prior_multi-inf-net05.txt']

values = [[], [], []]

differences = []
filename_to_write = 'metric_prior_multi.txt'

if __name__ == '__main__':
    for index, filename in enumerate(filenames):
        with open(filename, 'r') as f:
            value = f.read()
            lines = value.split('\n')
            for line in lines:
                try:
                    float_value = float(line)
                    values[index].append(float_value)
                except ValueError:
                    pass

    # compare different values
    for current_index in range(len(values[0])):
        current_value = ''
        if values[0][current_index] > values[1][current_index] and values[0][current_index] > values[2][current_index]:
            current_value = str(current_index).ljust(5) + ' ' + filenames[0].ljust(50) + \
                            f' {min(abs(values[0][current_index] - values[1][current_index]), abs(values[0][current_index] - values[2][current_index]))}'

        elif values[1][current_index] > values[0][current_index] and values[1][current_index] > values[2][current_index]:
            current_value = str(current_index).ljust(5) + ' ' + filenames[1].ljust(50) + \
                            f' {min(abs(values[1][current_index] - values[0][current_index]), abs(values[1][current_index] - values[2][current_index]))}'

        elif values[2][current_index] > values[0][current_index] and values[2][current_index] > values[1][current_index]:
            current_value = str(current_index).ljust(5) + ' ' + filenames[2].ljust(50) + \
                            f' {min(abs(values[2][current_index] - values[0][current_index]), abs(values[2][current_index] - values[1][current_index]))}'

        differences.append(current_value)

    with open(filename_to_write, 'w') as f:
        for difference in differences:
            f.write(difference + '\n')
