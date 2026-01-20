def printwrite(filename, *log):
    file = open(filename, "a")
    for i in log:
        print(i)
        file.write(str(i))
    file.write('\n')
    file.close()
    return
