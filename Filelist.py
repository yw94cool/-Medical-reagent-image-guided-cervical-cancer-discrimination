import os


def generate(dir, label, name):
    files = os.listdir(dir)
    files.sort()

    listText = open('D:\\AI4Medical\\project0315\\data180703train' + '\\' + name, 'w')

    for file in files:
        fileType = os.path.split(file)
        if fileType[1] == '.txt':
            continue
        path = dir+'\\'+file + ' ' + str(int(label)) + '\n'
        listText.write(path)
    listText.close()



if __name__ == '__main__':
    generate('D:\\AI4Medical\\project0315\\data180703train\\yin', 0, 'trainlist0.txt')
    generate('D:\\AI4Medical\\project0315\\data180703train\\yang', 1, 'trainlist1.txt')
    generate('D:\\AI4Medical\\project0315\\data180703train\\yisi', 2, 'trainlist2.txt')
    trainlist = open('D:\\AI4Medical\\project0315\\Train3.00\\train_list.txt', 'w')
    for filename in ['trainlist0.txt','trainlist1.txt','trainlist2.txt']:
        filepath = 'D:\\AI4Medical\\project0315\\data180703train\\'+filename
        for line in open(filepath):
            trainlist.write(line)
    trainlist.close()

