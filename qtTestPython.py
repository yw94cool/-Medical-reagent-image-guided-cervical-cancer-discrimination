import sys

def square(m):
    return int(m)**2

if __name__ == '__main__':
    filedir = sys.argv[1]
    print(filedir)
    temdir = list(filedir)
    temdir.reverse()
    lo = temdir.index('\\')
    infile = open(filedir, 'r')
    outfile = open(filedir[0:-lo]+'RRResult.txt', 'w')
    for each_line in infile:
        a = str(square(each_line))
        outfile.write(a)
        outfile.write('\n')
    infile.close()
    outfile.close()