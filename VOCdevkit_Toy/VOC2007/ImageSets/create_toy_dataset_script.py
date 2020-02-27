import imageio
import numpy as np
# import ipdb
import os


def main():
    DIR = '/Users/harsh/Desktop/CMU_Sem_4/VLR/Assignments/hw1/VOCdevkit_Toy/VOC2007/ImageSets/Main/'
    for file in os.listdir(DIR):
        if file.endswith(".txt"):
            print(file)
            with open(DIR+file,"r") as fp:
                # ipdb.set_trace()
                fp.seek(0)
                iter_1 = fp.readlines()
                fp.close()
            with open(DIR+file,"w") as fp:
                for line in iter_1:
                    # print(line)
                    test = line.split()
                    # if(test[0][0]=='\x00'):
                    #     break
                    if(int(test[0])<=15):
                        fp.write(line)
                        # continue
                    else:
                        break
                # ipdb.set_trace()
                fp.close()

if __name__ == '__main__':
    main()


