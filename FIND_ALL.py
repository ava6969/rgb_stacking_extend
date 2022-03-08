import glob
import tensorboardX
if __name__ == '__main__':

    b = tensorboardX.SummaryWriter()
    b.add_scalar('test', 0)
    b.add_scalar('test', 1)
    for cmake_list in glob.glob('/home/dewe/sam/**/CMakeLists.txt', recursive=True):
        printed_path = False
        with open(cmake_list) as cmake_list_file:
            for line in cmake_list_file.readlines():
                line = line.strip().rstrip()
                if not line.startswith('#') and (line.startswith('find_package') or ('.a'in line) or ('.so'  in line)):
                    if not printed_path:
                        print('path: ', cmake_list.replace('dewe', 'ava6969'))
                        printed_path = True
                    print('\t- ', line)

    b.close()

