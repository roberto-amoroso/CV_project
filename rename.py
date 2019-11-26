# Pythono3 code to rename multiple 
# files in a directory or folder 

# importing os module 
import os 

# Function to rename multiple files 
def main(): 
    i = 0
    dir = "mask"
    for filename in os.listdir(dir): 
        if ' copia' in filename:
            # dst = filename.replace(' copia', '')
            
            src = os.path.join(r'C:\Users\Vittorio\Desktop\Lab 07\\' + dir.replace('/', '\\'), filename)
            dst = os.path.join(r'C:\Users\Vittorio\Desktop\Lab 07\\' + dir.replace('/', '\\'), filename.replace(' copia', ''))
            # src = os.path.join(dir, filename)
            # dst = os.path.join(dir, filename.replace(' copia', ''))
            print(src + " -> " + dst)
            os.replace(src, dst) 
            i += 1
    print("Renamed " + str(i) + " files")

# Driver Code 
if __name__ == '__main__':
    main() 
