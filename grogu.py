import os

if os.path.exists("grogu.py"):
    os.remove("./test.txt")
    print("True")
else:
    print("False")