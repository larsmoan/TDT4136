

def print_table(n):
    for i in range(1,n+1):
        for j in range(1,n+1):
            print(i*j, end="\t")
        print("\n")

print_table(10)