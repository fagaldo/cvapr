import os

def clear_console():
    os.system('cls' if os.name == 'nt' else 'clear')

def run_script(script_name):
    clear_console()
    print(f"Running script: {script_name}\n")
    os.system(f"python {script_name}")

def main_menu():
    while True:
        clear_console()
        print("Welcome to the object tracking program!")
        print("1. Run meanshift")
        print("2. Run camshift")
        print("3. Run Lucas-Kanade")
        print("0. Exit program")
        choice = input("Choose an option: ")

        if choice == "1":
            run_script("meanshift.py")
        elif choice == "2":
            run_script("camshift.py")
        elif choice == "3":
            run_script("lukas_kanade.py")
        elif choice == "0":
            clear_console()
            print("Exiting the program.")
            break
        else:
            input("Invalid option. Press Enter to try again.")

if __name__ == "__main__":
    main_menu()
