import os

def clear_console():
    os.system('cls' if os.name == 'nt' else 'clear')

def run_script(script_name, folder_name, file_extension):
    clear_console()
    print(f"Running script: {script_name}\n")
    os.system(f"python {script_name} {folder_name} {file_extension}")

def main_menu():
    while True:
        clear_console()
        print("Welcome to the object tracking methods validation program!")
        print("1. Run meanshift")
        print("2. Run camshift")
        print("3. Run Lucas-Kanade")
        print("0. Exit program")
        choice = input("Choose a method: ")

        if choice == "1":
            folder_name = input("Enter the folder name: ")
            file_extension = input("Enter the file extension: ")
            run_script("meanshift.py", folder_name, file_extension)
        elif choice == "2":
            folder_name = input("Enter the folder name: ")
            file_extension = input("Enter the file extension: ")
            run_script("camshift.py", folder_name, file_extension)
        elif choice == "3":
            folder_name = input("Enter the folder name: ")
            file_extension = input("Enter the file extension: ")
            run_script("lukas_kanade.py", folder_name, file_extension)
        elif choice == "0":
            clear_console()
            print("Exiting the program.")
            break
        else:
            input("Invalid option. Press Enter to try again.")


if __name__ == "__main__":
    main_menu()

