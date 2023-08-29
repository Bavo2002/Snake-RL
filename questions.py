import pycodestyle


# Play again?
def play_again():
    while True:
        answer = input('\nDo you want to play again?\n')

        if answer.lower().startswith('y'):
            return True

        elif answer.lower().startswith('n'):
            print('\nThanks for playing!')
            return False

        else:
            print("\nPlease only type 'yes' or 'no'.")


# single_player or multi_player?
def choose_mode():
    while True:
        answer = input("\nDo you want to play with 1 or 2 players? (type '1' or '2')\n")

        if answer == '1' or answer == '2':
            return int(answer)

        elif answer.startswith('NN'):
            try:
                return int(answer[2:].strip()) * -1
            except ValueError:
                print("\nPlease only type '1' or '2'.")

        else:
            print("\nPlease only type '1' or '2'.")


if __name__ == '__main__':
    # Run PEP 8 check on this file
    pycodestyle.StyleGuide().check_files([__file__])
