from simulation import *




def main():
    running = True

    g = Game()

    screen = display.set_mode((WIDTH, HEIGHT))

    while running:
        for action in event.get():
            if action.type == QUIT:
                running = False

        g.display(screen)

        display.flip()


main()

quit()
                







        
