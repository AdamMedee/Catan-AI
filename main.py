from simulation import *




def main():
    running = True

    g = Game(False, False, False, False, True)

    screen = display.set_mode((WIDTH, HEIGHT))
    g.display(screen)
    while running:
        for action in event.get():
            if action.type == QUIT:
                running = False

        
        

        g.doTurn()

        g.display(screen)

        display.flip()


main()

quit()
                







        
