from simulation import *




def main():
    running = True

    g = Game(False, True, False, False, True)

    screen = display.set_mode((WIDTH, HEIGHT))
    screen.fill((100, 100, 200))
    g.display(screen)
    display.flip()
    while running:
        for action in event.get():
            if action.type == QUIT:
                running = False

        
        screen.fill((100, 100, 200))

        g.doTurn(display, screen)

        g.display(screen)

        display.flip()


main()

quit()
                







        
