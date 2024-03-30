from simulation import *




def main():
    running = True

    g = Game(True, True, True, True, True, 3)

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
        if g.gameOver:
            running = False
        g.display(screen)

        display.flip()


main()

#quit()
                







        
