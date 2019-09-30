import pygame
import numpy as np

def preprocess(s):
    rgb = s.img
    r, g, b = rgb[0, :, :], rgb[1, :, :], rgb[2, :, :]
    s_image = (0.2989 * r + 0.5870 * g + 0.1140 * b) / 256
    for i in range(64 // 2):
        backup = s_image[64 - 1 - i].copy()
        s_image[64 - 1 - i] = s_image[i]
        s_image[i] = backup
    return s_image


def up_state(s_img_, img_, step, IMAGE_NUM):
    if step < IMAGE_NUM - 1:
        s_img_[step+1] = img_
    else:
        s_img_ = np.vstack((s_img_[1:, :], img_.reshape(1, 64, 64)))
    return s_img_


class Console():
    def __init__(self):
        pygame.init()
        screen = pygame.display.set_mode((300, 200))
        pygame.display.set_caption('pygame event')

    def keyboard(self, a, manul, online_draw, record=False):

        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_LSHIFT:
                    manul = not manul
                    print('Manul', manul)
                if event.key == pygame.K_LCTRL:
                    online_draw = not online_draw
                    print("Draw", online_draw)
                # if event.key == pygame.K_DOWN:
                #     a = 5
                #     break
                # elif event.key == pygame.K_UP:
                #     a = 4
                #     break
                elif event.key == pygame.K_LEFT:
                    a = max(a - 2, 0)

                elif event.key == pygame.K_RIGHT:
                    a = min(a + 2, 16)

                elif event.key == pygame.K_SPACE:
                    record = True
        if manul:
            if a > 8:
                a = a - 1
            if a < 8:
                a = a + 1
        return manul, online_draw, a, record


