import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import cv2
from PIL import Image, ImageDraw
import torch


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Square:
    A = [0, 0]
    B = [0, 0]
    C = [0, 0]
    D = [0, 0]
    A_data = 0.0
    B_data = 0.0
    C_data = 0.0
    D_data = 0.0

    def GetCaseId(self, threshold):
        caseId = 0
        if self.A_data >= threshold:
            caseId |= 1
        if self.B_data >= threshold:
            caseId |= 2
        if self.C_data >= threshold:
            caseId |= 4
        if self.D_data >= threshold:
            caseId |= 8

        return caseId

    def GetLines(self, Threshold):
        lines = []
        caseId = self.GetCaseId(Threshold)

        if caseId in (0, 15):
            return []

        if caseId in (1, 14, 10):
            pX = (self.A[0] + self.B[0]) / 2
            pY = self.B[1]
            qX = self.D[0]
            qY = (self.A[1] + self.D[1]) / 2

            line = (pX, pY, qX, qY)

            lines.append(line)

        if caseId in (2, 13, 5):
            pX = (self.A[0] + self.B[0]) / 2
            pY = self.A[1]
            qX = self.C[0]
            qY = (self.A[1] + self.D[1]) / 2

            line = (pX, pY, qX, qY)

            lines.append(line)

        if caseId in (3, 12):
            pX = self.A[0]
            pY = (self.A[1] + self.D[1]) / 2
            qX = self.C[0]
            qY = (self.B[1] + self.C[1]) / 2

            line = (pX, pY, qX, qY)

            lines.append(line)

        if caseId in (4, 11, 10):
            pX = (self.C[0] + self.D[0]) / 2
            pY = self.D[1]
            qX = self.B[0]
            qY = (self.B[1] + self.C[1]) / 2

            line = (pX, pY, qX, qY)

            lines.append(line)

        elif caseId in (6, 9):
            pX = (self.A[0] + self.B[0]) / 2
            pY = self.A[1]
            qX = (self.C[0] + self.D[0]) / 2
            qY = self.C[1]

            line = (pX, pY, qX, qY)

            lines.append(line)

        elif caseId in (7, 8, 5):
            pX = (self.C[0] + self.D[0]) / 2
            pY = self.C[1]
            qX = self.A[0]
            qY = (self.A[1] + self.D[1]) / 2

            line = (pX, pY, qX, qY)

            lines.append(line)

        return lines


def marching_squares(xVector, yVector, Data, threshold):
    linesList = []

    Height = len(Data)  # rows
    Width = len(Data[1])  # cols

    if (Width == len(xVector)) and (Height == len(yVector)):
        squares = np.full((Height - 1, Width - 1), Square())

        sqHeight = squares.shape[0]  # rows count
        sqWidth = squares.shape[1]  # cols count

        for j in range(sqHeight):  # rows
            for i in range(sqWidth):  # cols
                a = Data[j + 1, i]
                b = Data[j + 1, i + 1]
                c = Data[j, i + 1]
                d = Data[j, i]
                A = [xVector[i], yVector[j + 1]]
                B = [xVector[i + 1], yVector[j + 1]]
                C = [xVector[i + 1], yVector[j]]
                D = [xVector[i], yVector[j]]

                squares[j, i].A_data = a
                squares[j, i].B_data = b
                squares[j, i].C_data = c
                squares[j, i].D_data = d

                squares[j, i].A = A
                squares[j, i].B = B
                squares[j, i].C = C
                squares[j, i].D = D

                list_toadd = squares[j, i].GetLines(threshold)
                linesList = linesList + list_toadd
    else:
        raise AssertionError

    return [linesList]


def get_img_marching_squares(sdf_numpy, threshold=0.0):
    n_x, n_y = sdf_numpy.shape
    x = [i for i in range(n_x)]
    y = [i for i in range(n_y)]

    im = Image.new("RGB", (n_x, n_y), (128, 128, 128))
    collection = marching_squares(x, y, sdf_numpy, threshold)

    draw = ImageDraw.Draw(im)

    for ln in collection:
        for toup in ln:
            draw.line(toup, fill=(255, 255, 0), width=1)

    return im


def display_multiple_shapes_sdf(net, resolution=200, figsize=(14, 5)):
    fig, plots = plt.subplots(2, 5, figsize=figsize)

    for time in range(10):
        i = time // 5
        j = time - i * 5
        t = 0.1 * time
        coords = torch.stack(
            list(
                torch.meshgrid(
                    [torch.linspace(-1, 1, resolution, device=device)] * 2,
                    indexing="xy",
                )
            ),
            dim=2,
        ).reshape(-1, 2)
        coords = torch.concat(
            (
                coords,
                t * torch.ones((coords.shape[0], 1), device=device),
            ),
            dim=1,
        )

        coords.requires_grad = True
        sdf = net(coords).reshape(resolution, resolution)
        numpy_sdf = sdf.detach().cpu().numpy()
        rec_img = get_img_marching_squares(numpy_sdf)
        plots[i, j].axis("off")
        plots[i, j].imshow(rec_img)
    plt.show()
