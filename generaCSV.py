"""Este script genera ficheros .csv con todas las rutas a las imágenes en cada conjunto de train, test y validación.
PRIMERO se debe dividir el dataset en subconjuntos y LUEGO generar los csvs """


import os
import pandas as pd


def genera_csv(path_subset, nombre_fichero) -> None:
    contador = 0
    carpetas = sorted(os.listdir(path_subset))

    names = []
    clases = []

    for carpeta in carpetas:
        path_carpeta = path_subset + carpeta
        images_in_carpeta = os.listdir(path_carpeta)
        for image in images_in_carpeta:
            path_image = path_carpeta + '/' + image
            names.append(path_image)
            clases.append(contador)

        contador = contador + 1

    data = pd.DataFrame(names, clases)
    data = data.sample(frac=1).reset_index(drop=False)

    data.to_csv("./paths_csv/" + nombre_fichero, header=None)

    print("csv " + nombre_fichero + " hecho")


def main():
    path_tr = "OnceClases/"
    path_test = "test_11/"
    path_val = "val_11/"

    genera_csv(path_tr, "images_tr_11.csv")
    genera_csv(path_test, "images_test_11.csv")
    genera_csv(path_val, "images_val_11.csv")


main()
