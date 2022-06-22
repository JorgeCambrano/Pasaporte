# Importamos las dependencias del proyecto
import sys
from argparse import ArgumentParser

import cv2
import imutils
import numpy as np
import pytesseract
from imutils.contours import sort_contours

# Definimos los argumentos de entrada del script.
argument_parser = ArgumentParser()
argument_parser.add_argument('-i', '--image', type=str, required=True, help='Ruta a la imagen de entrada.')
arguments = vars(argument_parser.parse_args())

pytesseract.pytesseract.tesseract_cmd = 'C:\Program Files\Tesseract-OCR\\tesseract.exe'


# Cargamos la imagen y la convertimos a escala de grises.
image = cv2.imread(arguments['image'])
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.imshow('Imagen', gray)
cv2.waitKey(0)

# Extraemos las dimensiones de la imagen
height, width = gray.shape

# Inicializamos un kernel rectangular y uno cuadrado
rectangular_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 7))
squared_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 21))

# Suavizamos la imagen aplicando un filtro Gaussiano de 3x3
gray = cv2.GaussianBlur(gray, (3, 3), 0)

# Aplicamos blackhat para encontrar las regiones oscuras sobre un fondo claro.
blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, rectangular_kernel)
cv2.imshow('Blackhat', blackhat)
cv2.waitKey(0)

# Calculamos el gradiente Scharr y escalamos resultado para que esté en el rango [0, 255]
gradient = cv2.Sobel(blackhat, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=-1)
gradient = np.absolute(gradient)
min_val, max_val = np.min(gradient), np.max(gradient)
gradient = (gradient - min_val) / (max_val - min_val)
gradient = (gradient * 255).astype('uint8')
cv2.imshow('Gradiente', gradient)
cv2.waitKey(0)

# Aplicamos la operación Clausura utilizando el kernel rectangular definido arriba.
# Esto lo hacemos para cerrar los espacios entre las letras, luego aplicamos thresholding
# automático usando Otsu.
gradient = cv2.morphologyEx(gradient, cv2.MORPH_CLOSE, rectangular_kernel)
thresholded = cv2.threshold(gradient, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
cv2.imshow('Clausura rectangular', thresholded)
cv2.waitKey(0)

# Llevamos a cabo otra clausura, pero esta vez usando el kernel cuadrado. Luego erosionamos la imagen
# para romper componentes conectadas indeseadas.
thresholded = cv2.morphologyEx(thresholded, cv2.MORPH_CLOSE, squared_kernel)
thresholded = cv2.erode(thresholded, None, iterations=2)
cv2.imshow('Clausura cuadrada', thresholded)
cv2.waitKey(0)

# Encontramos los contornos en la imagen procesada con Otsu. Después los ordenamos desde abajo
# hasta arriba (esto es porque la MRZ, que es la zona donde están los datos del pasaporte,
# siempre aparece en la parte de abajo).
contours = cv2.findContours(thresholded.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
contours = imutils.grab_contours(contours)
contours = sort_contours(contours, method='bottom-to-top')[0]

# Inicializamos el rectángulo asociado a la MRZ
mrz_bounding_box = None

# Iteramos sobre los contornos para hallar la MRZ
for contour in contours:
    x, y, w, h = cv2.boundingRect(contour)
    percent_width = w / float(width)
    percent_height = h / float(height)

    # Asumimos que encontramos la MRZ si el área del contorno ocupa más del 80% del ancho de la imagen
    # y más del 4% de su altura.
    if percent_height > 0.04 and percent_width > 0.8:
        mrz_bounding_box = (x, y, w, h)
        break

if mrz_bounding_box is None:
    print('No se pudo encontrar MRZ.')
    sys.exit(0)

# Añadimos padding al rectángulo de la MRZ
x, y, w, h = mrz_bounding_box
pad_x = int((x + w) * 0.03)
pad_y = int((y + h) * 0.03)
x, y = x - pad_x, y - pad_y

w, h = w + (pad_x * 2), h + (pad_y * 2)

# Extraemos la MRZ de la imagen
mrz = image[y:y + h, x:x + w]

# Usamos Tesseract para extraer el texto de la MRZ
mrz_text = pytesseract.image_to_string(mrz)
mrz_text = mrz_text.replace(' ', '')
print('MRZ:', mrz_text)

# Mostramos la MRZ
cv2.imshow('MRZ', mrz)
cv2.waitKey(0)

# Destruimos las ventanas creadas durante la ejecución del programa.
cv2.destroyAllWindows()
